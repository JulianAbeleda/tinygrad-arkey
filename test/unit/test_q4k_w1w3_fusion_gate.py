"""Fused w1+w3 (gate/up) decode GEMV gate tests (q4k-w1w3-fused-qv-implementation-record-20260803.md):
the promotion record is CLOSED by default, the loader never infers promotion from a target string, the
checked-in record promotes only the measured NV sm_120 target, the fused answers ride on the existing QK
admissions without changing their legacy `admitted` routes, and the fused kernel renders through both the
HIP and CUDA renderers without a GPU."""
import json, pathlib

import pytest

from tinygrad import dtypes, Tensor
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.model_route_plan import (decode_q4k_w1w3_fusion_promoted,
  load_decode_q4k_w1w3_fusion_promotion, _DECODE_Q4K_W1W3_FUSION_PROMOTED_TARGETS)
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.decode_routes import q4k_gate_up_primitive_linear_call
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission
from tinygrad.uop.ops import Ops, UOp


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_q4k_w1w3_fusion"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_q4k_w1w3_fusion_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_q4k_w1w3_fusion_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  promoted = load_decode_q4k_w1w3_fusion_promotion(p)
  assert ("NV", "sm_120") in promoted
  assert ("AMD", "gfx1100") not in promoted


def test_checked_in_record_promotes_only_the_measured_nv_target():
  assert _DECODE_Q4K_W1W3_FUSION_PROMOTED_TARGETS == frozenset({("NV", "sm_120")})
  assert decode_q4k_w1w3_fusion_promoted(("NV", "sm_120"))
  assert not decode_q4k_w1w3_fusion_promoted(("AMD", "gfx1100"))
  assert not decode_q4k_w1w3_fusion_promoted((None, None))


def test_w1w3_admission_does_not_change_legacy_admitted():
  cap = QKPrimitiveCapability(backend="NV", architecture="sm_120", wave_size=32, supports_warp_shfl_xor=True)
  open_adm = QKPrimitiveRouteAdmission(cap, True, q4k_w1w3_fusion_promoted=True)
  assert open_adm.admitted and open_adm.w1w3_fusion_admitted
  closed_adm = QKPrimitiveRouteAdmission(cap, True)
  assert closed_adm.admitted and not closed_adm.w1w3_fusion_admitted
  not_promoted = QKPrimitiveRouteAdmission(cap, False, q4k_w1w3_fusion_promoted=True)
  assert not not_promoted.admitted and not not_promoted.w1w3_fusion_admitted
  no_cap = QKPrimitiveRouteAdmission(QKPrimitiveCapability(), True, q4k_w1w3_fusion_promoted=True)
  assert not no_cap.admitted and not no_cap.w1w3_fusion_admitted


def test_w1w3_fused_kernel_names():
  quad = q4k_g3_lanemap_gemv_w1w3_kernel(12288, 4096, load_style="quad")
  scalar = q4k_g3_lanemap_gemv_w1w3_kernel(12288, 4096, load_style="scalar")
  out = UOp.placeholder((12288,), dtypes.float32, 0)
  gw = UOp.placeholder((12288 * 16 * 36,), dtypes.uint32, 1)
  uw = UOp.placeholder((12288 * 16 * 36,), dtypes.uint32, 2)
  x = UOp.placeholder((4096,), dtypes.float16, 3)
  assert quad(out, gw, uw, x).arg.name == "q4k_g3_lanemap_gemv_w1w3qv_12288_4096"
  assert scalar(out, gw, uw, x).arg.name == "q4k_g3_lanemap_gemv_w1w3fused_12288_4096"


def test_w1w3_quad_geometry_rejects_unsupported_shapes():
  with pytest.raises(ValueError, match="rows % 16"):
    q4k_g3_lanemap_gemv_w1w3_kernel(4008, 4096, load_style="quad")
  with pytest.raises(ValueError, match="blocks_per_group"):
    q4k_g3_lanemap_gemv_w1w3_kernel(12288, 2048, load_style="quad")


def test_w1w3_fused_renders_through_hip_and_cuda_without_gpu():
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  for style in ("quad", "scalar"):
    kernel = q4k_g3_lanemap_gemv_w1w3_kernel(12288, 4096, load_style=style)
    out = UOp.placeholder((12288,), dtypes.float32, 0)
    gw = UOp.placeholder((12288 * 16 * 36,), dtypes.uint32, 1)
    uw = UOp.placeholder((12288 * 16 * 36,), dtypes.uint32, 2)
    x = UOp.placeholder((4096,), dtypes.float16, 3)
    ast = kernel(out, gw, uw, x)
    for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")),
                CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
      src = next(u.arg for u in to_program(ast, ren).src if u.op is Ops.SOURCE)
      assert "__shfl_xor_sync" in src or "ds_bpermute" in src
    if style == "quad":
      src = next(u.arg for u in to_program(ast, CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)).src if u.op is Ops.SOURCE)
      assert "uint4" in src and "__syncthreads" in src


class _FakeQ4K:
  """Minimal double of an installed Q4K primitive: binds only the legacy decode shape."""
  def __init__(self, admitted=True, w1w3=True, bias=None, decode_enabled=True, n=12288, k=4096):
    self.route_admission = QKPrimitiveRouteAdmission(
      QKPrimitiveCapability("NV", "sm_120", 32, True), True, q4k_w1w3_fusion_promoted=w1w3) if admitted else \
      QKPrimitiveRouteAdmission(QKPrimitiveCapability(), False)
    self.bias, self.decode_enabled = bias, decode_enabled
    self.out_features, self.in_features = n, k
    self.q4k_storage = type("S", (), {"mode": "sidecar", "words": Tensor.empty(n * 16 * 36, dtype=dtypes.uint32)})()


def test_gate_up_call_falls_back_when_either_linear_not_admitted():
  x = Tensor.empty(1, 1, 4096, dtype=dtypes.float16)
  for gate, up in [( _FakeQ4K(w1w3=False), _FakeQ4K()),
                   ( _FakeQ4K(), _FakeQ4K(w1w3=False)),
                   ( _FakeQ4K(admitted=False), _FakeQ4K())]:
    hit = []
    out = q4k_gate_up_primitive_linear_call(gate, up, x, fallback=lambda: hit.append(1) or x)
    assert hit == [1] and out is x


def test_gate_up_call_falls_back_on_shape_mismatch():
  x = Tensor.empty(1, 1, 4096, dtype=dtypes.float16)
  gate, up = _FakeQ4K(n=12288), _FakeQ4K(n=4096)
  hit = []
  out = q4k_gate_up_primitive_linear_call(gate, up, x, fallback=lambda: hit.append(1) or x)
  assert hit == [1] and out is x


def test_gate_up_call_falls_back_on_bias_or_multi_token():
  x = Tensor.empty(1, 1, 4096, dtype=dtypes.float16)
  gate, up = _FakeQ4K(bias=Tensor.empty(12288)), _FakeQ4K()
  hit = []
  out = q4k_gate_up_primitive_linear_call(gate, up, x, fallback=lambda: hit.append(1) or x)
  assert hit == [1] and out is x
  gate2, up2 = _FakeQ4K(), _FakeQ4K()
  x2 = Tensor.empty(1, 2, 4096, dtype=dtypes.float16)
  hit2 = []
  out2 = q4k_gate_up_primitive_linear_call(gate2, up2, x2, fallback=lambda: hit2.append(1) or x2)
  assert hit2 == [1] and out2 is x2
