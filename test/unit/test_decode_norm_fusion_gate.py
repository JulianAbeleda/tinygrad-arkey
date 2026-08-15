"""L1 M3 fused decode RMSNorm gate tests (l1-decode-plumbing-fusion-design-20260802.md section 6,
m3-fused-norm-measurement-record-20260802.md): the norm-fusion promotion record is CLOSED for every
target (measured non-landing), independently of the M2 decode-epilogue record that stays promoted on
NV sm_120, and the fused emitter validates and renders through the same non-GPU pipeline as the other
decode kernels."""
import json, pathlib

import pytest

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, decode_norm_fusion_promoted,
  load_decode_norm_fusion_promotion, _DECODE_NORM_FUSION_PROMOTED_TARGETS)
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops, UOp


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_norm_fusion"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_norm_fusion_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_norm_fusion_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  assert load_decode_norm_fusion_promotion(p) == frozenset({("NV", "sm_120")})


def test_checked_in_norm_record_promotes_nothing_and_m2_record_stays():
  # M3 measured non-landing (boundary copies regress the M2 baseline): the norm record must promote
  # nothing, while the M2 decode-epilogue record keeps its measured NV sm_120 opt-in.
  assert _DECODE_NORM_FUSION_PROMOTED_TARGETS == frozenset()
  assert not decode_norm_fusion_promoted(("NV", "sm_120"))
  assert not decode_norm_fusion_promoted(("AMD", "gfx1100"))
  assert not decode_norm_fusion_promoted((None, None))
  assert decode_epilogue_fusion_promoted(("NV", "sm_120"))


def test_spec_validate_accepts_production_shapes():
  for spec in (
    DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=16, weight_dtype=dtypes.float16, out_dtype=dtypes.float16),
    DecodeRMSNormSpec(rows=32, dim=128, eps=1e-6, weight_dtype=dtypes.float16, out_dtype=dtypes.float32),
    DecodeRMSNormSpec(rows=8, dim=128, eps=1e-6, weight_dtype=dtypes.float16, out_dtype=dtypes.float32, x_rank=3),
  ):
    spec.validate()


def test_spec_validate_rejects_bad_contracts():
  bad = [
    (dict(rows=0, dim=4096, eps=1e-6), "rows>=1"),
    (dict(rows=1, dim=100, eps=1e-6), "dim >= lane_width"),
    (dict(rows=1, dim=4096, eps=1e-6, warps_per_row=7), r"dim % \(lane_width"),
    (dict(rows=1, dim=4096, eps=0.0), "eps>0"),
    (dict(rows=1, dim=4096, eps=1e-6, x_rank=4), r"x_rank in \(1, 2, 3\)"),
    (dict(rows=1, dim=4096, eps=1e-6, out_dtype=dtypes.float64), "out_dtype"),
  ]
  for kw, msg in bad:
    with pytest.raises(ValueError, match=msg):
      DecodeRMSNormSpec(**kw).validate()


def test_fused_norm_renders_through_hip_and_cuda_without_gpu():
  ren_hip = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  ren_cuda = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)
  for spec in (
    DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=16, weight_dtype=dtypes.float16, out_dtype=dtypes.float16),
    DecodeRMSNormSpec(rows=32, dim=128, eps=1e-6, weight_dtype=dtypes.float16, out_dtype=dtypes.float32),
  ):
    numel, dim = spec.rows * spec.dim, spec.dim
    out = UOp.placeholder((numel,), spec.out_dtype, 0)
    x = UOp.placeholder((numel,), dtypes.float32, 1)
    w = UOp.placeholder((dim,), dtypes.float16, 2)
    ast = emit_decode_rmsnorm_kernel(spec)(out, x, w)
    for ren in (ren_hip, ren_cuda):
      src = next(u.arg for u in to_program(ast, ren).src if u.op is Ops.SOURCE)
      assert src, f"empty rendered source for {spec.kernel_name} on {ren}"
      if spec.warps_per_row == 1:
        assert "__shfl_xor_sync" in src or "ds_bpermute" in src
      else:
        assert "__syncthreads" in src or "barrier" in src
    assert spec.kernel_name.startswith("decode_rmsnorm_")
