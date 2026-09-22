"""L1 P4 M1/M2 gate tests (l1-decode-plumbing-fusion-design-20260802.md section 5):
the decode epilogue-fusion promotion record is CLOSED by default, the loader never
infers promotion from a target string, the checked-in record promotes only the
measured M2 target (NV sm_120, the first fused consumer), and the fused answers
ride on the existing QK and flash admissions without changing their legacy
`admitted` routes."""
import json, pathlib

import pytest

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, load_decode_epilogue_fusion_promotion,
  _DECODE_EPILOGUE_FUSION_PROMOTED_TARGETS)
from tinygrad.llm.decode_kernels import Q6KGEMVRouteSpec, emit_q6k_gemv_kernel
from tinygrad.llm.flash_decode_attention import FlashDecodeAdmission, FlashDecodeCapability, FlashDecodeRouteConfig
from tinygrad.llm.qk_layout import Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission
from tinygrad.uop.ops import Ops, UOp


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_epilogue_fusion"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_epilogue_fusion_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_epilogue_fusion_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  promoted = load_decode_epilogue_fusion_promotion(p)
  assert ("NV", "sm_120") in promoted
  assert ("AMD", "gfx1100") not in promoted


def test_checked_in_record_promotes_only_the_measured_m2_target():
  # M2 lands the first fused consumer (coop in-kernel merge), so NV sm_120 is promoted;
  # every other target stays closed until its own runtime measurement.
  assert _DECODE_EPILOGUE_FUSION_PROMOTED_TARGETS == frozenset({("NV", "sm_120")})
  assert decode_epilogue_fusion_promoted(("NV", "sm_120"))
  assert not decode_epilogue_fusion_promoted(("AMD", "gfx1100"))
  assert not decode_epilogue_fusion_promoted((None, None))


def test_partial_in_kernel_rejected_by_validate():
  spec = Q6KGEMVRouteSpec(rows=1024, k=4096, route_family="q6k_partial", parts=4,
                          pos_axis="reduce", reduction="in_kernel")
  with pytest.raises(ValueError, match="not implemented for the q6k_partial family"):
    spec.validate()


def test_coop_in_kernel_single_warp_constraint():
  # AMD's row_tile=4 gives 4*16=64 lanes > 32: the in-kernel ladder cannot span two warps.
  too_wide = Q6KGEMVRouteSpec(rows=4096, k=12288, route_family="q6k_coop", row_tile=4,
                              reduction="in_kernel")
  with pytest.raises(ValueError, match="single warp"):
    too_wide.validate()
  nv_legal = Q6KGEMVRouteSpec(rows=4096, k=12288, route_family="q6k_coop", row_tile=2,
                              reduction="in_kernel")
  nv_legal.validate()


def test_coop_in_kernel_renders_through_hip_and_cuda_without_gpu():
  from tinygrad.helpers import Target
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  spec = Q6KGEMVRouteSpec(rows=4096, k=12288, route_family="q6k_coop", row_tile=2,
                          reduction="in_kernel")
  partials = UOp.placeholder((spec.rows,), dtypes.float32, 0)
  halfs = UOp.placeholder((spec.rows * (spec.k // Q6_K_BLOCK_ELEMS) * Q6K_HALFWORDS_PER_BLOCK,), dtypes.uint16, 1)
  x = UOp.placeholder((spec.k,), dtypes.float16, 2)
  ast = emit_q6k_gemv_kernel(spec)(partials, halfs, x)
  for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")), CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
    src = next(u.arg for u in to_program(ast, ren).src if u.op is Ops.SOURCE)
    assert "__shfl_xor_sync" in src or "ds_bpermute" in src
    assert "_inkernel" in spec.kernel_name


def test_qk_admission_fused_answer_does_not_change_legacy_admitted():
  cap = QKPrimitiveCapability(backend="NV", architecture="sm_120", wave_size=32, supports_warp_shfl_xor=True)
  open_adm = QKPrimitiveRouteAdmission(cap, True, epilogue_fusion_promoted=True)
  assert open_adm.admitted and open_adm.fusion_admitted
  closed_adm = QKPrimitiveRouteAdmission(cap, True)  # default flag False
  assert closed_adm.admitted and not closed_adm.fusion_admitted
  not_promoted = QKPrimitiveRouteAdmission(cap, False, epilogue_fusion_promoted=True)
  assert not not_promoted.admitted and not not_promoted.fusion_admitted
  no_cap = QKPrimitiveRouteAdmission(QKPrimitiveCapability(), True, epilogue_fusion_promoted=True)
  assert not no_cap.admitted and not no_cap.fusion_admitted


def test_flash_admission_fused_answer_does_not_change_legacy_admitted():
  cap = FlashDecodeCapability(supports_warp_shfl_xor=True, supports_fdot2=True)
  open_adm = FlashDecodeAdmission(True, cap, True, epilogue_fusion_promoted=True)
  assert open_adm.admitted and open_adm.fusion_admitted
  closed_adm = FlashDecodeAdmission(True, cap, True)
  assert closed_adm.admitted and not closed_adm.fusion_admitted
  shape_bad = FlashDecodeAdmission(False, cap, True, epilogue_fusion_promoted=True)
  assert not shape_bad.admitted and not shape_bad.fusion_admitted


def test_flash_route_evaluate_defaults_to_closed_fusion(tmp_path):
  cfg = FlashDecodeRouteConfig("c", "r", 32, 48, None, 1)
  cap = FlashDecodeCapability(supports_warp_shfl_xor=True, supports_fdot2=True)
  adm = cfg.evaluate(1, 32, 8, 128, cap, True)
  assert adm.admitted and not adm.fusion_admitted
  adm_open = cfg.evaluate(1, 32, 8, 128, cap, True, epilogue_fusion_promoted=True)
  assert adm_open.admitted and adm_open.fusion_admitted
