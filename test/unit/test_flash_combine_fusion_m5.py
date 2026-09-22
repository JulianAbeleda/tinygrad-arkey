"""L1 M5 unit tests (l1-decode-plumbing-fusion-design-20260802.md section 6.5, row E_32_32_4_0a5e):
flash-decode combine fp16 absorption -- spec validation, admission wiring, fused kernel
render arms, and legacy byte-identity guarantees."""
import hashlib

import pytest

from tinygrad import dtypes
from tinygrad.helpers import Target
from tinygrad.llm.flash_decode_attention import (FlashCombineSpec, FlashDecodeAdmission, FlashDecodeCapability,
  FlashDecodeRouteConfig, describe_flash_decode_attention, flash_fused_gmax_combine_kernel)
from tinygrad.uop.ops import Ops, UOp


# ── spec validation ──────────────────────────────────────────────────────────

def test_combine_spec_validation():
  spec = FlashCombineSpec(Hd=128, Hq=32, split_count=48)
  spec.validate()
  fp16 = FlashCombineSpec(Hd=128, Hq=32, split_count=48, output_fp16=True)
  fp16.validate()
  with pytest.raises(ValueError, match="positive"):
    FlashCombineSpec(Hd=0, Hq=32, split_count=48).validate()
  with pytest.raises(ValueError, match="stride must be >= 1"):
    FlashCombineSpec(Hd=128, Hq=32, split_count=48, stride=0).validate()
  with pytest.raises(ValueError, match="Hd%32"):
    flash_fused_gmax_combine_kernel(100, 32, 48)


def test_describe_defaults_to_fp32_combine():
  spec = describe_flash_decode_attention(32, 128, 8, 4608, 48, fused_combine=True)
  assert isinstance(spec.combine, FlashCombineSpec)
  assert not spec.combine.output_fp16
  assert spec.combine.kernel_name == "flash_fused_gmax_combine_32_128"


def test_describe_combine_fp16_flag():
  spec = describe_flash_decode_attention(32, 128, 8, 4608, 48, fused_combine=True, combine_fp16=True)
  assert spec.combine.output_fp16
  assert spec.emitted_kernel_names == (
    "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128",
    "flash_fused_gmax_combine_f16_32_128")


# ── legacy byte-identity ─────────────────────────────────────────────────────

def _combine_uop(output_fp16: bool = False):
  spec = FlashCombineSpec(Hd=128, Hq=32, split_count=48, output_fp16=output_fp16)
  out = UOp.placeholder((32 * 128,), dtypes.float16 if output_fp16 else dtypes.float32, 0)
  pout = UOp.placeholder((32 * 48 * 130,), dtypes.float32, 1)
  return spec.emit()(out, pout)


def test_legacy_combine_name_and_digest_preserved():
  # Pinned from the M2-on NV baseline (flash_decode_attention.py FlashCombineSpec.emit,
  # Hq=32 Hd=128 S=48): the M5 fp16 variant must not move the legacy combine.
  uop = _combine_uop()
  assert uop.arg.name == "flash_fused_gmax_combine_32_128"
  assert hashlib.sha256(repr(uop.key).encode()).hexdigest() == \
    "560ce2902832f4864e2776673061ca71f91acd685d4ca81e991a2bbade0e8fdf"


def test_fp16_variant_has_new_name_and_distinct_digest():
  legacy, variant = _combine_uop(), _combine_uop(True)
  assert variant.arg.name == "flash_fused_gmax_combine_f16_32_128"
  assert variant.key != legacy.key


# ── render arms (HIP + CUDA, no GPU needed) ──────────────────────────────────

def _render_combine(output_fp16, ren):
  from tinygrad.codegen import to_program
  uop = _combine_uop(output_fp16)
  return next(u.arg for u in to_program(uop, ren).src if u.op is Ops.SOURCE)


def test_fp16_combine_renders_through_hip_and_cuda():
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")),
              CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
    src = _render_combine(True, ren)
    assert "(half)" in src
    # the write buffer is fp16: the store carries the RNE half cast
    assert "flash_fused_gmax_combine_f16_32_128" in src


def test_legacy_combine_render_has_no_half_cast():
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")),
              CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
    src = _render_combine(False, ren)
    assert "(half)" not in src
    assert "flash_fused_gmax_combine_32_128" in src


# ── admission wiring ─────────────────────────────────────────────────────────

def test_combine_fusion_admission_closed_default():
  cap = FlashDecodeCapability(supports_warp_shfl_xor=True, supports_fdot2=True)
  adm = FlashDecodeAdmission(True, cap, True)
  assert adm.admitted
  assert not adm.combine_fusion_admitted
  # M2's epilogue-fusion promotion must NOT open the M5 combine variant
  adm_m2 = FlashDecodeAdmission(True, cap, True, epilogue_fusion_promoted=True)
  assert adm_m2.fusion_admitted
  assert not adm_m2.combine_fusion_admitted
  # Only the M5 combine record opens it
  adm_open = FlashDecodeAdmission(True, cap, True, combine_fusion_promoted=True)
  assert adm_open.combine_fusion_admitted
  assert not adm_open.fusion_admitted
  # Neither capability nor promotion satisfied -> nothing works
  adm_none = FlashDecodeAdmission(False, FlashDecodeCapability(), False)
  assert not adm_none.admitted
  assert not adm_none.combine_fusion_admitted


def test_route_evaluate_threads_combine_fusion():
  cfg = FlashDecodeRouteConfig("c", "r", 32, 48, None, 1)
  cap = FlashDecodeCapability(supports_warp_shfl_xor=True, supports_fdot2=True)
  adm = cfg.evaluate(1, 32, 8, 128, cap, True)
  assert adm.admitted and not adm.combine_fusion_admitted
  adm_open = cfg.evaluate(1, 32, 8, 128, cap, True, combine_fusion_promoted=True)
  assert adm_open.admitted and adm_open.combine_fusion_admitted
  adm_both = cfg.evaluate(1, 32, 8, 128, cap, True, epilogue_fusion_promoted=True, combine_fusion_promoted=True)
  assert adm_both.combine_fusion_admitted and adm_both.fusion_admitted
