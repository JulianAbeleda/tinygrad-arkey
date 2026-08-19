#!/usr/bin/env python3
from tinygrad import dtypes
from tinygrad.uop.ops import UOp

from extra.llm_research.decode.flash_decode_attention_spec import (FlashCombineSpec, FlashDecodeAttentionSpec, LiveSplitGeometrySpec,
                                                 FlashDecodeTileSpec, describe_flash_decode_attention,
                                                 emit_flash_decode_combine, emit_flash_decode_tile)


def test_live_split_geometry_spec_arithmetics():
  geo = LiveSplitGeometrySpec(split_count=48, token_block=16)
  assert geo.per_split_length(1024) == 22
  assert geo.aligned_per_split_length(1024) == 32
  assert geo.blocks(1024) == 2
  geo.validate()


def test_flash_decode_attention_descriptor_defaults():
  spec = describe_flash_decode_attention(Hq=40, Hd=128, Hkv=8, MAXC=8192, S=48, fused_combine=True, quant=False, rope=False)
  assert isinstance(spec, FlashDecodeAttentionSpec)
  assert spec.tile.Hq == 40
  assert spec.tile.quant is False
  assert spec.tile.rope is False
  assert isinstance(spec.combine, FlashCombineSpec)
  assert spec.emitted_kernel_names == (
    "flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128_s48",
    "flash_fused_gmax_combine_40_128_s48")


def test_tile_emit_kernel_name_matches_flash_kernels():
  spec = FlashDecodeAttentionSpec(
    tile=FlashDecodeTileSpec(Hq=32, Hd=128, Hkv=8, MAXC=8192, split_count=48, staging="KV_BOTH", quant=False),
    combine=FlashCombineSpec(Hd=128, Hq=32, split_count=48),
  )
  tc = UOp.variable("Tc", 0, 8192)
  pout = UOp.placeholder((32 * 48 * 130,), dtypes.float32, 0)
  q = UOp.placeholder((32 * 128,), dtypes.float16, 1)
  cache = UOp.placeholder((2, 1, 8, 8192, 128), dtypes.float16, 2)
  kernel = emit_flash_decode_tile(spec, tc)
  uops = kernel(pout, q, cache)
  assert uops.arg.name == "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"


def test_combine_emit_kernel_name_matches_flash_kernels():
  spec = describe_flash_decode_attention(Hq=40, Hd=128, Hkv=8, MAXC=8192, S=48, fused_combine=True)
  out = UOp.placeholder((40 * 128,), dtypes.float32, 0)
  pout = UOp.placeholder((40 * 48 * 130,), dtypes.float32, 1)
  kernel = emit_flash_decode_combine(spec)
  uops = kernel(out, pout)
  assert uops.arg.name == "flash_fused_gmax_combine_40_128_s48"


def test_geometry_fields_are_descriptor_owned_through_the_alias():
  """P1 (nv-search-genericization-flash-shape-scope-20260818.md section 4): the descriptor owns the
  geometry fields; a non-default geometry gets a deterministic kernel-name suffix."""
  spec = FlashDecodeTileSpec(Hq=32, Hd=128, Hkv=8, MAXC=8192, split_count=48, lane_width=16,
                             token_block=32, stage_width=2, reduce_structure="inline",
                             score_group_width=16, warps=8, dot_pair_width=1)
  spec.validate()
  assert spec.kernel_name == "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128_lw16_tk32_sw2_ri_dpw1_sgw16_w8"
  assert spec.to_json()["lane_width"] == 16
  assert spec.to_json()["reduce_structure"] == "inline"
  assert spec.to_json()["target"] is None
  combine = FlashCombineSpec(Hd=128, Hq=32, split_count=48, lane_width=16)
  combine.validate()
  assert combine.kernel_name == "flash_fused_gmax_combine_32_128_lw16"
