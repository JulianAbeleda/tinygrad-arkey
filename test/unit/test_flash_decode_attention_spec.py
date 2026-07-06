#!/usr/bin/env python3
from tinygrad import dtypes
from tinygrad.uop.ops import UOp

from extra.qk.flash_decode_attention_spec import (FlashCombineSpec, FlashDecodeAttentionSpec, LiveSplitGeometrySpec,
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
    "flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128",
    "flash_fused_gmax_combine_40_128")


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
  assert uops.arg.name == "flash_fused_gmax_combine_40_128"
