"""CPU contract for the production-owned G4/G5 flash-decode runtime."""
import hashlib
from pathlib import Path

import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.llm.flash_decode_attention import (FLASH_DECODE_G4, FLASH_DECODE_G5, FlashDecodeTileSpec,
  describe_flash_decode_attention)


def _tile_inputs(hq:int, split_count:int, max_context:int=8192):
  return (UOp.placeholder((hq * split_count * 130,), dtypes.float32, 0),
          UOp.placeholder((hq * 128,), dtypes.float16, 1),
          UOp.placeholder((2, 1, 8, max_context, 128), dtypes.float16, 2))


@pytest.mark.parametrize("route,expected", [
  (FLASH_DECODE_G4, ("attention_decode.flash_live_split", "decode_flash_live_split_g4_kvboth", 32, 48, None, 1)),
  (FLASH_DECODE_G5, ("attention_decode.flash_live_split_g5", "decode_flash_live_split_g5_kvboth", 40, 32, 2, 4)),
])
def test_selected_route_configs_are_frozen(route, expected):
  assert (route.candidate_id, route.route_id, route.query_heads, route.split_size,
          route.query_group_size, route.stage_width) == expected
  assert route.staging == "KV_BOTH" and route.kv_heads == 8 and route.head_dim == 128
  assert route.supports(1, route.query_heads, 8, 128, "AMD:0")
  assert not route.supports(2, route.query_heads, 8, 128, "AMD:0")
  assert not route.supports(1, route.query_heads, 4, 128, "AMD:0")
  assert not route.supports(1, route.query_heads, 8, 64, "AMD:0")
  assert not route.supports(1, route.query_heads, 8, 128, "CPU")


@pytest.mark.parametrize("hq,split_count,query_group_size,stage_width,kernel_names,tile_sha,combine_sha", [
  (32, 48, None, 1,
   ("flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", "flash_fused_gmax_combine_32_128"),
   "365bbf7102d830a58acf276ea529229bbff750b4548b89b9a8e9e836f88ee314",
   "6362f488a66d7cd7f972355416311ef79f6d7f3a1b7bf91def84ab984025efb4"),
  (40, 32, 2, 4,
   ("flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128_qg2", "flash_fused_gmax_combine_40_128"),
   "6e31a7f99c575d6752e9f473c5400e83ce5910e603f699f0ad6d33d59fe71d2f",
   "107abfbe59269e8a91340ef02cf87a37e30745b5e457454003ef7290d01b1023"),
])
def test_production_emitters_match_frozen_promoted_graphs(hq, split_count, query_group_size, stage_width,
                                                           kernel_names, tile_sha, combine_sha):
  tc = UOp.variable("Tc", 0, 8192)
  args = _tile_inputs(hq, split_count)
  production = describe_flash_decode_attention(hq, 128, 8, 8192, split_count,
    query_group_size=query_group_size, stage_width=stage_width)
  production_tile = production.emit_tile(tc)(*args)
  out = UOp.placeholder((hq * 128,), dtypes.float32, 3)
  production_combine = production.emit_combine()(out, args[0])
  assert production.emitted_kernel_names == kernel_names
  assert hashlib.sha256(repr(production_tile.key).encode()).hexdigest() == tile_sha
  assert hashlib.sha256(repr(production_combine.key).encode()).hexdigest() == combine_sha


def test_quant_and_rope_binding_order_and_fail_loud_contract():
  spec = describe_flash_decode_attention(32, 128, 8, 8192, 48, quant=True, rope=True)
  assert [role.name for role in spec.tile.buffer_roles] == ["pout", "q", "cache", "kvscale", "freqs"]
  tc = UOp.variable("Tc", 0, 8192)
  with pytest.raises(ValueError, match="scale buffer"):
    spec.emit_tile(tc)(*_tile_inputs(32, 48))
  with pytest.raises(ValueError, match="freqs"):
    spec.emit_tile(tc)(*_tile_inputs(32, 48), UOp.placeholder((2, 1, 8, 8192), dtypes.float16, 4))


@pytest.mark.parametrize("kwargs,match", [
  ({"Hq": 0}, "positive"),
  ({"Hq": 30}, "divisible"),
  ({"staging": "K_ONLY"}, "KV_BOTH"),
  ({"stage_width": 3}, "stage_width"),
])
def test_invalid_specs_fail_closed(kwargs, match):
  values = {"Hq": 32, "Hd": 128, "Hkv": 8, "MAXC": 8192, "split_count": 48, "staging": "KV_BOTH"}
  values.update(kwargs)
  spec = FlashDecodeTileSpec(**values)
  with pytest.raises(ValueError, match=match): spec.validate()


def test_production_module_has_no_research_import():
  source = (Path(__file__).parents[2] / "tinygrad/llm/flash_decode_attention.py").read_text()
  assert "extra.llm_research" not in source
