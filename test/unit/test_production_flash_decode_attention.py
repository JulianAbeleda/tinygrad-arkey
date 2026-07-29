"""CPU contract for the production-owned G4/G5 flash-decode runtime."""
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


@pytest.mark.parametrize("hq,split_count,query_group_size,stage_width", [(32, 48, None, 1), (40, 32, 2, 4)])
def test_production_emitters_are_structurally_identical_to_promoted_emitters(hq, split_count, query_group_size, stage_width):
  # This is a migration parity gate, not a runtime dependency: the production module itself has no extra import.
  from extra.llm_research.decode.flash_decode_attention_spec import describe_flash_decode_attention as legacy_describe
  tc = UOp.variable("Tc", 0, 8192)
  args = _tile_inputs(hq, split_count)
  production = describe_flash_decode_attention(hq, 128, 8, 8192, split_count,
    query_group_size=query_group_size, stage_width=stage_width)
  legacy = legacy_describe(hq, 128, 8, 8192, split_count,
    query_group_size=query_group_size, stage_width=stage_width)
  production_tile, legacy_tile = production.emit_tile(tc)(*args), legacy.emit_tile(tc)(*args)
  out = UOp.placeholder((hq * 128,), dtypes.float32, 3)
  production_combine, legacy_combine = production.emit_combine()(out, args[0]), legacy.emit_combine()(out, args[0])
  assert production.emitted_kernel_names == legacy.emitted_kernel_names
  assert production_tile.key == legacy_tile.key
  assert production_combine.key == legacy_combine.key


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
