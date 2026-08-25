"""CPU contract for the production-owned G4/G5 flash-decode runtime."""
from pathlib import Path

import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.llm.flash_decode_attention import (FLASH_DECODE_G4, FLASH_DECODE_G5, FlashDecodeCapability,
  FlashDecodeTileSpec, describe_flash_decode_attention, flash_decode_coarse_split_override)


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
  # TG7 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): shape_ok() is the
  # pre-TG7 supports() shape check, unchanged, and is device-agnostic on purpose (scope section 3.1 -- shape
  # gates stay exactly where they were; capability/policy are separate questions, tested below).
  assert route.shape_ok(1, route.query_heads, 8, 128)
  assert not route.shape_ok(2, route.query_heads, 8, 128)
  assert not route.shape_ok(1, route.query_heads, 4, 128)
  assert not route.shape_ok(1, route.query_heads, 8, 64)


def test_evaluate_splits_shape_capability_and_policy_with_distinct_reasons():
  """TG7: the pre-TG7 `device == "AMD"` gate collapsed shape, codegen capability and promotion into one
  boolean. evaluate() answers all three independently and every rejection carries its own reason -- never a
  silent fallback (scope section 4.4)."""
  route = FLASH_DECODE_G4
  satisfied = FlashDecodeCapability(True, True)
  unsatisfied_no_shuffle = FlashDecodeCapability(False, True)
  unreported = FlashDecodeCapability(None, None)  # e.g. an unopened/unknown renderer

  admitted = route.evaluate(1, route.query_heads, 8, 128, satisfied, True)
  assert admitted.admitted and admitted.reason is None

  bad_shape = route.evaluate(1, 999, 8, 128, satisfied, True)
  assert not bad_shape.admitted and bad_shape.reason == "shape_not_supported"

  no_capability = route.evaluate(1, route.query_heads, 8, 128, unsatisfied_no_shuffle, True)
  assert not no_capability.admitted and no_capability.reason == "capability_missing"

  unreported_capability = route.evaluate(1, route.query_heads, 8, 128, unreported, True)
  assert not unreported_capability.admitted and unreported_capability.reason == "capability_missing"

  not_promoted = route.evaluate(1, route.query_heads, 8, 128, satisfied, False)
  assert not not_promoted.admitted and not_promoted.reason == "policy_target_not_promoted"

  # Shape is checked first: a shape mismatch is reported as such even when capability also fails.
  bad_shape_and_capability = route.evaluate(1, 999, 8, 128, unsatisfied_no_shuffle, True)
  assert bad_shape_and_capability.reason == "shape_not_supported"


def test_capability_requires_both_providers_and_never_treats_unreported_as_satisfied():
  assert FlashDecodeCapability(True, True).satisfied
  assert not FlashDecodeCapability(True, False).satisfied
  assert not FlashDecodeCapability(False, True).satisfied
  assert not FlashDecodeCapability(None, True).satisfied   # unreported -- never truthy-coerced
  assert not FlashDecodeCapability(True, None).satisfied
  assert not FlashDecodeCapability().satisfied


def test_real_metal_renderer_now_reports_flash_decode_capability():
  """This machine's real, already-open default device is Metal (scope section 8 note: this is genuine
  hardware, not a structural proxy). Before TG7, Metal could never admit flash decode no matter what -- the
  gate was `device == "AMD"`. After TG1's warp_shfl_xor and this package's fdot2 providers, Metal's real
  renderer reports both, so it now clears the capability question -- proving capability is read from the
  renderer, not inferred from the backend string."""
  from tinygrad import Device
  from tinygrad.llm.flash_decode_attention import flash_decode_capability_from_renderer
  if Device.DEFAULT != "METAL": pytest.skip("this proof requires the real Metal backend")
  capability = flash_decode_capability_from_renderer(Device["METAL"].renderer)
  assert capability.supports_warp_shfl_xor is True
  assert capability.supports_fdot2 is True
  assert capability.satisfied


@pytest.mark.parametrize("hq,split_count,query_group_size,stage_width", [(32, 48, None, 1), (40, 32, 2, 4)])
def test_production_emitters_are_structurally_identical_to_promoted_emitters(hq, split_count, query_group_size, stage_width):
  """P1 migration parity gate (nv-search-genericization-flash-shape-scope-20260818.md section 6):
  the descriptor-driven production emitter must render byte-identical AMD gfx1100 source to the pre-change
  baseline. The pre-change baseline is `extra/llm_research/flash_kernels.py` / `live_split_geometry.py`,
  which were deliberately NOT edited by the reconciliation -- the extra spec module now re-exports the
  canonical one, so the untouched legacy builders are the only surviving byte-for-byte baseline. TG7
  intentionally changed the tile kernel's pre-lowering UOp shape (target-agnostic fdot2/exp2f CUSTOMI args
  resolved per renderer instead of literal AMD strings baked in at construction time), so raw `.key`
  equality on the unlowered tile AST no longer holds; what must hold is byte-identical rendered AMD source,
  which includes the kernel name (the C function name). The combine kernel touches neither intrinsic, so
  its raw `.key` equality is still asserted directly."""
  from tinygrad.helpers import Target
  from tinygrad.renderer.cstyle import HIPRenderer
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import Ops
  from tinygrad.llm.flash_decode_attention import LiveSplitGeometrySpec
  from extra.llm_research.flash_kernels import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel as legacy_tile_builder
  from extra.llm_research.live_split_geometry import flash_fused_gmax_combine_kernel as legacy_combine_builder
  tc = UOp.variable("Tc", 0, 8192)
  args = _tile_inputs(hq, split_count)
  production = describe_flash_decode_attention(hq, 128, 8, 8192, split_count,
    query_group_size=query_group_size, stage_width=stage_width)
  production_tile = production.emit_tile(tc)(*args)
  split_length = LiveSplitGeometrySpec(split_count, 16).aligned_per_split_length(tc)
  legacy_tile = legacy_tile_builder(128, hq, 8, 8192, split_length, split_count, tc,
                                    query_group_size=query_group_size, stage_width=stage_width)(*args)
  out = UOp.placeholder((hq * 128,), dtypes.float32, 3)
  production_combine = production.emit_combine()(out, args[0])
  legacy_combine = legacy_combine_builder(128, hq, split_count)(out, args[0])
  assert production_combine.key == legacy_combine.key

  def _amd_source(ast):
    ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
    prog = to_program(ast, ren)
    return next(u.arg for u in prog.src if u.op is Ops.SOURCE)

  assert _amd_source(production_tile) == _amd_source(legacy_tile)


def test_research_alias_module_reexports_the_canonical_spec():
  """P1 reconciliation (scope section 6): `extra/llm_research/decode/flash_decode_attention_spec.py` is now
  a thin alias over `tinygrad.llm.flash_decode_attention` -- the canonical owner -- so existing research
  imports keep working and the two routes cannot drift. Identity is checked at the object level."""
  from extra.llm_research.decode import flash_decode_attention_spec as alias
  from tinygrad.llm import flash_decode_attention as canonical
  for name in ("BufferRole", "FlashCombineSpec", "FlashDecodeAttentionSpec", "FlashDecodeTileSpec",
               "LiveSplitGeometrySpec", "describe_flash_decode_attention", "emit_flash_decode_combine",
               "emit_flash_decode_tile"):
    assert getattr(alias, name) is getattr(canonical, name), name
  # The alias path renders the same AMD source as the canonical path (same objects, but prove the intent:
  # research harnesses that import the old module still get the production emitter byte-for-byte).
  from extra.llm_research.decode.flash_decode_attention_spec import describe_flash_decode_attention as alias_describe
  tc = UOp.variable("Tc", 0, 8192)
  args = _tile_inputs(32, 48)
  alias_tile = alias_describe(32, 128, 8, 8192, 48).emit_tile(tc)(*args)
  canonical_tile = describe_flash_decode_attention(32, 128, 8, 8192, 48).emit_tile(tc)(*args)
  assert alias_tile.key == canonical_tile.key


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
  ({"token_block": 0}, "token_block"),
  ({"lane_width": 12}, "lane_width"),
  ({"score_group_width": 64}, "score_group_width"),
  ({"warps": 0}, "warps"),
  ({"dot_pair_width": 3}, "divisible"),
  ({"reduce_structure": "weird"}, "reduce_structure"),
])
def test_invalid_specs_fail_closed(kwargs, match):
  values = {"Hq": 32, "Hd": 128, "Hkv": 8, "MAXC": 8192, "split_count": 48, "staging": "KV_BOTH"}
  values.update(kwargs)
  spec = FlashDecodeTileSpec(**values)
  with pytest.raises(ValueError, match=match): spec.validate()


def test_production_module_has_no_research_import():
  source = (Path(__file__).parents[2] / "tinygrad/llm/flash_decode_attention.py").read_text()
  assert "extra.llm_research" not in source


def test_coarse_split_env_override_contract(monkeypatch):
  """nv-flash-coarse-split A/B gate: FLASH_DECODE_COARSE_SPLIT is read once (getenv is cached per
  process), returns 0 when unset, and the env value when set. Unset env must leave the promoted
  route's kernel names byte-identical; a set env must render distinct, deterministic names for the
  G4 d512 shape so the candidate JIT cache cannot collide with the S=48 control."""
  from tinygrad.helpers import getenv as _getenv
  _getenv.cache_clear()
  monkeypatch.delenv("FLASH_DECODE_COARSE_SPLIT", raising=False)
  assert flash_decode_coarse_split_override() == 0
  promoted = describe_flash_decode_attention(32, 128, 8, 4608, 48)
  assert promoted.tile.kernel_name == "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"
  assert promoted.combine.kernel_name == "flash_fused_gmax_combine_32_128"

  _getenv.cache_clear()
  monkeypatch.setenv("FLASH_DECODE_COARSE_SPLIT", "4")
  assert flash_decode_coarse_split_override() == 4
  s4 = describe_flash_decode_attention(32, 128, 8, 4608, 4)
  assert s4.tile.kernel_name == "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128_s4"
  assert s4.combine.kernel_name == "flash_fused_gmax_combine_32_128_s4"

  _getenv.cache_clear()
  monkeypatch.setenv("FLASH_DECODE_COARSE_SPLIT", "2")
  assert flash_decode_coarse_split_override() == 2
  s2 = describe_flash_decode_attention(32, 128, 8, 4608, 2)
  assert s2.tile.kernel_name == "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128_s2"


def test_adaptive_s64_context_band_is_bounded():
  from tinygrad.llm.model import _adaptive_flash_split_count
  assert _adaptive_flash_split_count(True, 767, 1024) is None
  assert _adaptive_flash_split_count(True, 768, 1024) == 64
  assert _adaptive_flash_split_count(True, 1023, 1024) == 64
  assert _adaptive_flash_split_count(False, 800, 1024) is None
  assert _adaptive_flash_split_count(True, 800, 2048) is None
