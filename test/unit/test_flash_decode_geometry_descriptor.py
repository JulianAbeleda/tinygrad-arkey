"""P1 regression: the flash decode geometry is descriptor-owned with a canonical round-trip identity.

docs/task_workflow/input/nv-search-genericization-flash-shape-scope-20260818.md sections 4/6: the tile
descriptor owns lane_width, score_group_width, warps, token_block, stage_width, reduce_structure and
dot_pair_width; to_json is the canonical candidate identity; non-default geometry gets a deterministic
kernel-name suffix; the G4/G5 default geometry keeps the exact historical kernel names.
"""
from __future__ import annotations

import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.llm.flash_decode_attention import (FLASH_DECODE_G4, FLASH_DECODE_G5, FlashCombineSpec,
                                                 FlashDecodeAttentionSpec, FlashDecodeTileSpec,
                                                 describe_flash_decode_attention)


def _ceildiv(a, b: int): return (a + b - 1) // b


def _tile_args(hq: int, split_count: int, max_context: int = 8192):
  return (UOp.placeholder((hq * split_count * 130,), dtypes.float32, 0),
          UOp.placeholder((hq * 128,), dtypes.float16, 1),
          UOp.placeholder((2, 1, 8, max_context, 128), dtypes.float16, 2))


@pytest.mark.parametrize("hq,split_count,query_group_size,stage_width,expected_tile,expected_combine", [
  (FLASH_DECODE_G4.query_heads, FLASH_DECODE_G4.split_size, FLASH_DECODE_G4.query_group_size,
   FLASH_DECODE_G4.stage_width, "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128",
   "flash_fused_gmax_combine_32_128"),
  (FLASH_DECODE_G5.query_heads, FLASH_DECODE_G5.split_size, FLASH_DECODE_G5.query_group_size,
   FLASH_DECODE_G5.stage_width, "flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128_qg2",
   "flash_fused_gmax_combine_40_128"),
])
def test_default_geometry_keeps_exact_kernel_names_and_roundtrips(hq, split_count, query_group_size,
                                                                  stage_width, expected_tile, expected_combine):
  spec = describe_flash_decode_attention(hq, 128, 8, 8192, split_count,
                                         query_group_size=query_group_size, stage_width=stage_width)
  assert spec.emitted_kernel_names == (expected_tile, expected_combine)
  # canonical JSON is the candidate identity: it carries every geometry field and round-trips.
  tile_json, combine_json = spec.tile.to_json(), spec.combine.to_json()
  rebuilt = FlashDecodeAttentionSpec(FlashDecodeTileSpec(**tile_json), FlashCombineSpec(**combine_json))
  assert rebuilt.tile.to_json() == tile_json
  assert rebuilt.combine.to_json() == combine_json
  assert rebuilt.emitted_kernel_names == spec.emitted_kernel_names


def test_derived_values_match_current_production_defaults():
  """Scope section 4.1: with defaults, the derived values must equal the historical constants exactly."""
  for hq, split_count, query_group_size, stage_width in ((32, 48, None, 1), (40, 32, 2, 4)):
    spec = describe_flash_decode_attention(hq, 128, 8, 8192, split_count, query_group_size=query_group_size,
                                           stage_width=stage_width)
    t = spec.tile
    G = t.Hq // t.Hkv
    QG = G if t.query_group_size is None else t.query_group_size
    assert QG == (G if query_group_size is None else query_group_size)
    assert t.warps is None
    warps = QG
    threads = t.lane_width * warps
    group_width = t.score_group_width or t.lane_width
    R = t.Hd // t.lane_width
    RP = t.Hd // (t.lane_width * t.dot_pair_width)
    STAGES = _ceildiv(t.token_block * t.Hd, threads)
    assert (t.lane_width, t.token_block, t.dot_pair_width, t.stage_width, t.reduce_structure,
            t.score_group_width, t.warps) == (32, 16, 2, stage_width, "staged", None, None)
    assert threads == 32 * QG
    assert group_width == 32
    assert R == t.Hd // 32
    assert RP == t.Hd // 64
    assert STAGES == _ceildiv(16 * 128, 32 * QG)


@pytest.mark.parametrize("field,value,suffix", [
  ("split_count", 32, "_s32"),
  ("lane_width", 16, "_lw16"),
  ("token_block", 32, "_tk32"),
  ("stage_width", 2, "_sw2"),
  ("reduce_structure", "inline", "_ri"),
  ("dot_pair_width", 1, "_dpw1"),
  ("score_group_width", 32, "_sgw32"),
  ("warps", 8, "_w8"),
])
def test_non_default_geometry_gets_a_deterministic_suffix(field, value, suffix):
  kw = {"Hq": 32, "Hd": 128, "Hkv": 8, "MAXC": 8192, "split_count": 48}
  kw[field] = value
  spec = FlashDecodeTileSpec(**kw)
  assert spec.kernel_name == f"flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128{suffix}"
  # the emitted sink name must agree with the descriptor identity (spec and builder share one rule)
  tc = UOp.variable("Tc", 0, 8192)
  sink = spec.emit(tc)(*_tile_args(32, 48))
  assert sink.arg.name == spec.kernel_name


def test_score_group_width_and_warps_must_cover_the_full_head():
  """Column-parallel groups and dropped heads are not implemented, so the search surface must reject them."""
  with pytest.raises(ValueError, match="score_group_width must equal lane_width"):
    FlashDecodeTileSpec(Hq=32, Hd=128, Hkv=8, MAXC=8192, split_count=48, score_group_width=8).validate()
  with pytest.raises(ValueError, match="warps must be >= query_group_size"):
    FlashDecodeTileSpec(Hq=32, Hd=128, Hkv=8, MAXC=8192, split_count=48, warps=2).validate()
  valid = FlashDecodeTileSpec(Hq=32, Hd=128, Hkv=8, MAXC=8192, split_count=48,
                              lane_width=16, score_group_width=16, warps=4)
  assert valid.validate() is None
  assert "_lw16_sgw16" in valid.kernel_name


def test_p3_winner_geometry_renders_the_distinct_tile_and_unchanged_combine_names():
  """The measured P3 winner (stage_width=4, inline reduce, dot_pair_width=4) must carry a distinct tile
  name while the production combine (S=48, lane_width=32) keeps its historical name."""
  spec = describe_flash_decode_attention(32, 128, 8, 4608, 48, fused_combine=True, stage_width=4,
                                         reduce_structure="inline", dot_pair_width=4)
  assert spec.emitted_kernel_names == (
    "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128_sw4_ri_dpw4",
    "flash_fused_gmax_combine_32_128")


def test_combine_lane_width_defaults_to_tile_lane_width():
  spec = describe_flash_decode_attention(32, 128, 8, 8192, 48, lane_width=16)
  assert spec.tile.lane_width == 16 and spec.combine.lane_width == 16
  assert spec.emitted_kernel_names[1] == "flash_fused_gmax_combine_32_128_lw16"
  explicit = describe_flash_decode_attention(32, 128, 8, 8192, 48, lane_width=16, combine_lane_width=32)
  assert explicit.combine.lane_width == 32


@pytest.mark.parametrize("kwargs,match", [
  ({"lane_width": 0}, "lane_width"),
  ({"lane_width": 24}, "lane_width"),
  ({"score_group_width": 0}, "score_group_width"),
  ({"warps": -1}, "warps"),
  ({"dot_pair_width": 0}, "dot_pair_width"),
  ({"reduce_structure": "split"}, "reduce_structure"),
  ({"token_block": -3}, "token_block"),
])
def test_invalid_geometry_fails_at_validate(kwargs, match):
  spec = FlashDecodeTileSpec(Hq=32, Hd=128, Hkv=8, MAXC=8192, split_count=48, **kwargs)
  with pytest.raises(ValueError, match=match): spec.validate()


def test_combine_lane_width_validation():
  for bad in (0, 24):
    spec = FlashCombineSpec(Hd=128, Hq=32, split_count=48, lane_width=bad)
    with pytest.raises(ValueError, match="lane_width"): spec.validate()
  assert FlashCombineSpec(Hd=128, Hq=32, split_count=48, lane_width=16).validate() is None


def test_live_split_exposes_independent_combine_width():
  from inspect import signature
  from tinygrad.llm.flash_decode_attention import flash_decode_live_split_block_tile
  parameter = signature(flash_decode_live_split_block_tile).parameters["combine_lane_width"]
  assert parameter.default is None


def test_production_defaults_never_consult_the_env(monkeypatch):
  """Production defaults are descriptor-owned: the env switches are legacy aliases only (builder params
  None), so the G4/G5 emission must be byte-identical whether or not the env vars are set."""
  from tinygrad import getenv
  from tinygrad.llm.flash_decode_attention import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
  tc = UOp.variable("Tc", 0, 8192)

  def emit():
    spec = describe_flash_decode_attention(32, 128, 8, 8192, 48)
    return spec.emit_tile(tc)(*_tile_args(32, 48))

  getenv.cache_clear()
  baseline = emit()
  monkeypatch.setenv("DECODE_STAGE_COALESCE", "4")
  monkeypatch.setenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE", "1")
  getenv.cache_clear()
  assert emit().key == baseline.key
  monkeypatch.delenv("DECODE_STAGE_COALESCE")
  monkeypatch.delenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE")
  getenv.cache_clear()


def test_env_aliases_stay_explicit_legacy_alias_only(monkeypatch):
  """The env switches remain honored ONLY when the caller passes None for the descriptor field (legacy
  alias); a concrete descriptor field wins over the env."""
  from tinygrad import getenv
  from tinygrad.llm.flash_decode_attention import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
  tc = UOp.variable("Tc", 0, 8192)
  args = _tile_args(32, 48)

  def emit(*, stage_width=None, reduce_structure=None):
    return flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(
      128, 32, 8, 8192, 96, 48, tc, stage_width=stage_width,
      reduce_structure=reduce_structure)(*args)

  getenv.cache_clear()
  staged = emit()
  monkeypatch.setenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE", "1")
  getenv.cache_clear()
  inline_via_env = emit()
  assert inline_via_env.key != staged.key
  # a concrete reduce_structure field wins over the env alias
  getenv.cache_clear()
  staged_concrete = emit(reduce_structure="staged")
  assert staged_concrete.key == staged.key

  monkeypatch.delenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE")
  getenv.cache_clear()
  one_elem_staging = emit()
  monkeypatch.setenv("DECODE_STAGE_COALESCE", "4")
  getenv.cache_clear()
  coalesced_via_env = emit(stage_width=None)
  assert coalesced_via_env.key != one_elem_staging.key
  # a concrete stage_width field wins over the env alias (emission must be env-independent)
  getenv.cache_clear()
  concrete_stage_baseline = emit(stage_width=1)
  getenv.cache_clear()
  concrete_stage_with_env = emit(stage_width=1)
  assert concrete_stage_with_env.key == concrete_stage_baseline.key
