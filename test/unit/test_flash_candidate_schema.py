"""CPU-only contracts for the standalone flash_decode_candidate.v1 schema."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from extra.llm_research import flash_candidate_schema as s


def _descriptor(**tile_kwargs):
  defaults = {"Hq": 8, "Hd": 128, "Hkv": 8, "MAXC": 4096, "split_count": 1}
  defaults.update(tile_kwargs)
  return s.to_spec_dict(tile=s.tile_fields(**defaults))


def test_canonical_json_is_stable_across_key_order_and_whitespace():
  descriptor = _descriptor()
  shuffled = {"combine": descriptor["combine"], "tile": dict(reversed(tuple(descriptor["tile"].items()))),
              "schema_version": descriptor["schema_version"]}
  assert s.canonical_json(descriptor) == s.canonical_json(shuffled)
  assert json.loads(s.canonical_json(descriptor)) == descriptor


def test_defaults_reproduce_production_geometry():
  descriptor = _descriptor()
  tile = descriptor["tile"]
  assert tile["token_block"] == 16
  assert tile["lane_width"] == 32
  assert tile["score_group_width"] is None
  assert tile["warps"] is None
  assert tile["query_group_size"] is None
  assert tile["stage_width"] == 1
  assert tile["reduce_structure"] == "staged"
  assert tile["dot_pair_width"] == 2
  assert tile["staging"] == "KV_BOTH"
  assert tile["quant"] is False and tile["rope"] is False
  assert descriptor["combine"] is None


def test_hash_is_deterministic_and_covers_every_tile_field():
  baseline = _descriptor()
  assert s.candidate_hash(baseline) == s.candidate_hash(dict(baseline))
  assert len(s.candidate_hash(baseline)) == 64
  changes = {
    "Hq": 16, "Hd": 192, "Hkv": 4, "MAXC": 8192, "split_count": 2,
    "staging": "K_ONLY", "quant": True, "rope": True, "token_block": 32,
    "lane_width": 16, "score_group_width": 32, "warps": 2, "query_group_size": 1,
    "stage_width": 2, "reduce_structure": "inline", "dot_pair_width": 4,
  }
  for field, value in changes.items():
    changed = _descriptor(**{field: value})
    assert s.candidate_hash(changed) != s.candidate_hash(baseline), f"field {field} did not change the hash"


def test_hash_changes_for_every_combine_field_and_presence():
  baseline = _descriptor()
  combine = s.combine_fields(Hd=128, Hq=8, split_count=1)
  with_combine = s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1), combine=combine)
  assert s.candidate_hash(with_combine) != s.candidate_hash(baseline)
  for field, value in {"Hd": 192, "Hq": 4, "split_count": 2, "stride": 4,
                       "output_fp16": True, "lane_width": 16}.items():
    combine_kwargs = {"Hd": 128, "Hq": 8, "split_count": 1}
    combine_kwargs[field] = value
    changed = s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1),
                             combine=s.combine_fields(**combine_kwargs))
    assert s.candidate_hash(changed) != s.candidate_hash(with_combine), f"combine field {field} did not change the hash"


def test_validate_accepts_production_geometry_and_returns_canonical_descriptor():
  descriptor = _descriptor()
  assert s.validate(descriptor) == descriptor
  assert s.from_dict(descriptor) == descriptor


def test_score_group_and_warps_must_cover_the_full_head():
  valid = _descriptor(Hq=32, lane_width=32, score_group_width=32, warps=4)
  assert s.validate(valid) == valid
  with pytest.raises(s.SchemaError, match="score_group_width must equal lane_width"):
    s.validate(_descriptor(Hq=32, score_group_width=8))
  with pytest.raises(s.SchemaError, match="warps must be >= query_group_size"):
    s.validate(_descriptor(Hq=32, warps=2))


@pytest.mark.parametrize("mutate, fragment", [
  (lambda d: d["tile"].__setitem__("lane_width", 24), "power of two"),
  (lambda d: d["tile"].__setitem__("score_group_width", 48), "must equal lane_width"),
  (lambda d: d["tile"].__setitem__("score_group_width", 0), "positive int"),
  (lambda d: d["tile"].__setitem__("dot_pair_width", 0), "positive int"),
  (lambda d: d["tile"].__setitem__("Hd", 100), "divisible by lane_width"),
  (lambda d: d["tile"].__setitem__("token_block", 0), "positive int"),
  (lambda d: d["tile"].__setitem__("reduce_structure", "recursive"), "reduce_structure"),
  (lambda d: d["tile"].__setitem__("staging", "BOTH"), "staging"),
  (lambda d: d["tile"].__setitem__("Hq", 9), "divisible by tile.Hkv"),
  (lambda d: d["tile"].__setitem__("split_count", 0), "positive int"),
  (lambda d: d["tile"].__setitem__("warps", 0), "positive int"),
  (lambda d: d["tile"].__setitem__("quant", "yes"), "bool"),
  (lambda d: d["tile"].pop("Hd"), "positive int"),
  (lambda d: d["tile"].__setitem__("MAXC", float("nan")), "JSON-safe"),
  (lambda d: d.__setitem__("combine", {"Hd": 100, "Hq": 8, "split_count": 1}), "combine.Hd"),
  (lambda d: d.__setitem__("combine", {"Hd": 128, "Hq": 8, "split_count": 1, "stride": 0}), "combine.stride"),
  (lambda d: d.__setitem__("combine", {"Hd": 128, "Hq": 8, "split_count": 1, "lane_width": 3}), "power of two"),
  (lambda d: d.__setitem__("schema_version", "other.v1"), "schema_version"),
  (lambda d: d.__setitem__("tile", "not-an-object"), "tile object"),
])
def test_validate_rejects_bad_geometry(mutate, fragment):
  descriptor = _descriptor()
  mutate(descriptor)
  with pytest.raises(s.SchemaError, match=fragment):
    s.validate(descriptor)


def test_from_dict_materializes_defaults_and_resolves_combine_lane_width():
  combine = s.combine_fields(Hd=128, Hq=8, split_count=1)
  descriptor = s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1), combine=combine)
  assert descriptor["combine"]["lane_width"] == 32  # defaults to tile.lane_width
  explicit = s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1, lane_width=16),
                            combine=s.combine_fields(Hd=128, Hq=8, split_count=1, lane_width=16))
  assert s.candidate_hash(explicit) == s.candidate_hash(
    s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1, lane_width=16),
                   combine=s.combine_fields(Hd=128, Hq=8, split_count=1)))
  assert s.from_dict(s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1))) == \
    s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1))


def test_candidate_hash_ignores_the_outer_envelope_hash_key():
  descriptor = _descriptor()
  envelope = s.candidate_envelope(descriptor)
  assert envelope["candidate_hash"] == s.candidate_hash(descriptor)
  assert s.candidate_hash(envelope) == s.candidate_hash(descriptor)
  forged = dict(envelope, candidate_hash="f" * 64)
  assert s.candidate_hash(forged) == s.candidate_hash(descriptor)


def test_derived_geometry_arithmetic():
  tile = _descriptor()["tile"]
  assert s.derived_group_width(tile) == 32
  assert s.derived_warps(tile) == 1  # G = Hq // Hkv
  assert s.derived_threads(tile) == 32
  assert s.kv_tile_bytes(tile) == 2 * 16 * 128 * 2
  assert s.kv_tile_bytes(dict(tile, staging="K_ONLY")) == 16 * 128 * 2
  assert s.ladder_spans_lanes(tile) is True
  assert s.reduce_stages(tile) == 5
  assert s.reduce_stages(dict(tile, reduce_structure="inline")) == 1
  assert s.reduce_stages(dict(tile, lane_width=8)) == 3
  descriptor = s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1),
                              combine=s.combine_fields(Hd=128, Hq=8, split_count=1))
  assert s.combine_scratch_bytes(descriptor["combine"]) == 4
  assert s.local_memory_bytes(descriptor) == 2 * 16 * 128 * 2 + 4
  assert s.ceil_log2_ratio(32, 1) == 5
  assert s.ceil_log2_ratio(32, 4) == 3
  assert s.ceil_log2_ratio(8, 16) == 0


def test_schema_module_is_standalone_stdlib_only():
  root = Path(__file__).resolve().parents[2]
  code = (
    "import sys\n"
    "import extra.llm_research.flash_candidate_schema as s\n"
    "assert 'tinygrad' not in sys.modules, 'schema module must not import tinygrad'\n"
    "assert 'extra.llm_research.search_provider' not in sys.modules\n"
    "d = s.to_spec_dict(tile=s.tile_fields(Hq=8, Hd=128, Hkv=8, MAXC=4096, split_count=1))\n"
    "print(s.candidate_hash(d))\n"
  )
  env = {**os.environ, "PYTHONPATH": str(root)}
  out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
  assert out.stdout.strip() == s.candidate_hash(_descriptor())
