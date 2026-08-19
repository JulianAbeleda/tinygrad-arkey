"""CPU-only contract for the nv_flash_geometry_population.v1 builder."""
import extra.llm_research.decode.nv_flash_geometry_population as population


def test_build_population_sm120_contract():
  facts = dict(population.SM120_TARGET_FACTS)
  result = population.build_population(facts)

  assert result["schema"] == population.SCHEMA
  assert result["control_tile_name"] == "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"
  assert result["target_facts"] == facts

  candidates = result["candidates"]
  assert candidates
  for row in candidates:
    assert row["lane_width"] in (8, 16, 32)
    assert len(row["candidate_hash"]) == 64
    assert row["score_group_width"] is None
    assert row["warps"] is None

  accepted = [row for row in candidates if row["legality"] is None]
  rejected = [row for row in candidates if row["legality"] is not None]
  assert len(candidates) == len(accepted) + len(rejected)
  for row in accepted:
    assert isinstance(row["kernel_name"], str) and row["kernel_name"]

  expected_order = sorted(candidates, key=lambda row: (-row["priority_score"], row["candidate_hash"]))
  assert [row["deterministic_order"] for row in candidates] == list(range(len(candidates)))
  assert [row["candidate_hash"] for row in candidates] == [row["candidate_hash"] for row in expected_order]

  by_geometry: dict[tuple, dict[int, str]] = {}
  for row in accepted:
    key = (row["lane_width"], row["token_block"], row["stage_width"],
           row["reduce_structure"], row["dot_pair_width"])
    by_geometry.setdefault(key, {})[row["split_count"]] = row["kernel_name"]
  split_names = sorted(next(iter(by_geometry.values())).items())
  assert len(split_names) >= 2
  assert split_names[0][1] != split_names[1][1]
