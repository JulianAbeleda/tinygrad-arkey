import json

from extra.qk.prefill.lds2_s9_combined_search import candidate_space


def test_combined_search_candidates_are_bounded_with_default_memory(tmp_path):
  candidates, blockers = candidate_space(memory_artifact=tmp_path / "missing-memory-search.json")

  assert len(candidates) == 8
  assert blockers and "memory axis default-only" in blockers[0]
  assert candidates[0].name == "wait_default__reg_default__lifecycle_default__memory_default"
  assert candidates[-1].name == (
    "wait_lgkm_after_coop_store_2__reg_block_shift_plus_1__"
    "lifecycle_prologue_init_counter_before_adv_k__memory_default"
  )
  assert {c.wait_policy.lgkm_after_coop_store for c in candidates} == {0, 2}
  assert {c.reg_layout.FA for c in candidates} == {10, 11}
  assert {c.memory_source for c in candidates} == {"default"}


def test_combined_search_can_parse_valid_memory_artifact(tmp_path):
  artifact = tmp_path / "memory-search.json"
  artifact.write_text(json.dumps({
    "rows": [
      {"name": "pad24", "status": "ok", "tflops": 99.0,
       "memory_layout": {"SA": 88, "SB": 88, "LDS_A": 11264, "BUFSZ": 22528, "NBUF": 2}},
    ],
  }))

  candidates, blockers = candidate_space(memory_artifact=artifact)

  assert blockers == []
  assert len(candidates) == 16
  assert {c.memory_source for c in candidates} == {"default", "artifact_best:pad24"}
