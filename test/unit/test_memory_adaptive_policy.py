import copy
import json

from extra.qk.memory_adaptive_policy import (
  build_production_eligibility, build_production_eligibility_requirement,
  cache_matches, canonical_search_key, make_cache_record, select_policy,
)


def candidate_rows():
  return [
    {"candidate_id": "baseline", "strategy": "DIRECT_PACKED_FALLBACK", "route": "direct"},
    {"candidate_id": "tile", "strategy": "BOUNDED_PACKED_TILES", "route": "bounded",
     "production_eligibility_requirement": build_production_eligibility_requirement(
       authority={"schema": "test.cross_route_parity_requirement.v1"})},
  ]


def eligibility(candidate=None, *, eligible=True, blocker=None):
  row = candidate or candidate_rows()[1]
  return build_production_eligibility(
    candidate=row, promotion_eligible=eligible,
    authority=row["production_eligibility_requirement"]["authority"],
    blocker=blocker)


def inputs():
  return {
    "gpu_facts": {"backend": "AMD", "arch": "gfx1100", "wave": 32, "free_bytes": 20_000},
    "model_facts": {"filename": "old.gguf", "size_label": "14B", "content_hash": "abc", "tensors": [{"shape": [64, 32], "quant": "Q4_K"}]},
    "workload": {"context": 4096, "prefill": 512, "objective": "steady_state_end_to_end_tok_s"},
    "candidates": candidate_rows(),
    "compiler_runtime_revision": {"compiler": "c1", "runtime": "r1"},
  }


def proof(samples=(100.0, 101.0, 99.0), **overrides):
  value = {
    "correctness": {"status": "PASS"}, "resource": {"status": "PASS"},
    "gpu_health": {"status": "PASS"}, "route_census": {"status": "PASS", "complete": True},
    "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": list(samples)},
  }
  value.update(overrides)
  return value


def test_key_ignores_filename_size_profile_and_candidate_order():
  a = inputs()
  b = copy.deepcopy(a)
  b["model_facts"].update(filename="renamed.gguf", model_name="renamed model", size_label="8B", profile="some_profile")
  b["candidates"].reverse()
  assert canonical_search_key(**a) == canonical_search_key(**b)
  b["model_facts"]["content_hash"] = "different"
  assert canonical_search_key(**a) != canonical_search_key(**b)


def test_key_invalidates_every_material_input_class():
  base = inputs()
  key = canonical_search_key(**base)
  mutations = [
    ("gpu_facts", "free_bytes", 19_999), ("model_facts", "content_hash", "def"),
    ("workload", "context", 8192), ("compiler_runtime_revision", "runtime", "r2"),
  ]
  for section, field, value in mutations:
    changed = copy.deepcopy(base); changed[section][field] = value
    assert canonical_search_key(**changed) != key
  changed = copy.deepcopy(base); changed["candidates"][1]["route"] = "new-tile"
  assert canonical_search_key(**changed) != key
  changed = copy.deepcopy(base)
  changed["candidates"][1]["production_eligibility_requirement"] = build_production_eligibility_requirement(
    authority={"schema": "test.cross_route_parity_requirement.v2"})
  assert canonical_search_key(**changed) != key


def test_accelerated_production_eligibility_fails_closed_but_direct_baseline_is_exempt():
  for mutation, reason in (
      (lambda gate: gate.clear(), "fields are malformed"),
      (lambda gate: gate.update(promotion_eligible=False), "identity mismatch"),
      (lambda gate: gate.update(eligibility(
        eligible=False, blocker="cross-route parity is not certified")), "is BLOCKED"),
      (lambda gate: gate.update(candidate_id="other"), "candidate binding mismatch"),
      (lambda gate: gate.update(authority={"schema": "other"}), "authority does not match"),
      (lambda gate: gate.update(extra=True), "fields are malformed"),
  ):
    args = inputs()
    gate = eligibility(args["candidates"][1]); mutation(gate)
    result = select_policy(
      **args, evidence={"baseline": proof((100, 100, 100)),
                        "tile": proof((200, 200, 200), production_eligibility=gate)},
      baseline_candidate_id="baseline")
    assert result["selected_candidate_id"] == "baseline"
    rejected = next(row for row in result["rejected_candidates"] if row["candidate_id"] == "tile")
    assert any(reason in item for item in rejected["reasons"])
  baseline = inputs()
  selected = select_policy(
    **baseline, evidence={"baseline": proof(),
                          "tile": proof((80, 80, 80), production_eligibility=eligibility())},
    baseline_candidate_id="baseline")
  assert selected["selected_candidate_id"] == "baseline"


def test_rejects_each_missing_gate_and_kernel_only_timing():
  args = inputs()
  gates = ("correctness", "resource", "gpu_health", "route_census")
  evidence = {"baseline": proof(), "tile": proof(production_eligibility=eligibility())}
  for gate in gates: evidence["tile"].pop(gate)
  evidence["baseline"]["end_to_end_timing"]["scope"] = "kernel"
  result = select_policy(**args, evidence=evidence, baseline_candidate_id="baseline")
  assert result["decision"] == "REFUSE"
  assert result["selected_candidate_id"] is None
  reasons = {r["candidate_id"]: r["reasons"] for r in result["rejected_candidates"]}
  assert "timing evidence must be end-to-end tok/s" in reasons["baseline"]
  assert all(any(gate in reason for reason in reasons["tile"]) for gate in gates)


def test_route_census_must_explicitly_attest_complete_coverage():
  incomplete = proof(route_census={"status": "PASS", "complete": False})
  result = select_policy(**inputs(), evidence={
    "baseline": incomplete,
    "tile": {**incomplete, "production_eligibility": eligibility()},
  }, baseline_candidate_id="baseline")
  assert result["decision"] == "REFUSE"
  assert all("route_census evidence does not attest complete coverage" in row["reasons"] for row in result["rejected_candidates"])


def test_selects_end_to_end_winner_and_records_stable_json():
  result = select_policy(**inputs(), evidence={"baseline": proof((100, 101, 99)),
    "tile": proof((120, 121, 119), production_eligibility=eligibility())},
                         baseline_candidate_id="baseline")
  assert result["decision"] == "SELECTED"
  assert result["selected_candidate_id"] == "tile"
  assert result["tie_candidate_ids"] == ["tile"]
  assert json.loads(json.dumps(result, sort_keys=True)) == result


def test_confidence_overlap_tie_prefers_named_baseline():
  baseline = proof((100, 100, 100), end_to_end_timing={"scope": "end_to_end", "metric": "tok_s", "samples": [100, 100, 100], "confidence_interval_tok_s": [98, 102]})
  tile = proof((101, 101, 101), end_to_end_timing={"scope": "end_to_end", "metric": "tok_s", "samples": [101, 101, 101], "confidence_interval_tok_s": [99, 103]})
  tile["production_eligibility"] = eligibility()
  result = select_policy(**inputs(), evidence={"baseline": baseline, "tile": tile}, baseline_candidate_id="baseline")
  assert result["selected_candidate_id"] == "baseline"
  assert result["tie_candidate_ids"] == ["baseline", "tile"]
  assert result["decision_reason"] == "statistical tie resolved deterministically"


def test_noisy_fast_candidate_is_rejected_and_safe_baseline_wins():
  result = select_policy(**inputs(), evidence={"baseline": proof((100, 100, 100)),
    "tile": proof((50, 200, 350), production_eligibility=eligibility())},
                         baseline_candidate_id="baseline", max_relative_noise=0.05)
  assert result["selected_candidate_id"] == "baseline"
  assert any("relative timing noise" in reason for reason in result["rejected_candidates"][0]["reasons"])


def test_cache_is_exact_keyed_and_malformed_cache_fails_closed():
  args = inputs()
  result = select_policy(**args, evidence={"baseline": proof(),
    "tile": proof((120, 120, 120), production_eligibility=eligibility())}, baseline_candidate_id="baseline")
  cache = make_cache_record(result)
  assert cache_matches(cache, **args)
  changed = copy.deepcopy(args); changed["workload"]["context"] += 1
  assert not cache_matches(cache, **changed)
  cache["result"]["search_key"] = "forged"
  assert not cache_matches(cache, **args)
