import copy

import pytest

from extra.qk.memory_adaptive_autoscan import AutoscanCandidate, autoscan_selected_model
from extra.qk.memory_adaptive_policy import (
  build_production_eligibility, build_production_eligibility_requirement,
)
from tinygrad.llm.device_facts import (
  DeviceCapabilities, DeviceFacts, MalformedDeviceFactsError, ProbeRecord, UnsupportedDeviceFactsSchemaError,
)
from tinygrad.llm.prefill_memory_plan import (ByteLifetime, ByteTerm, CandidateMemoryCoverage, Strategy)


def term(name, size, lifetime=ByteLifetime.PERSISTENT):
  return ByteTerm(name, size, "selected model inventory", f"sum({name})", lifetime)


def device(free, *, queue_mode="PM4", schema_version=2):
  probe = ProbeRecord("fake", "2026-07-15T00:00:00+00:00")
  return DeviceFacts("AMD:0", "AMD", "gfx", 1000, free, DeviceCapabilities(wave_size=32), probe, probe,
                     schema_version=schema_version, queue_mode=queue_mode)


def candidate(cid, strategy, workspace):
  memory = CandidateMemoryCoverage(cid, strategy, (term("workspace", workspace, ByteLifetime.CANDIDATE_WORKSPACE),),
                                   ("row",), ("row",))
  policy = {"candidate_id": cid, "route": strategy.value}
  if strategy is not Strategy.DIRECT_PACKED_FALLBACK:
    policy["production_eligibility_requirement"] = build_production_eligibility_requirement(
      authority={"schema": "test.autoscan_production_authority.v1"})
  return AutoscanCandidate(memory, policy)


def proof(speed=100, candidate=None):
  row = {"correctness": {"status": "PASS"}, "resource": {"status": "PASS"},
         "gpu_health": {"status": "PASS"}, "route_census": {"status": "PASS", "complete": True},
         "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [speed]*3}}
  if candidate is not None and candidate.memory.strategy is not Strategy.DIRECT_PACKED_FALLBACK:
    policy = candidate.policy_record()
    row["production_eligibility"] = build_production_eligibility(
      candidate=policy, promotion_eligible=True,
      authority=policy["production_eligibility_requirement"]["authority"])
  return row


def args(free=200):
  return dict(selected_model_facts={"content_hash": "abc", "filename": "one.gguf", "size_label": "14B"},
              selected_model_inventory={"rows": [{"tensor": "x", "shape": [4, 8], "quant": "Q4_K"}]},
              base_terms=(term("packed", 80),),
              candidates=(candidate("baseline", Strategy.DIRECT_PACKED_FALLBACK, 10),
                          candidate("overlay", Strategy.FULL_RESIDENT_OVERLAY, 100)),
              workload={"context": 128}, compiler_runtime_revision={"runtime": "r1"},
              baseline_candidate_id="baseline", device_facts=device(free))


def test_runs_only_feasible_candidates_and_free_vram_changes_feasibility():
  seen = []
  low = autoscan_selected_model(**args(120), evidence_runner=lambda c: seen.append(c.candidate_id) or proof(candidate=c))
  assert seen == ["baseline"] and low["selected_candidate_id"] == "baseline"
  seen.clear()
  high = autoscan_selected_model(**args(250), evidence_runner=lambda c: seen.append(c.candidate_id) or
    proof(120 if c.candidate_id == "overlay" else 100, candidate=c))
  assert seen == ["baseline", "overlay"] and high["selected_candidate_id"] == "overlay"


def test_rename_is_nonsemantic_and_exact_cache_skips_runner():
  first = autoscan_selected_model(**args(250), evidence_runner=lambda c: proof(candidate=c))
  renamed = args(250); renamed["selected_model_facts"] = copy.deepcopy(renamed["selected_model_facts"])
  renamed["selected_model_facts"].update(filename="other.gguf", size_label="8B", profile="renamed")
  second = autoscan_selected_model(**renamed, cache_record=first["cache_record"],
                                   evidence_runner=lambda c: (_ for _ in ()).throw(AssertionError("runner called")))
  assert second["from_cache"] and second["selected_candidate_id"] == first["selected_candidate_id"]


def test_queue_mode_changes_exact_cache_identity_and_runs_search_again():
  first = autoscan_selected_model(**args(250), evidence_runner=lambda c: proof(candidate=c))
  changed = args(250); changed["device_facts"] = device(250, queue_mode="AQL")
  seen = []
  second = autoscan_selected_model(**changed, cache_record=first["cache_record"],
                                   evidence_runner=lambda c: seen.append(c.candidate_id) or proof(candidate=c))
  assert not second["from_cache"] and seen == ["baseline", "overlay"]
  assert first["cache_record"]["search_key"] != second["cache_record"]["search_key"]


def test_supplied_v1_facts_trigger_one_explicit_reprobe_before_cache_or_execution():
  values = args(250)
  values["device_facts"] = device(250, queue_mode=None, schema_version=1)
  scans = []
  result = autoscan_selected_model(
    **values, device_scanner=lambda *, selected_device: scans.append(selected_device) or device(250, queue_mode="AQL"),
    evidence_runner=lambda c: proof(candidate=c))
  assert scans == ["AMD:0"] and result["decision"] == "SELECTED"
  assert result["cache_record"]["result"]["canonical_inputs"]["gpu_facts"]["queue_mode"] == "AQL"


@pytest.mark.parametrize("facts,error", (
  (device(250, queue_mode=None, schema_version=2), MalformedDeviceFactsError),
  (device(250, queue_mode="PM4", schema_version=3), UnsupportedDeviceFactsSchemaError),
))
def test_current_malformed_or_future_facts_fail_without_reprobe(facts, error):
  values = args(250); values["device_facts"] = facts
  scans = []
  with pytest.raises(error):
    autoscan_selected_model(
      **values, device_scanner=lambda *, selected_device: scans.append(selected_device) or device(250),
      evidence_runner=lambda c: (_ for _ in ()).throw(AssertionError("runner called")))
  assert scans == []


def test_interruption_returns_only_guarded_safe_baseline_or_refuses():
  def interrupted(candidate):
    if candidate.candidate_id == "baseline": return proof()
    return None
  result = autoscan_selected_model(**args(250), evidence_runner=interrupted)
  assert result["decision"] == "SELECTED" and result["selected_candidate_id"] == "baseline" and result["interrupted"]
  assert result["cache_record"] is None

  bad = proof(); bad["gpu_health"] = {"status": "FAIL"}
  refused = autoscan_selected_model(**args(250), evidence_runner=lambda c: bad if c.candidate_id == "baseline" else None)
  assert refused["decision"] == "REFUSE" and refused["selected_candidate_id"] is None


def test_unknown_device_memory_refuses_without_execution():
  values = args(); values["device_facts"] = device(None)
  result = autoscan_selected_model(**values, evidence_runner=lambda c: (_ for _ in ()).throw(AssertionError("runner called")))
  assert result["decision"] == "REFUSE"


def test_whole_policy_identity_is_read_only_and_derived_from_policy():
  value = candidate("baseline", Strategy.DIRECT_PACKED_FALLBACK, 10)
  value.policy["whole_policy_identity"] = "whole-policy:sha256:test"
  assert value.whole_policy_identity == "whole-policy:sha256:test"
  with pytest.raises(AttributeError): value.whole_policy_identity = "forged"
