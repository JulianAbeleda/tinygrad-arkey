import copy

import pytest

from extra.qk.memory_adaptive_autoscan import AutoscanCandidate, autoscan_selected_model
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.prefill_memory_plan import (ByteLifetime, ByteTerm, CandidateMemoryCoverage, Strategy)


def term(name, size, lifetime=ByteLifetime.PERSISTENT):
  return ByteTerm(name, size, "selected model inventory", f"sum({name})", lifetime)


def device(free):
  probe = ProbeRecord("fake", "2026-07-15T00:00:00+00:00")
  return DeviceFacts("AMD:0", "AMD", "gfx", 1000, free, DeviceCapabilities(wave_size=32), probe, probe)


def candidate(cid, strategy, workspace):
  memory = CandidateMemoryCoverage(cid, strategy, (term("workspace", workspace, ByteLifetime.CANDIDATE_WORKSPACE),),
                                   ("row",), ("row",))
  return AutoscanCandidate(memory, {"candidate_id": cid, "route": strategy.value})


def proof(speed=100):
  return {"correctness": {"status": "PASS"}, "resource": {"status": "PASS"},
          "gpu_health": {"status": "PASS"}, "route_census": {"status": "PASS", "complete": True},
          "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [speed]*3}}


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
  low = autoscan_selected_model(**args(120), evidence_runner=lambda c: seen.append(c.candidate_id) or proof())
  assert seen == ["baseline"] and low["selected_candidate_id"] == "baseline"
  seen.clear()
  high = autoscan_selected_model(**args(250), evidence_runner=lambda c: seen.append(c.candidate_id) or proof(120 if c.candidate_id == "overlay" else 100))
  assert seen == ["baseline", "overlay"] and high["selected_candidate_id"] == "overlay"


def test_rename_is_nonsemantic_and_exact_cache_skips_runner():
  first = autoscan_selected_model(**args(250), evidence_runner=lambda c: proof())
  renamed = args(250); renamed["selected_model_facts"] = copy.deepcopy(renamed["selected_model_facts"])
  renamed["selected_model_facts"].update(filename="other.gguf", size_label="8B", profile="renamed")
  second = autoscan_selected_model(**renamed, cache_record=first["cache_record"],
                                   evidence_runner=lambda c: (_ for _ in ()).throw(AssertionError("runner called")))
  assert second["from_cache"] and second["selected_candidate_id"] == first["selected_candidate_id"]


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
