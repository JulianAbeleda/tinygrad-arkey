import inspect, json

import pytest

from extra.qk.memory_adaptive_candidate_catalog import CandidateSpec
import extra.qk.memory_adaptive_search_controller as controller
from extra.qk.memory_adaptive_search_controller import SelectedModelScan, _run_controller_with_seam, main, run_controller
from extra.qk.memory_adaptive_transport import SelectedModelScan as TransportSelectedModelScan
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.prefill_memory_plan import ByteLifetime, ByteTerm, Strategy
from extra.qk.memory_adaptive_allocation_observer import EXACT_MEMORY_KEYS, make_memory_facts

INVENTORY_IDENTITY = "inventory:sha256:" + "a" * 64


def test_selected_model_scan_public_import_is_transport_identity_and_positional():
  assert SelectedModelScan is TransportSelectedModelScan
  values = ({"content_hash": "x"}, {"inventory_identity": INVENTORY_IDENTITY, "rows": []},
            (), {"prompt_tokens": 1}, {"git": "abc"})
  scan = SelectedModelScan(*values)
  assert tuple(getattr(scan, field) for field in scan.__dataclass_fields__) == values


def _device(free=900):
  probe = ProbeRecord("fake-live-probe", "2026-07-15T00:00:00+00:00")
  return DeviceFacts("AMD:0", "AMD", "gfx-test", 1000, free,
                     DeviceCapabilities(wave_size=32, global_allocation_granularity=64), probe, probe,
                     queue_mode="PM4")


@pytest.fixture(autouse=True)
def _live_scanner(monkeypatch):
  monkeypatch.setattr(controller, "scan_device_facts", lambda: _device())


def _artifacts(ids, speed, whole_policy_identity):
  return {"actual_whole_model_run": True, "artifacts": {
    "whole_policy_identity": whole_policy_identity,
    "execution": {"phases": [
      {"phase": "compile", "status": "passed", "evidence": {}},
      {"phase": "execution", "status": "passed", "evidence": {"dispatch_state": "completed",
        "health": {"preflight": True, "postflight": True, "device_fault": False}}},
      {"phase": "correctness", "status": "passed", "evidence": {"full_output_compared": True,
        "numerical_passed": True, "finite_output": True, "inputs_unchanged": True}}]},
    "resource": {"status": "PASS"},
    "route_census": {"status": "PASS", "complete": True, "covered_invocations": ids,
      "whole_policy_identity": whole_policy_identity},
    "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [speed]*3}}}


def _memory(candidate_id):
  facts = {key: (1 if key in ("resident_copies", "batch_size", "kv_element_bytes") else 0) for key in EXACT_MEMORY_KEYS}
  provenance = {key: {"source": "synthetic allocation test fixture", "detail": f"explicit {key} measurement"} for key in EXACT_MEMORY_KEYS}
  return make_memory_facts(candidate_id, facts, provenance)


class Seam:
  def __init__(self): self.calls = []
  def scan_selected_model(self, path, device):
    return SelectedModelScan({"content_hash": "sha256:model", "model_name": path},
      {"inventory_identity": INVENTORY_IDENTITY,
       "rows": [{"invocation_id": "a", "shape": [2, 4]}, {"invocation_id": "b", "shape": [4, 8]}]},
      (ByteTerm("model", 100, "GGUF inventory", "sum tensor bytes", ByteLifetime.PERSISTENT),),
      {"prompt_tokens": 32}, {"git": "abc"})
  def enumerate_candidate_specs(self, model, device):
    ids = ("a", "b")
    return [CandidateSpec("fast", Strategy.FULL_RESIDENT_OVERLAY, ids,
              (ByteTerm("overlay", 100, "candidate", "exact", ByteLifetime.CANDIDATE_WORKSPACE),),
              full_m_values=(16, 32), tail_m_values=(), correctness_m_values=(16, 32),
              invocation_bytes=({"m": 16, "activation_bytes": 8, "scratch_bytes": 4},
                                {"m": 32, "activation_bytes": 16, "scratch_bytes": 8})),
            CandidateSpec("partial", Strategy.BOUNDED_PACKED_TILES, ("a",), full_m_values=(32,),
              correctness_m_values=(32,), invocation_bytes=({"m": 32, "activation_bytes": 1, "scratch_bytes": 1},)),
            CandidateSpec("baseline", Strategy.DIRECT_PACKED_FALLBACK, ids,
              (ByteTerm("scratch", 1, "candidate", "exact", ByteLifetime.CANDIDATE_WORKSPACE),),
              full_m_values=(32,), correctness_m_values=(32,),
              invocation_bytes=({"m": 32, "activation_bytes": 1, "scratch_bytes": 1},))]
  def collect_whole_model_artifacts(self, path, model, candidate, *, samples):
    self.calls.append((candidate.candidate_id, samples))
    row = _artifacts(list(candidate.memory.required_invocations),
      120 if candidate.policy["policy_candidate_id"] == "fast" else 100, candidate.whole_policy_identity)
    if candidate.memory.strategy is not Strategy.DIRECT_PACKED_FALLBACK: row["memory_fact_evidence"] = _memory(candidate.candidate_id)
    return row


def test_complete_feasible_policies_run_baseline_first_and_cache_exact_facts():
  seam = Seam()
  first = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  assert seam.calls == [("baseline:M32", 3), ("fast:M16", 3), ("fast:M32", 3)]
  assert first["decision"] == "SELECTED" and first["selected_candidate_id"] in ("fast:M16", "fast:M32")
  selected = next(x for x in first["cache_record"]["result"]["canonical_inputs"]["candidates"]
                  if x["candidate_id"] == first["selected_candidate_id"])
  assert set(selected["memory_facts"]) == set(EXACT_MEMORY_KEYS)
  seam.calls.clear()
  second = _run_controller_with_seam(model_path="renamed.gguf", seam=seam, cache_record=first["cache_record"])
  assert second["from_cache"] and seam.calls == []


def test_workload_expansion_has_distinct_semantic_identities_in_policy_and_cache():
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=Seam())
  candidates = result["cache_record"]["result"]["canonical_inputs"]["candidates"]
  fast = {x["candidate_id"]: x for x in candidates if x["policy_candidate_id"] == "fast"}
  assert set(fast) == {"fast:M16", "fast:M32"}
  assert fast["fast:M16"]["whole_policy_identity"] != fast["fast:M32"]["whole_policy_identity"]
  assert all(x["whole_policy_identity"].startswith("whole-policy:sha256:") for x in fast.values())


def test_accelerated_candidate_without_complete_measured_facts_is_rejected():
  seam = Seam()
  seam.collect_whole_model_artifacts = lambda path, model, candidate, samples: _artifacts(
    list(candidate.memory.required_invocations), 120 if candidate.memory.strategy is not Strategy.DIRECT_PACKED_FALLBACK else 100,
    candidate.whole_policy_identity)
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  assert result["selected_candidate_id"] == "baseline:M32"


def test_incomplete_or_unattested_evidence_never_selects():
  seam = Seam()
  seam.collect_whole_model_artifacts = lambda *args, **kwargs: {"actual_whole_model_run": False, "artifacts": {}}
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  assert result["decision"] == "REFUSE" and result["selected_candidate_id"] is None
  assert "evidence missing" in str(result["policy"]["rejected_candidates"])


def test_evidence_exception_refuses_with_bounded_candidate_diagnostic_and_no_cache():
  seam = Seam()
  def fail(path, model, candidate, *, samples):
    raise RuntimeError("whole-model boom\nwithout traceback details")
  seam.collect_whole_model_artifacts = fail
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  diagnostic = result["candidate_diagnostics"]["baseline:M32"]
  assert result["decision"] == "REFUSE" and result["interrupted"] is True
  assert result["policy"] is None and result["cache_record"] is None
  assert diagnostic == {"candidate_id": "baseline:M32", "actual_whole_model_run": False,
    "exception_type": "RuntimeError", "exception_message": "whole-model boom without traceback details",
    "blockers": ["whole-model evidence exception: RuntimeError"], "memory_fact_evidence": False}
  assert seam.calls == []


def test_incomplete_diagnostic_projects_bounded_structure_and_aggregates_schedule_failures():
  seam = Seam()
  def incomplete(path, model, candidate, *, samples):
    return {"actual_whole_model_run": True,
      "blockers": ["schedule manifest 0: ownership secret=/tmp/private", "schedule manifest 0: ownership another",
                   "schedule manifest 2: physical mismatch", "traceback /home/private/file.py"],
      "measured_allocation": {"physical_ledger": {"structural_summary": {
        "count_by_phase": {"compute": 4}, "bytes_by_category": {"workspace": 128},
        "binding_presence": {"bound": 3}, "reuse": {"reused": 2},
        "lifetimes": [{"secret": "do-not-copy"}], "unbounded": {str(x): x for x in range(1000)}}}},
      "physical_memory_ledger": {"structural_summary": {"count_by_category": {"fallback": 99}}},
      "artifacts": {}}
  seam.collect_whole_model_artifacts = incomplete
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  diagnostic = result["candidate_diagnostics"]["baseline:M32"]
  assert result["decision"] == "REFUSE" and diagnostic["memory_fact_evidence"] is False
  assert diagnostic["blocker_count"] == 4
  assert diagnostic["schedule_failure_summary"] == [
    {"manifest": 0, "category": "ownership", "count": 2},
    {"manifest": 2, "category": "physical_ledger", "count": 1},
    {"manifest": None, "category": "other", "count": 1}]
  assert diagnostic["physical_structural_summary"] == {
    "binding_presence": {"bound": 3}, "bytes_by_category": {"workspace": 128},
    "count_by_phase": {"compute": 4}, "reuse": {"reused": 2}}
  assert "secret" not in json.dumps(diagnostic) and "private" not in json.dumps(diagnostic)
  assert len(json.dumps(diagnostic)) < 2000


def test_malformed_or_missing_structural_summary_is_safe_and_bounded():
  seam = Seam()
  seam.collect_whole_model_artifacts = lambda *args, **kwargs: {
    "actual_whole_model_run": False, "blockers": ["missing summary"], "measured_allocation": {
      "structural_summary": ["not-a-summary"]}, "artifacts": {}}
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  diagnostic = result["candidate_diagnostics"]["baseline:M32"]
  assert diagnostic["physical_structural_summary"] is None
  assert diagnostic["schedule_failure_summary"] == [{"manifest": None, "category": "other", "count": 1}]


def test_requires_three_samples_and_exactly_one_complete_baseline():
  seam = Seam()
  try: _run_controller_with_seam(model_path="chosen.gguf", seam=seam, min_samples=2)
  except ValueError as exc: assert "at least 3" in str(exc)
  else: raise AssertionError("min_samples=2 was accepted")
  seam.enumerate_candidate_specs = lambda model, device: [CandidateSpec("only", Strategy.FULL_RESIDENT_OVERLAY, ("a", "b"),
    full_m_values=(32,), correctness_m_values=(32,), invocation_bytes=({"m": 32, "activation_bytes": 1, "scratch_bytes": 1},))]
  assert "exactly one" in _run_controller_with_seam(model_path="chosen.gguf", seam=seam)["reason"]


def test_partial_remainder_or_unknown_per_m_bytes_never_reaches_timing():
  seam = Seam()
  seam.scan_selected_model = lambda path, device: SelectedModelScan({"content_hash": "x"},
    {"inventory_identity": INVENTORY_IDENTITY, "rows": [{"invocation_id": "a"}, {"invocation_id": "b"}]},
    (ByteTerm("model", 100, "scan", "exact", ByteLifetime.PERSISTENT),),
    {"prompt_tokens": 33, "context_tokens": 64}, {"git": "abc"})
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  assert result["decision"] == "REFUSE"
  assert seam.calls == []


def test_cli_uses_internal_production_seam(monkeypatch, capsys):
  seam = Seam()
  monkeypatch.setattr(controller, "_production_seam", lambda: seam)
  assert main(["--model", "chosen.gguf"]) == 0
  row = json.loads(capsys.readouterr().out)
  assert row["decision"] == "SELECTED" and seam.calls


def test_hardware_injection_is_absent_from_public_controller_and_cli():
  parameters = inspect.signature(run_controller).parameters
  assert not {"seam", "selected_device", "device_facts", "reserve_policy"} & parameters.keys()
  with pytest.raises(SystemExit): main(["--model", "chosen.gguf", "--device", "AMD:7"])
  with pytest.raises(SystemExit): main(["--model", "chosen.gguf", "--seam", "anything:SEAM"])


def test_one_live_scan_derives_dynamic_reserve_for_admission(monkeypatch):
  calls = []
  monkeypatch.setattr(controller, "scan_device_facts", lambda: calls.append(True) or _device(free=901))
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=Seam())
  assert calls == [True]
  assert result["memory_plan"]["device"]["safety_reserve"]["bytes"] == 128  # align_up(1000-901, 64)
  assert result["memory_plan"]["device"]["safety_reserve"]["formula"] == \
    "align_up(total_vram_bytes - free_vram_bytes, scanned_allocator_granularity)"
  assert "live selected-device scan" in result["memory_plan"]["device"]["safety_reserve"]["provenance"]


def test_unknown_scanned_allocator_granularity_fails_closed(monkeypatch):
  facts = _device()
  monkeypatch.setattr(controller, "scan_device_facts", lambda: DeviceFacts(facts.selected_device, facts.backend,
    facts.architecture, facts.total_vram_bytes, facts.free_vram_bytes, DeviceCapabilities(wave_size=32),
    facts.target_probe, facts.memory_probe, queue_mode=facts.queue_mode))
  seam = Seam()
  result = _run_controller_with_seam(model_path="chosen.gguf", seam=seam)
  assert result["decision"] == "REFUSE" and "allocator granularity" in result["reason"]
  assert seam.calls == []
