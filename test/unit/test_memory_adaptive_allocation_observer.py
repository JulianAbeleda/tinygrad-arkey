import pytest

from extra.qk.memory_adaptive_allocation_observer import (AllocationObserver, EXACT_MEMORY_KEYS,
  derive_memory_facts, make_memory_facts, validate_memory_facts)


class Counter:
  def __init__(self): self.values = {"AMD:0": 10}
  def __call__(self): return dict(self.values)


def probe(): return {"card0": {"total_bytes": 1000, "used_bytes": 100}}


def test_records_boundaries_peak_identities_and_cleanup():
  counter = Counter()
  observer = AllocationObserver(["AMD:0"], probe=probe, counter_source=counter, poll_interval=100)
  observer.start()
  counter.values["AMD:0"] += 40
  observer.post_load()
  with observer.phase("prefill"):
    counter.values["AMD:0"] += 20
    observer.checkpoint("prefill_kv")
    counter.values["AMD:0"] += 5
    observer.checkpoint("prefill_workspace")
    counter.values["AMD:0"] -= 25
  counter.values["AMD:0"] -= 40
  evidence = observer.stop()
  assert evidence["complete"] and evidence["peak_bytes"] == 75 and evidence["post_run_retained_bytes"] == 0
  assert "allocations" not in evidence and "no allocation ownership attribution" in evidence["authority"]
  assert {x["phase"] for x in evidence["checkpoints"]} >= {"pre_load", "post_load", "prefill_begin", "prefill_end", "post_run_cleanup"}
  assert all(x["device_vram"] == probe() for x in evidence["checkpoints"])


def test_untagged_growth_is_valid_checkpoint_evidence_but_missing_probe_fails_closed():
  counter = Counter()
  def broken(): raise FileNotFoundError("rocm-smi")
  observer = AllocationObserver(probe=broken, counter_source=counter, poll_interval=100).start()
  counter.values["AMD:0"] += 7
  observer.checkpoint("peak")
  counter.values["AMD:0"] -= 7
  evidence = observer.stop()
  assert not evidence["complete"]
  assert evidence["peak_growth_bytes"] == 7 and evidence["post_run_retained_bytes"] == 0
  assert any("device probe unavailable" in x for x in evidence["blockers"])
  assert not any("attribution" in x for x in evidence["blockers"])


def test_measured_peak_above_exact_plan_fails_closed():
  counter = Counter()
  observer = AllocationObserver(probe=None, counter_source=counter, poll_interval=100, planned_peak_bytes=15).start()
  counter.values["AMD:0"] += 6
  observer.checkpoint("peak")
  counter.values["AMD:0"] -= 6
  evidence = observer.stop()
  assert evidence["peak_bytes"] == 16 and evidence["planned_peak_bytes"] == 15
  assert not evidence["complete"]
  assert any("measured peak 16 exceeds planned peak 15" in x for x in evidence["blockers"])


def test_nonzero_retained_counter_bytes_fail_checkpoint_completeness():
  counter = Counter()
  evidence = AllocationObserver(probe=None, counter_source=counter, poll_interval=100).start()
  counter.values["AMD:0"] += 1
  evidence = evidence.stop()
  assert not evidence["complete"] and evidence["post_run_retained_bytes"] == 1
  assert any("post-run cleanup retained" in x for x in evidence["blockers"])


def test_rocm_parser(monkeypatch):
  from extra.qk import memory_adaptive_allocation_observer as module
  class Result:
    stdout = "GPU[0] : VRAM Total Memory (B): 1024\nGPU[0] : VRAM Used Memory (B): 256\n"
  monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Result())
  assert module.rocm_smi_vram_probe() == {"card0": {"total_bytes": 1024, "used_bytes": 256}}


def test_memory_fact_bundle_binds_all_values_provenance_and_candidate():
  facts = {key: 1 for key in EXACT_MEMORY_KEYS}
  provenance = {key: {"source": "allocator", "detail": key} for key in EXACT_MEMORY_KEYS}
  bundle = make_memory_facts("candidate", facts, provenance)
  assert validate_memory_facts(bundle, candidate_id="candidate") == bundle
  forged = dict(bundle); forged["facts"] = {**facts, "batch_size": 2}
  assert validate_memory_facts(forged, candidate_id="candidate") is None
  partial = dict(facts); partial.pop("peak_prefill_output_bytes")
  with pytest.raises(ValueError): make_memory_facts("candidate", partial, provenance)


def test_derives_bytes_only_from_explicit_attributed_rows_and_rejects_unknown():
  structure = {"resident_copies": 1, "batch_size": 2, "kv_element_bytes": 1, "provenance": {
    key: {"source": "selected runtime config", "detail": key} for key in ("resident_copies", "batch_size", "kv_element_bytes")}}
  kinds = ("candidate_workspace", "runtime_persistent", "prefill_activation", "prefill_output", "prefill_scratch")
  rows = [{"identity": kind, "kind": kind, "candidate_id": "c" if kind == "candidate_workspace" else None, "bytes": i}
          for i, kind in enumerate(kinds)]
  evidence = {"schema": "tinygrad.reconciled_measured_allocation.v1", "complete": True, "allocations": rows}
  bundle = derive_memory_facts("c", structure, evidence)
  assert bundle["facts"]["candidate_workspace_bytes"] == 0
  bad = {**evidence, "complete": False, "allocations": rows + [{"identity": "u", "kind": "unknown", "bytes": 3}]}
  with pytest.raises(ValueError, match="incomplete"): derive_memory_facts("c", structure, bad)
