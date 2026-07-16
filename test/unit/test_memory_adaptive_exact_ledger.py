import pytest

from tinygrad.llm.admission import admit_exact_selected_model, plan_exact_selected_model_load, scanned_device_memory_budget
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.gguf_memory_scan import RuntimeGeometry
from tinygrad.llm.memory_ledger import (AllocationKind, AllocationProvenance, LedgerAllocation,
                                        ScannedMemoryBudget, SelectedModelMemoryLedger)


P = AllocationProvenance("unit inventory", "selected GGUF/runtime allocation observation")

def _budget(free, reserve):
  return ScannedMemoryBudget(free, reserve, AllocationProvenance("unit test", "explicit private test budget"))


def _ledger(*, tensor_payload=101, scratch=17, workspace=19):
  return SelectedModelMemoryLedger((
    LedgerAllocation.gguf_tensor("blk.0.attn_q.weight", tensor_payload, 64, 2, P),
    LedgerAllocation("kv", AllocationKind.KV_CACHE, 200, P),
    LedgerAllocation("runtime", AllocationKind.RUNTIME_PERSISTENT, 23, P),
    LedgerAllocation("activations", AllocationKind.PREFILL_ACTIVATION, 29, P),
    LedgerAllocation("outputs", AllocationKind.PREFILL_OUTPUT, 31, P),
    LedgerAllocation("scratch", AllocationKind.PREFILL_SCRATCH, scratch, P),
    LedgerAllocation("workspace:direct", AllocationKind.CANDIDATE_WORKSPACE, workspace, P, candidate_id="direct"),
  ))


def test_tensor_allocation_accounts_for_alignment_and_duplication():
  tensor = _ledger().allocations[0]
  assert tensor.payload_bytes == 101
  assert tensor.bytes == 256  # align_up(101, 64) * two resident copies
  decision = admit_exact_selected_model(_ledger(), _budget(1000, 100))[0]
  assert decision.admitted
  assert decision.peak_bytes == 256 + 200 + 23 + 29 + 31 + 17 + 19


def test_unknown_bytes_fail_closed_with_provenance_preserved():
  decision = admit_exact_selected_model(_ledger(scratch=None), _budget(1000, 0))[0]
  assert not decision.admitted and decision.peak_bytes is None
  assert "unknown allocation bytes: scratch" in decision.reasons
  assert decision.to_dict()["allocations"][5]["provenance"]["source"] == "unit inventory"


def test_omitted_required_class_is_not_treated_as_zero():
  ledger = SelectedModelMemoryLedger(tuple(x for x in _ledger().allocations if x.kind is not AllocationKind.PREFILL_OUTPUT))
  decision = admit_exact_selected_model(ledger, _budget(1000, 0))[0]
  assert not decision.admitted
  assert "missing exact allocation class: prefill_output" in decision.reasons


def test_candidate_workspace_is_scoped_and_exact_budget_boundary_is_inclusive():
  peak = admit_exact_selected_model(_ledger(), _budget(10_000, 0))[0].peak_bytes
  assert peak is not None
  assert admit_exact_selected_model(_ledger(), _budget(peak, 0))[0].admitted
  refused = admit_exact_selected_model(_ledger(), _budget(peak-1, 0))[0]
  assert not refused.admitted
  assert refused.reasons == (f"exact peak {peak} exceeds scanned-memory budget {peak-1} by 1 bytes",)


def test_invalid_claimed_tensor_allocation_is_rejected():
  with pytest.raises(ValueError, match="after alignment/duplication"):
    LedgerAllocation("tensor:x", AllocationKind.GGUF_TENSOR, 101, P, tensor_name="x",
                     payload_bytes=101, alignment=64, copies=2)


def test_load_plan_uses_single_device_granularity_and_explicit_route_facts(tmp_path):
  probe = ProbeRecord("single scan", "2026-07-15T00:00:00+00:00")
  facts = DeviceFacts("AMD", "AMD", "gfx", 10_000, 10_000,
                      DeviceCapabilities(global_allocation_granularity=64), probe, probe)
  metadata = ({}, {"data_start": 128, "tensor_infos": [("weight", (4, 8), 1, 0)]})
  geometry = RuntimeGeometry(1, 1, 4, 8, 2, batch_size=1, kv_element_bytes=2,
    runtime_persistent_bytes=3, peak_prefill_activation_bytes=4, peak_prefill_output_bytes=5,
    peak_prefill_scratch_bytes=6)
  route = {"candidate_id": "accelerated", "resident_copies": 2, "candidate_workspace_bytes": 7,
           "provenance": "measured complete route"}
  plan = plan_exact_selected_model_load(tmp_path/"selected.gguf", metadata=metadata, geometry=geometry,
                                         route_memory_facts=route, facts=facts)
  tensor = plan.scan.ledger.allocations[0]
  assert tensor.alignment == 64 and tensor.copies == 2 and tensor.bytes == 128
  assert plan.decision.admitted
  with pytest.raises((AttributeError, TypeError)):
    plan.decision.admitted = False


def test_accelerated_load_plan_fails_closed_when_route_copy_or_runtime_fact_is_unknown(tmp_path):
  probe = ProbeRecord("single scan", "2026-07-15T00:00:00+00:00")
  facts = DeviceFacts("AMD", "AMD", "gfx", 10_000, 10_000,
                      DeviceCapabilities(global_allocation_granularity=64), probe, probe)
  metadata = ({}, {"data_start": 64, "tensor_infos": [("weight", (4, 8), 1, 0)]})
  geometry = RuntimeGeometry(1, 1, 4, 8, 2, batch_size=1, kv_element_bytes=2,
    runtime_persistent_bytes=None, peak_prefill_activation_bytes=4, peak_prefill_output_bytes=5,
    peak_prefill_scratch_bytes=6)
  plan = plan_exact_selected_model_load(tmp_path/"selected.gguf", metadata=metadata, geometry=geometry,
    route_memory_facts={"candidate_id": "accelerated", "resident_copies": None,
                        "candidate_workspace_bytes": 0, "provenance": "incomplete route"},
    facts=facts)
  assert not plan.decision.admitted
  assert "unknown allocation bytes: tensor:weight" in plan.decision.reasons
  assert "unknown allocation bytes: runtime_persistent" in plan.decision.reasons


def test_production_budget_reserve_is_derived_from_live_scan_not_a_vram_tier():
  probe = ProbeRecord("single scan", "2026-07-15T00:00:00+00:00")
  facts = DeviceFacts("AMD", "AMD", "gfx", 10_000, 8_901,
                      DeviceCapabilities(global_allocation_granularity=64), probe, probe)
  budget = scanned_device_memory_budget(facts)
  assert budget.reserve_bytes == 1_152  # align_up(total-free=1099, allocator granularity=64)
  assert budget.admitted_bytes == 7_749
  assert "live occupied-byte" in budget.provenance.detail


def test_production_budget_fails_closed_without_scanned_allocator_granularity():
  probe = ProbeRecord("single scan", "2026-07-15T00:00:00+00:00")
  facts = DeviceFacts("AMD", "AMD", "gfx", 10_000, 8_901, DeviceCapabilities(), probe, probe)
  budget = scanned_device_memory_budget(facts)
  assert budget.free_bytes is None and budget.reserve_bytes is None and budget.admitted_bytes is None
