from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context
import gc, weakref
import json

import pytest

from tinygrad import Tensor, UOp
from extra.qk.physical_memory_ledger import PhysicalMemoryLedger, allocation_phase
from tinygrad.llm.physical_memory_ledger import (AllocationOwner, allocation_owner, allocation_owner_from_semantic, bind_allocation_owner)
from tinygrad.llm.memory_semantics import (MODEL_PARAMETER, PREFILL_SCRATCH, RUNTIME_OUTPUT, candidate_workspace)
from tinygrad.uop.ops import buffers


def _buf(device="CPU", nbytes=16): return Buffer(device, nbytes, dtypes.uint8)


def test_base_and_view_share_identity_and_view_adds_zero_bytes():
  ledger, base = PhysicalMemoryLedger(), _buf()
  with ledger.active(), allocation_owner(kind="weight", lifetime="model", semantic_owner_id="w0"):
    base.allocate()
    view = base.view(4, dtypes.uint8, 2).allocate()
    assert ledger.identity(view) == ledger.identity(base)
    view.deallocate()
    base.deallocate()
  assert [(x.event, x.requested_nbytes) for x in ledger.events] == [("alloc", 16), ("view", 0), ("view_free", 0), ("free", 16)]
  assert ledger.events[0].kind == "weight" and ledger.events[0].semantic_owner_id == "w0"
  assert ledger.complete


def test_reallocation_gets_new_lifetime_even_when_allocator_cache_reuses_storage():
  ledger, base = PhysicalMemoryLedger(), _buf(nbytes=33)
  with Context(LRU=1), ledger.active(), allocation_owner(kind="scratch", lifetime="step"):
    base.allocate(); first = ledger.identity(base); base.deallocate()
    base.allocate(); second = ledger.identity(base); base.deallocate()
  assert first != second
  assert [x.allocation_id for x in ledger.events if x.event == "alloc"] == [first, second]
  assert ledger.complete


def test_unowned_and_conflicting_ownership_are_incomplete():
  unowned, a = PhysicalMemoryLedger(), _buf(nbytes=7)
  with unowned.active(): a.allocate(); a.deallocate()
  assert not unowned.complete and "no explicit ownership" in unowned.issues[0]

  conflict, b = PhysicalMemoryLedger(), _buf(nbytes=9)
  with conflict.active(), allocation_owner(kind="kv", lifetime="model"):
    with allocation_owner(kind="workspace", lifetime="candidate", candidate_id="c0"):
      b.allocate()
    b.deallocate()
  assert not conflict.complete and "conflicting ownership" in conflict.issues[0]
  assert conflict.events[0].owner is None


def test_missing_free_is_incomplete():
  ledger, base = PhysicalMemoryLedger(), _buf()
  with ledger.active(), allocation_owner(kind="weight", lifetime="model"): base.allocate()
  assert not ledger.complete and "no free event" in ledger.issues[-1]
  base.deallocate()


def test_mapped_devices_have_distinct_physical_base_identities(monkeypatch):
  ledger, base = PhysicalMemoryLedger(), _buf()
  target = Device["CPU:1"].allocator
  monkeypatch.setattr(target, "_map", lambda opaque: opaque)
  monkeypatch.setattr(target, "_unmap", lambda opaque: None)
  with ledger.active(), allocation_owner(kind="weight", lifetime="model"):
    base.allocate()
    base.get_buf("CPU:1")
    assert ledger.identity(base, "CPU:1") != ledger.identity(base, "CPU")
    base.deallocate()
  assert {x.device for x in ledger.events if x.event == "alloc"} == {"CPU", "CPU:1"}
  assert ledger.complete


def test_bound_owner_survives_lazy_later_allocation():
  ledger, tensor = PhysicalMemoryLedger(), Tensor.empty(4, device="CPU")
  base = tensor.uop.buffer
  owner = AllocationOwner("weight", "model", semantic_owner_id="lazy")
  bind_allocation_owner(base, owner)
  with ledger.active(): base.allocate(); base.deallocate()
  assert ledger.events[0].owner == owner
  assert ledger.complete


def test_phase_owner_propagates_to_lazy_allocation_and_does_not_leak():
  ledger, base = PhysicalMemoryLedger(), _buf()
  with ledger.active():
    with allocation_owner(kind="gguf_tensor", lifetime="model"):
      base.allocate(); base.deallocate()
    later = _buf()
    later.allocate(); later.deallocate()
  assert ledger.events[0].owner == AllocationOwner("gguf_tensor", "model")
  assert ledger.events[2].owner is None
  assert not ledger.complete


def test_exact_uop_owner_bridges_recreated_lazy_arena_buffer():
  ledger = PhysicalMemoryLedger()
  arena = UOp.new_buffer("CPU", 19, dtypes.uint8)
  owner = AllocationOwner("schedule_arena", "schedule", semantic_owner_id="manifest:0:arena:CPU:compute")
  ledger.bind_uop_owner(arena, owner)
  # Reproduce a later materialization replacing the Buffer object which was
  # present at the synchronous manifest callback.
  later = _buf(nbytes=19)
  buffers[arena] = later
  ledger.record_manifest()
  with ledger.active(), allocation_phase("prefill_capture_dispatch"):
    later.allocate(); later.deallocate()
  event = ledger.events[0]
  assert event.owner == owner
  assert event.structure is not None
  assert event.structure.category == "manifest_arena_backing"
  assert event.structure.phase == "prefill_capture_dispatch" and event.structure.manifests_seen == 1
  assert not event.structure.buffer_owner_bound and event.structure.uop_owner_bound
  assert ledger.complete


def test_exact_uop_owner_survives_allocator_cache_reuse_with_new_buffer_base():
  ledger = PhysicalMemoryLedger()
  arena = UOp.new_buffer("CPU", 33, dtypes.uint8)
  owner = AllocationOwner("schedule_arena", "schedule", semantic_owner_id="manifest:1:arena:CPU:compute")
  ledger.bind_uop_owner(arena, owner)
  first = arena.buffer
  with Context(LRU=1), ledger.active(), allocation_phase("prefill_capture_dispatch"):
    first.allocate(); first.deallocate()
    second = _buf(nbytes=33)
    buffers[arena] = second
    second.allocate(); second.deallocate()
  assert [event.owner for event in ledger.events if event.event == "alloc"] == [owner, owner]
  allocs = [event for event in ledger.events if event.event == "alloc"]
  assert len({event.allocation_id for event in allocs}) == 2
  assert allocs[1].structure is not None and allocs[1].structure.allocator_storage_reused
  assert ledger.complete


def test_structural_diagnostics_do_not_upgrade_unowned_uop_root():
  ledger = PhysicalMemoryLedger()
  root = UOp.new_buffer("CPU", 11, dtypes.uint8)
  base = root.buffer
  with ledger.active(), allocation_phase("warmup_dispatch"):
    base.allocate(); base.deallocate()
  evidence = ledger.export_evidence(scanned_granularities={"CPU": 1}).to_json()
  assert evidence["lifetimes"][0]["owner"] is None
  assert evidence["lifetimes"][0]["structure"]["category"] == "runtime_temp_root"
  assert evidence["structural_summary"]["unowned_count_by_category"] == {"runtime_temp_root": 1}
  assert evidence["structural_summary"]["unowned_requested_bytes_by_category"] == {"runtime_temp_root": 11}
  assert evidence["structural_summary"]["unowned_count_by_phase"] == {"warmup_dispatch": 1}
  assert not evidence["complete"]


def test_uop_bridge_conflict_with_buffer_binding_fails_closed():
  ledger = PhysicalMemoryLedger()
  root = UOp.new_buffer("CPU", 13, dtypes.uint8)
  ledger.bind_uop_owner(root, AllocationOwner("schedule_arena", "schedule", semantic_owner_id="arena"))
  later = _buf(nbytes=13)
  buffers[root] = later
  bind_allocation_owner(later, AllocationOwner("runtime_persistent", "model"))
  with ledger.active(): later.allocate(); later.deallocate()
  assert ledger.events[0].owner is None
  assert "conflicting ownership" in ledger.issues[0]
  assert not ledger.complete


def test_view_inherits_base_binding_even_when_bound_through_view():
  ledger, base = PhysicalMemoryLedger(), _buf()
  view = base.view(4, dtypes.uint8, 2)
  owner = AllocationOwner("kv", "candidate", candidate_id="c0")
  bind_allocation_owner(view, owner)
  with ledger.active():
    view.allocate(); view.deallocate(); base.deallocate()
  assert [x.owner for x in ledger.events] == [owner, owner, owner, owner]
  assert ledger.complete


def test_bound_and_ambient_owners_must_agree():
  owner = AllocationOwner("weight", "model")
  agreed, a = PhysicalMemoryLedger(), _buf()
  bind_allocation_owner(a, owner)
  with agreed.active(), allocation_owner(kind="weight", lifetime="model"):
    a.allocate(); a.deallocate()
  assert agreed.complete and agreed.events[0].owner == owner

  conflict, b = PhysicalMemoryLedger(), _buf()
  bind_allocation_owner(b, owner)
  with conflict.active(), allocation_owner(kind="scratch", lifetime="step"):
    b.allocate(); b.deallocate()
  assert conflict.events[0].owner is None
  assert not conflict.complete and "conflicting ownership" in conflict.issues[0]


def test_rebinding_is_idempotent_only_for_identical_owner():
  base = _buf()
  owner = AllocationOwner("weight", "model")
  assert bind_allocation_owner(base, owner) == owner
  assert bind_allocation_owner(base.view(1, dtypes.uint8, 1), owner) == owner
  with pytest.raises(ValueError, match="already has different ownership"):
    bind_allocation_owner(base, AllocationOwner("scratch", "step"))


def test_binding_does_not_keep_inactive_buffer_alive():
  base = _buf()
  ref = weakref.ref(base)
  bind_allocation_owner(base, AllocationOwner("weight", "model"))
  del base
  gc.collect()
  assert ref() is None


def test_evidence_reuse_has_distinct_lifetimes_and_scanned_rounding():
  ledger, base = PhysicalMemoryLedger(), _buf(nbytes=33)
  with Context(LRU=1), ledger.active(), allocation_owner(kind="scratch", lifetime="step"):
    base.allocate(); base.deallocate()
    base.allocate(); base.deallocate()
  evidence = ledger.export_evidence(scanned_granularities={"CPU": 16})
  assert evidence.complete
  assert [(x.allocation_id, x.requested_nbytes, x.physical_nbytes) for x in evidence.lifetimes] == [(1, 33, 48), (2, 33, 48)]
  assert evidence.peak_physical_bytes == 48
  assert dict(evidence.peak_physical_bytes_per_device) == {"CPU": 48}
  assert json.loads(json.dumps(evidence.to_json())) == evidence.to_json()


def test_evidence_mapped_alias_is_explicit_and_not_double_counted(monkeypatch):
  ledger, base = PhysicalMemoryLedger(), _buf(nbytes=17)
  target = Device["CPU:1"].allocator
  monkeypatch.setattr(target, "_map", lambda opaque: opaque)
  monkeypatch.setattr(target, "_unmap", lambda opaque: None)
  with ledger.active(), allocation_owner(kind="weight", lifetime="model"):
    base.allocate(); base.get_buf("CPU:1"); base.deallocate()
  evidence = ledger.export_evidence(scanned_granularities={"CPU": 8})
  source, mapping = evidence.lifetimes
  assert mapping.mapped and mapping.physical_nbytes == 0
  assert mapping.physical_base_id == source.allocation_id
  assert mapping.allocation_id != source.allocation_id
  assert evidence.peak_physical_bytes == 24
  assert dict(evidence.peak_physical_bytes_per_device) == {"CPU": 24}


def test_evidence_per_device_and_global_peaks_follow_event_lifetimes():
  ledger, a, b, c = PhysicalMemoryLedger(), _buf("CPU", 9), _buf("CPU:1", 17), _buf("CPU", 25)
  with ledger.active(), allocation_owner(kind="scratch", lifetime="step"):
    a.allocate(); b.allocate(); a.deallocate()
    c.allocate(); b.deallocate(); c.deallocate()
  evidence = ledger.evidence(scanned_granularities={"CPU": 8, "CPU:1": 16})
  assert evidence.complete
  assert dict(evidence.peak_physical_bytes_per_device) == {"CPU": 32, "CPU:1": 32}
  assert evidence.peak_physical_bytes == 64


def test_evidence_accepts_explicit_granularity_resolver():
  ledger, base = PhysicalMemoryLedger(), _buf(nbytes=65)
  with ledger.active(), allocation_owner(kind="weight", lifetime="model"): base.allocate(); base.deallocate()
  evidence = ledger.export_evidence(granularity_resolver=lambda device: 64 if device == "CPU" else None)
  assert evidence.complete and evidence.lifetimes[0].physical_nbytes == 128


def test_evidence_unknown_granularity_fails_closed_precisely():
  ledger, base = PhysicalMemoryLedger(), _buf(nbytes=7)
  with ledger.active(), allocation_owner(kind="weight", lifetime="model"): base.allocate(); base.deallocate()
  evidence = ledger.export_evidence()
  assert not evidence.complete and evidence.peak_physical_bytes is None
  assert evidence.blockers == ("allocation 1 on CPU has no valid scanned physical-size granularity",)


def test_device_filter_records_only_selected_memory_domain():
  ledger, selected, unrelated = PhysicalMemoryLedger(("CPU",)), _buf("CPU", 9), _buf("CPU:1", 17)
  with ledger.active(), allocation_owner(kind="scratch", lifetime="step"):
    unrelated.allocate(); selected.allocate(); unrelated.deallocate(); selected.deallocate()
  evidence = ledger.export_evidence(scanned_granularities={"CPU": 8})
  assert evidence.complete and len(evidence.lifetimes) == 1
  assert evidence.lifetimes[0].device == "CPU" and evidence.peak_physical_bytes == 16
def test_typed_semantics_map_to_logical_physical_lifetimes():
  assert allocation_owner_from_semantic(MODEL_PARAMETER) == AllocationOwner("model_parameter", "model")
  assert allocation_owner_from_semantic(PREFILL_SCRATCH) == AllocationOwner("prefill_scratch", "prefill")
  assert allocation_owner_from_semantic(RUNTIME_OUTPUT) == AllocationOwner("runtime_output", "invocation")
  assert allocation_owner_from_semantic(candidate_workspace("candidate-a")) == \
    AllocationOwner("candidate_workspace", "candidate", candidate_id="candidate-a")
