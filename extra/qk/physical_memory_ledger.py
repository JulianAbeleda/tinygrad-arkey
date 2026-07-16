"""Opt-in evidence ledger for physical Buffer allocation lifetimes.

Classification is deliberately accepted only through :func:`allocation_owner`
or :func:`bind_allocation_owner`.  The ledger never guesses ownership from
allocation size or the Python stack.
"""
from __future__ import annotations

import contextlib, contextvars
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping
from tinygrad.llm.physical_memory_ledger import (AllocationOwner, _bound_owners, _owners, allocation_owner,
  allocation_owner_from_semantic, bind_allocation_owner)


@dataclass(frozen=True)
class AllocationStructure:
  """Non-authoritative structural diagnostics captured at a physical allocation event."""
  phase:str|None
  manifests_seen:int
  category:str
  buffer_owner_bound:bool
  uop_owner_bound:bool
  uop_ops:tuple[str, ...]
  allocator_storage_reused:bool

  def to_json(self) -> dict[str, Any]:
    return {"phase": self.phase, "manifests_seen": self.manifests_seen, "category": self.category,
            "buffer_owner_bound": self.buffer_owner_bound, "uop_owner_bound": self.uop_owner_bound,
            "uop_ops": list(self.uop_ops), "allocator_storage_reused": self.allocator_storage_reused}


@dataclass(frozen=True)
class AllocationEvent:
  sequence:int
  event:str
  allocation_id:int
  device:str
  requested_nbytes:int
  physical_base_id:int
  owner:AllocationOwner|None
  mapped:bool = False
  structure:AllocationStructure|None = None

  @property
  def kind(self) -> str|None: return None if self.owner is None else self.owner.kind
  @property
  def lifetime(self) -> str|None: return None if self.owner is None else self.owner.lifetime
  @property
  def candidate_id(self) -> str|None: return None if self.owner is None else self.owner.candidate_id
  @property
  def semantic_owner_id(self) -> str|None: return None if self.owner is None else self.owner.semantic_owner_id


@dataclass(frozen=True)
class AllocationLifetime:
  allocation_id:int
  physical_base_id:int
  device:str
  alloc_sequence:int
  free_sequence:int|None
  requested_nbytes:int
  physical_nbytes:int|None
  owner:AllocationOwner|None
  mapped:bool
  structure:AllocationStructure|None = None

  def to_json(self) -> dict[str, Any]:
    return {"allocation_id": self.allocation_id, "physical_base_id": self.physical_base_id, "device": self.device,
            "alloc_sequence": self.alloc_sequence, "free_sequence": self.free_sequence,
            "requested_nbytes": self.requested_nbytes, "physical_nbytes": self.physical_nbytes,
            "mapped": self.mapped, "structure": None if self.structure is None else self.structure.to_json(),
            "owner": None if self.owner is None else {
              "kind": self.owner.kind, "lifetime": self.owner.lifetime, "candidate_id": self.owner.candidate_id,
              "semantic_owner_id": self.owner.semantic_owner_id}}


@dataclass(frozen=True)
class PhysicalMemoryEvidence:
  """Immutable reconciliation result; :meth:`to_json` returns a transport-shaped copy."""
  schema:str
  complete:bool
  blockers:tuple[str, ...]
  lifetimes:tuple[AllocationLifetime, ...]
  peak_physical_bytes:int|None
  peak_physical_bytes_per_device:tuple[tuple[str, int], ...]

  def to_json(self) -> dict[str, Any]:
    by_category:dict[str, int] = {}
    requested_by_category:dict[str, int] = {}
    unowned_by_category:dict[str, int] = {}
    unowned_requested_by_category:dict[str, int] = {}
    unowned_by_phase:dict[str, int] = {}
    for lifetime in self.lifetimes:
      category = "unclassified" if lifetime.structure is None else lifetime.structure.category
      by_category[category] = by_category.get(category, 0) + 1
      requested_by_category[category] = requested_by_category.get(category, 0) + lifetime.requested_nbytes
      if lifetime.owner is None:
        unowned_by_category[category] = unowned_by_category.get(category, 0) + 1
        unowned_requested_by_category[category] = unowned_requested_by_category.get(category, 0) + lifetime.requested_nbytes
        phase = "unmarked" if lifetime.structure is None or lifetime.structure.phase is None else lifetime.structure.phase
        unowned_by_phase[phase] = unowned_by_phase.get(phase, 0) + 1
    return {"schema": self.schema, "complete": self.complete, "blockers": list(self.blockers),
            "lifetimes": [x.to_json() for x in self.lifetimes], "peak_physical_bytes": self.peak_physical_bytes,
            "peak_physical_bytes_per_device": dict(self.peak_physical_bytes_per_device),
            "structural_summary": {"allocation_count_by_category": by_category,
              "requested_bytes_by_category": requested_by_category, "unowned_count_by_category": unowned_by_category,
              "unowned_requested_bytes_by_category": unowned_requested_by_category,
              "unowned_count_by_phase": unowned_by_phase}}


_phases:contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("physical_allocation_phases", default=())
_active_ledger:PhysicalMemoryLedger|None = None


@contextlib.contextmanager
def allocation_phase(phase:str) -> Iterator[str]:
  """Label allocation timing without supplying ownership evidence."""
  if not isinstance(phase, str) or not phase: raise ValueError("allocation phase must be a non-empty string")
  token = _phases.set(_phases.get() + (phase,))
  try: yield phase
  finally: _phases.reset(token)


class PhysicalMemoryLedger:
  def __init__(self, devices:Iterable[str]|None=None):
    self.events:list[AllocationEvent] = []
    self.issues:list[str] = []
    self.devices = None if devices is None else frozenset(str(x) for x in devices)
    self._next_id = 1
    self._identities:dict[tuple[int, str], int] = {}
    self._active_ids:set[int] = set()
    self._active = False
    self._uop_owners:dict[bytes, AllocationOwner] = {}
    self._manifests_seen = 0
    self._seen_allocator_storage:set[tuple[str, int]] = set()

  def _tracks(self, device:str) -> bool: return self.devices is None or device in self.devices

  @property
  def complete(self) -> bool: return not self.issues and not self._active_ids

  def identity(self, buffer, device:str|None=None) -> int|None:
    """Return a buffer/view's physical base identity on ``device``."""
    base = buffer.base
    return self._identities.get((id(base), device if device is not None else buffer.device))

  def bind_uop_owner(self, uop, owner:AllocationOwner) -> AllocationOwner:
    """Bind an owner to an exact structural UOp identity, including later Buffer materializations."""
    if not isinstance(owner, AllocationOwner): raise TypeError("owner must be an AllocationOwner")
    key = uop.key
    if not isinstance(key, bytes): raise TypeError("uop key must be bytes")
    if (existing := self._uop_owners.get(key)) is not None and existing != owner:
      raise ValueError("uop already has different ownership")
    self._uop_owners[key] = owner
    # Bind the current materialization too, when one exists. The UOp-key
    # contract remains authoritative if this Buffer is later recreated.
    try: bind_allocation_owner(uop.buffer, owner)
    except (AttributeError, AssertionError): pass
    return owner

  def record_manifest(self) -> int:
    """Advance the synchronous manifest boundary used by allocation diagnostics."""
    self._manifests_seen += 1
    return self._manifests_seen

  @contextlib.contextmanager
  def active(self) -> Iterator[PhysicalMemoryLedger]:
    global _active_ledger
    from tinygrad.device import Buffer
    if _active_ledger is not None: raise RuntimeError("a physical allocation ledger is already active")

    # Physical allocation evidence is research instrumentation, so install it only
    # for the measurement window instead of carrying observer state and branches in
    # Buffer's execution path. Events are emitted after the underlying operation
    # succeeds, matching the allocation lifetime visible to runtime callers.
    original_allocate, original_get_buf, original_deallocate = Buffer.allocate, Buffer.get_buf, Buffer.deallocate

    def allocate(buffer, opaque=None, external_ptr=None):
      ret = original_allocate(buffer, opaque, external_ptr)
      if buffer._base is not None:
        self._allocate_view(buffer, buffer.device)
      elif opaque is None and (buffer.options is None or buffer.options.external_ptr is None):
        self._allocate_base(buffer, buffer.device, buffer.nbytes)
      return ret

    def get_buf(buffer, device):
      already_mapped = device in buffer._bufs
      ret = original_get_buf(buffer, device)
      if not already_mapped and device != buffer.device:
        if buffer._base is not None: self._allocate_view(buffer, device)
        else: self._allocate_base(buffer.base, device, buffer.nbytes, mapped=True)
      return ret

    def deallocate(buffer):
      devices, is_view = tuple(buffer._bufs), buffer._base is not None
      ret = original_deallocate(buffer)
      if is_view:
        for device in devices: self._free_view(buffer, device)
      else:
        for device in devices: self._free_base(buffer, device)
      return ret

    Buffer.allocate, Buffer.get_buf, Buffer.deallocate = allocate, get_buf, deallocate
    _active_ledger, self._active = self, True
    try: yield self
    finally:
      self._active = False
      _active_ledger = None
      Buffer.allocate, Buffer.get_buf, Buffer.deallocate = original_allocate, original_get_buf, original_deallocate
      for allocation_id in sorted(self._active_ids): self._issue_once(f"allocation {allocation_id} has no free event")

  def _matching_uops(self, buffer) -> tuple[Any, ...]:
    """Find live UOps which structurally materialize this exact Buffer base."""
    try:
      from tinygrad.device import Buffer, MultiBuffer
      from tinygrad.uop.ops import buffers
      matches = []
      for uop, materialized in tuple(buffers.items()):
        candidates = materialized.bufs if isinstance(materialized, MultiBuffer) else (materialized,)
        if any(isinstance(candidate, Buffer) and candidate.base is buffer.base for candidate in candidates): matches.append(uop)
      return tuple(matches)
    except (ImportError, RuntimeError): return ()

  def _owner_and_structure(self, allocation_id:int, buffer, device:str) -> tuple[AllocationOwner|None, AllocationStructure]:
    uops = self._matching_uops(buffer)
    bound = _bound_owners.get(buffer.base)
    owners = _owners.get()
    if bound is not None: owners += (bound,)
    uop_owners = tuple(self._uop_owners[uop.key] for uop in uops if uop.key in self._uop_owners)
    owners += uop_owners
    ops = tuple(sorted({getattr(getattr(uop, "op", None), "name", type(uop).__name__) for uop in uops}))
    if uop_owners and any(owner.kind == "schedule_arena" for owner in uop_owners): category = "manifest_arena_backing"
    elif "PARAM" in ops: category = "call_argument"
    elif "SLICE" in ops: category = "arena_view_base"
    elif "BUFFER" in ops: category = "runtime_temp_root"
    else: category = "allocator_or_unregistered_buffer"
    storage_key = (device, id(buffer._bufs.get(device)))
    storage_reused = storage_key in self._seen_allocator_storage
    self._seen_allocator_storage.add(storage_key)
    structure = AllocationStructure(_phases.get()[-1] if _phases.get() else None, self._manifests_seen, category,
                                    bound is not None, bool(uop_owners), ops, storage_reused)
    if not owners:
      self._issue_once(f"allocation {allocation_id} has no explicit ownership")
      return None, structure
    if any(x != owners[0] for x in owners[1:]):
      self._issue_once(f"allocation {allocation_id} has conflicting ownership")
      return None, structure
    return owners[0], structure

  def _event(self, event:str, allocation_id:int, device:str, nbytes:int, owner:AllocationOwner|None,
             mapped:bool=False, physical_base_id:int|None=None, structure:AllocationStructure|None=None):
    self.events.append(AllocationEvent(len(self.events)+1, event, allocation_id, device, nbytes,
                                       allocation_id if physical_base_id is None else physical_base_id, owner, mapped, structure))

  def _allocate_base(self, buffer, device:str, nbytes:int, mapped:bool=False):
    if not self._tracks(device): return
    key = (id(buffer.base), device)
    if key in self._identities:
      self._issue_once(f"duplicate allocation event for active base on {device}")
      return
    allocation_id = self._next_id
    self._next_id += 1
    physical_base_id = allocation_id
    if mapped:
      physical_base_id = self.identity(buffer, buffer.device)
      if physical_base_id is None:
        self._issue_once(f"mapping allocation {allocation_id} on {device} has no source physical identity")
        physical_base_id = allocation_id
    self._identities[key] = allocation_id
    self._active_ids.add(allocation_id)
    owner, structure = self._owner_and_structure(allocation_id, buffer, device)
    self._event("alloc", allocation_id, device, nbytes, owner, mapped, physical_base_id, structure)

  def _free_base(self, buffer, device:str):
    if not self._tracks(device): return
    allocation_id = self._identities.pop((id(buffer.base), device), None)
    # The ledger is an opt-in measurement window. A Buffer allocated before
    # that window can legitimately become unreachable while it is active; its
    # free must not be mistaken for an incomplete in-window lifetime.
    if allocation_id is None: return
    if allocation_id not in self._active_ids:
      self._issue_once(f"allocation {allocation_id} has duplicate free event")
      return
    self._active_ids.remove(allocation_id)
    alloc = next(x for x in self.events if x.event == "alloc" and x.allocation_id == allocation_id)
    self._event("free", allocation_id, device, alloc.requested_nbytes, alloc.owner, alloc.mapped,
                alloc.physical_base_id, alloc.structure)

  def _allocate_view(self, buffer, device:str):
    if not self._tracks(device): return
    allocation_id = self.identity(buffer, device)
    if allocation_id is None:
      self._issue_once(f"view on {device} has no physical base identity")
      return
    alloc = next(x for x in self.events if x.event == "alloc" and x.allocation_id == allocation_id)
    self._event("view", allocation_id, device, 0, alloc.owner, alloc.mapped, alloc.physical_base_id, alloc.structure)

  def _free_view(self, buffer, device:str):
    if not self._tracks(device): return
    allocation_id = self.identity(buffer, device)
    if allocation_id is None: return
    alloc = next(x for x in self.events if x.event == "alloc" and x.allocation_id == allocation_id)
    self._event("view_free", allocation_id, device, 0, alloc.owner, alloc.mapped, alloc.physical_base_id, alloc.structure)

  def export_evidence(self, *, scanned_granularities:Mapping[str, int|None]|None=None,
                      granularity_resolver:Callable[[str], int|None]|None=None) -> PhysicalMemoryEvidence:
    """Reconcile events into exact lifetimes and peaks without assuming allocator alignment.

    ``scanned_granularities`` and ``granularity_resolver`` are explicit physical-size authorities. A non-mapped
    allocation whose device has no positive granularity makes the evidence incomplete. Mappings remain visible but
    contribute zero bytes because they alias their ``physical_base_id``.
    """
    if scanned_granularities is not None and granularity_resolver is not None:
      raise ValueError("supply scanned_granularities or granularity_resolver, not both")
    blockers = list(self.issues)
    by_id:dict[int, list[AllocationEvent]] = {}
    for event in self.events:
      if event.event in ("alloc", "free"): by_id.setdefault(event.allocation_id, []).append(event)
    lifetimes:list[AllocationLifetime] = []
    for allocation_id in sorted(by_id):
      events = by_id[allocation_id]
      allocs, frees = [x for x in events if x.event == "alloc"], [x for x in events if x.event == "free"]
      if len(allocs) != 1: blockers.append(f"allocation {allocation_id} has {len(allocs)} alloc events; expected exactly one")
      if len(frees) != 1: blockers.append(f"allocation {allocation_id} has {len(frees)} free events; expected exactly one")
      if not allocs: continue
      alloc, free = allocs[0], frees[0] if frees else None
      if free is not None and (free.sequence <= alloc.sequence or free.device != alloc.device or
                               free.physical_base_id != alloc.physical_base_id):
        blockers.append(f"allocation {allocation_id} has conflicting lifetime events")
      physical_nbytes:int|None = 0 if alloc.mapped else None
      if not alloc.mapped:
        resolution_failed = False
        try:
          granularity = (scanned_granularities or {}).get(alloc.device) if granularity_resolver is None else granularity_resolver(alloc.device)
        except Exception as exc:
          granularity, resolution_failed = None, True
          blockers.append(f"allocation {allocation_id} on {alloc.device} granularity resolver failed: {type(exc).__name__}: {exc}")
        if not resolution_failed and (not isinstance(granularity, int) or isinstance(granularity, bool) or granularity <= 0):
          blockers.append(f"allocation {allocation_id} on {alloc.device} has no valid scanned physical-size granularity")
        elif not resolution_failed: physical_nbytes = ((alloc.requested_nbytes + granularity - 1) // granularity) * granularity
      lifetimes.append(AllocationLifetime(allocation_id, alloc.physical_base_id, alloc.device, alloc.sequence,
                                           None if free is None else free.sequence, alloc.requested_nbytes,
                                           physical_nbytes, alloc.owner, alloc.mapped, alloc.structure))

    # Count storage at alloc and release at free. Mapping lifetimes are explicit zero-byte aliases.
    deltas:dict[int, list[tuple[str, int]]] = {}
    for lifetime in lifetimes:
      if lifetime.physical_nbytes is None or lifetime.mapped or lifetime.free_sequence is None: continue
      deltas.setdefault(lifetime.alloc_sequence, []).append((lifetime.device, lifetime.physical_nbytes))
      deltas.setdefault(lifetime.free_sequence, []).append((lifetime.device, -lifetime.physical_nbytes))
    current:dict[str, int] = {}
    peaks:dict[str, int] = {}
    global_peak = 0
    for sequence in sorted(deltas):
      for device, delta in deltas[sequence]:
        current[device] = current.get(device, 0) + delta
        peaks[device] = max(peaks.get(device, 0), current[device])
      global_peak = max(global_peak, sum(current.values()))
    blockers = list(dict.fromkeys(blockers))
    exact = not blockers
    return PhysicalMemoryEvidence("tinygrad.physical_memory_ledger.v1", exact, tuple(blockers), tuple(lifetimes),
                                  global_peak if exact else None, tuple(sorted(peaks.items())))

  evidence = export_evidence

  def _issue_once(self, issue:str):
    if issue not in self.issues: self.issues.append(issue)


__all__ = ["AllocationEvent", "AllocationLifetime", "AllocationOwner", "AllocationStructure", "PhysicalMemoryEvidence",
           "PhysicalMemoryLedger", "allocation_owner", "allocation_owner_from_semantic", "allocation_phase", "bind_allocation_owner"]
