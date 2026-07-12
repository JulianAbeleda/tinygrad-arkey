"""Typed plans and ownership proofs for staged kernel pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PipelinePhase = Literal["prologue", "body", "drain"]
PipelineOp = Literal["produce", "ready", "consume", "release"]


@dataclass(frozen=True)
class KernelStage1PipelinePlan:
  """Memory and ownership contract for a one-stage A/B pipeline."""
  buffer_count: int
  slot_bytes: int
  stage_count: int = 1
  roles: tuple[str, ...] = ("A", "B")

  def __post_init__(self) -> None:
    if not isinstance(self.buffer_count, int) or isinstance(self.buffer_count, bool) or self.buffer_count not in (1, 2):
      raise ValueError("stage-1 pipeline buffer_count must be 1 or 2")
    if self.stage_count != 1: raise ValueError("only stage_count=1 is currently proved")
    if self.roles != ("A", "B"): raise ValueError("stage-1 pipeline roles must be exactly ('A', 'B')")
    if not isinstance(self.slot_bytes, int) or isinstance(self.slot_bytes, bool) or self.slot_bytes <= 0:
      raise ValueError("slot_bytes must be a positive int")

  @property
  def active_lds_bytes(self) -> int: return self.buffer_count * self.slot_bytes

  def slot_for_epoch(self, epoch:int) -> int:
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0: raise ValueError("epoch must be a non-negative int")
    return epoch % self.buffer_count

  def slot_window(self, slot:int) -> tuple[int, int]:
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < self.buffer_count:
      raise ValueError(f"slot must be in [0, {self.buffer_count})")
    return slot*self.slot_bytes, (slot+1)*self.slot_bytes


@dataclass(frozen=True)
class KernelStage1LifecycleEvent:
  phase: PipelinePhase
  op: PipelineOp
  epoch: int
  slot: int
  role: str | None = None


@dataclass(frozen=True)
class KernelStage1LifecycleProof:
  passed: bool
  errors: tuple[str, ...]
  produced: tuple[tuple[str, int, int], ...]
  consumed: tuple[tuple[str, int, int], ...]
  released_slots: tuple[tuple[int, int], ...]


def stage1_lifecycle_events(plan:KernelStage1PipelinePlan, k_tiles:int) -> tuple[KernelStage1LifecycleEvent, ...]:
  """Build the canonical lifecycle. Buffer-2 fills the alternate slot before consuming the current one."""
  if not isinstance(k_tiles, int) or isinstance(k_tiles, bool) or k_tiles <= 0: raise ValueError("k_tiles must be a positive int")
  events:list[KernelStage1LifecycleEvent] = []

  def produce(epoch:int, phase:PipelinePhase) -> None:
    slot = plan.slot_for_epoch(epoch)
    events.extend(KernelStage1LifecycleEvent(phase, "produce", epoch, slot, role) for role in plan.roles)
    events.append(KernelStage1LifecycleEvent(phase, "ready", epoch, slot))

  def consume(epoch:int, phase:PipelinePhase) -> None:
    slot = plan.slot_for_epoch(epoch)
    events.extend(KernelStage1LifecycleEvent(phase, "consume", epoch, slot, role) for role in plan.roles)
    events.append(KernelStage1LifecycleEvent(phase, "release", epoch, slot))

  produce(0, "prologue")
  for epoch in range(k_tiles-1):
    if plan.buffer_count == 2: produce(epoch+1, "body")
    consume(epoch, "body")
    if plan.buffer_count == 1: produce(epoch+1, "body")
  consume(k_tiles-1, "drain")
  return tuple(events)


def prove_stage1_lifecycle(plan:KernelStage1PipelinePlan, k_tiles:int,
                           events:tuple[KernelStage1LifecycleEvent, ...]) -> KernelStage1LifecycleProof:
  """Independently prove slot ownership, readiness, exact consumption, and complete drain."""
  if not isinstance(k_tiles, int) or isinstance(k_tiles, bool) or k_tiles <= 0: raise ValueError("k_tiles must be a positive int")
  errors:list[str] = []
  live:dict[int, dict[str, int]] = {}
  ready:set[tuple[int, int]] = set()
  produced:set[tuple[str, int, int]] = set()
  consumed:set[tuple[str, int, int]] = set()
  released:list[tuple[int, int]] = []

  for index,event in enumerate(events):
    where = f"event {index}"
    if event.phase not in ("prologue", "body", "drain") or event.op not in ("produce", "ready", "consume", "release"):
      errors.append(f"{where}: invalid phase or operation")
      continue
    if not isinstance(event.epoch, int) or isinstance(event.epoch, bool) or not 0 <= event.epoch < k_tiles:
      errors.append(f"{where}: epoch {event.epoch!r} is out of range")
      continue
    expected_slot = plan.slot_for_epoch(event.epoch)
    if event.slot != expected_slot:
      errors.append(f"{where}: epoch {event.epoch} must use slot {expected_slot}, got {event.slot}")
      continue
    key = (event.epoch, event.slot)
    slot_live = live.setdefault(event.slot, {})
    if event.op in ("produce", "consume") and event.role not in plan.roles:
      errors.append(f"{where}: {event.op} requires role A or B")
      continue
    if event.op in ("ready", "release") and event.role is not None:
      errors.append(f"{where}: {event.op} must not name a role")
      continue
    if event.op == "produce":
      assert event.role is not None
      owner = slot_live.get(event.role)
      if owner is not None: errors.append(f"{where}: overwrite hazard: slot {event.slot} role {event.role} still owns epoch {owner}")
      item = (event.role, event.epoch, event.slot)
      if item in produced: errors.append(f"{where}: duplicate producer for {item}")
      else: produced.add(item)
      if owner is None: slot_live[event.role] = event.epoch
    elif event.op == "ready":
      missing = tuple(role for role in plan.roles if slot_live.get(role) != event.epoch)
      if missing: errors.append(f"{where}: epoch {event.epoch} ready before roles {missing} were produced")
      elif key in ready: errors.append(f"{where}: duplicate ready for epoch {event.epoch} slot {event.slot}")
      else: ready.add(key)
    elif event.op == "consume":
      assert event.role is not None
      item = (event.role, event.epoch, event.slot)
      if key not in ready: errors.append(f"{where}: consume before epoch {event.epoch} is ready")
      if slot_live.get(event.role) != event.epoch:
        errors.append(f"{where}: slot {event.slot} role {event.role} does not own epoch {event.epoch}")
      if item in consumed: errors.append(f"{where}: duplicate consumer for {item}")
      else: consumed.add(item)
    else:
      missing = tuple(role for role in plan.roles if (role, event.epoch, event.slot) not in consumed)
      if missing: errors.append(f"{where}: release before roles {missing} consumed epoch {event.epoch}")
      else:
        ready.discard(key)
        live.pop(event.slot, None)
        released.append(key)

  expected = {(role, epoch, plan.slot_for_epoch(epoch)) for epoch in range(k_tiles) for role in plan.roles}
  for item in sorted(expected-produced): errors.append(f"missing producer for {item}")
  for item in sorted(expected-consumed): errors.append(f"missing consumer for {item}")
  if live: errors.append(f"live slots remain after drain: {tuple(sorted((slot, tuple(sorted(owners.items()))) for slot,owners in live.items()))}")
  if ready: errors.append(f"ready epochs remain after drain: {tuple(sorted(ready))}")
  return KernelStage1LifecycleProof(not errors, tuple(errors), tuple(sorted(produced)), tuple(sorted(consumed)), tuple(released))
