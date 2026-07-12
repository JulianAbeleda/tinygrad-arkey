"""Typed plans and ownership proofs for staged kernel pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp

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


@dataclass(frozen=True)
class KernelStage1UOpGraph:
  plan: KernelStage1PipelinePlan
  k_tiles: int
  sink: UOp
  accumulator: UOp
  accumulator_reg: UOp
  accumulator_init: UOp
  body_range: UOp|None
  loop_end: UOp|None
  drain: UOp
  events: tuple[KernelStage1LifecycleEvent, ...]
  event_nodes: dict[KernelStage1LifecycleEvent, UOp]


@dataclass(frozen=True)
class KernelStage1UOpProof:
  passed: bool
  errors: tuple[str, ...]
  lifecycle: KernelStage1LifecycleProof


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


def build_stage1_uop_graph(plan:KernelStage1PipelinePlan, k_tiles:int,
                           produce:Callable[[str, int, int, UOp|None], UOp],
                           wmma:Callable[[int, int, UOp, UOp], UOp]) -> KernelStage1UOpGraph:
  """Build a host-only typed prologue/body/drain graph without modifying production TC lowering."""
  events = stage1_lifecycle_events(plan, k_tiles)
  nodes:dict[KernelStage1LifecycleEvent,UOp] = {}
  ready:list[UOp|None] = [None]*k_tiles
  release:list[UOp|None] = [None]*k_tiles
  accumulator_reg = UOp.placeholder((1,),dtypes.float,9200,addrspace=AddrSpace.REG)
  acc_index = UOp.const(dtypes.weakint,0)
  accumulator_init = accumulator_reg.index(acc_index).store(UOp.const(dtypes.float,0.0))
  body_range = UOp.range(k_tiles-1,9100,AxisType.REDUCE) if k_tiles > 1 else None
  previous_update:UOp = accumulator_init

  def emit_produce(epoch:int, phase:PipelinePhase, reuse:UOp|None) -> UOp:
    slot = plan.slot_for_epoch(epoch)
    role_nodes = []
    for role in plan.roles:
      node = produce(role, epoch, slot, reuse)
      if reuse is not None and reuse not in node.backward_slice: node = node.after(reuse)
      nodes[KernelStage1LifecycleEvent(phase,"produce",epoch,slot,role)] = node
      role_nodes.append(node)
    out = UOp.barrier(UOp.group(*role_nodes))
    nodes[KernelStage1LifecycleEvent(phase,"ready",epoch,slot)] = out
    ready[epoch] = out
    return out

  emit_produce(0, "prologue", None)
  for epoch in range(k_tiles-1):
    assert ready[epoch] is not None
    slot = plan.slot_for_epoch(epoch)
    acc_read = accumulator_reg.after(previous_update,*(() if body_range is None else (body_range,))).index(acc_index)
    compute = wmma(epoch, slot, ready[epoch], acc_read)
    if ready[epoch] not in compute.backward_slice: compute = compute.after(ready[epoch])
    update = accumulator_reg.index(acc_index).store(compute)
    for role in plan.roles: nodes[KernelStage1LifecycleEvent("body","consume",epoch,slot,role)] = compute
    if plan.buffer_count == 2:
      # Epoch e+2 reuses epoch e's slot, so e+1 production depends on release e-1 once reuse begins.
      next_ready = emit_produce(epoch+1, "body", release[epoch-1] if epoch else None)
      joined = UOp.barrier(UOp.group(update, next_ready))
    else:
      joined = UOp.barrier(update)
      emit_produce(epoch+1, "body", joined)
    release[epoch] = joined
    previous_update = update
    nodes[KernelStage1LifecycleEvent("body","release",epoch,slot)] = joined

  last = k_tiles-1
  assert ready[last] is not None
  loop_end = previous_update.end(body_range) if body_range is not None else None
  drain_acc = accumulator_reg.after(loop_end if loop_end is not None else accumulator_init).index(acc_index)
  drain = wmma(last, plan.slot_for_epoch(last), ready[last], drain_acc)
  if ready[last] not in drain.backward_slice: drain = drain.after(ready[last])
  for role in plan.roles: nodes[KernelStage1LifecycleEvent("drain","consume",last,plan.slot_for_epoch(last),role)] = drain
  drain_release = UOp.barrier(drain)
  nodes[KernelStage1LifecycleEvent("drain","release",last,plan.slot_for_epoch(last))] = drain_release
  release[last] = drain_release

  accumulator = drain
  sink = UOp.sink(accumulator, accumulator_init, *(x for x in release if x is not None), *((loop_end,) if loop_end is not None else ()))
  return KernelStage1UOpGraph(plan,k_tiles,sink,accumulator,accumulator_reg,accumulator_init,body_range,loop_end,drain,events,nodes)


def prove_stage1_uop_graph(graph:KernelStage1UOpGraph) -> KernelStage1UOpProof:
  lifecycle = prove_stage1_lifecycle(graph.plan,graph.k_tiles,graph.events)
  errors = list(lifecycle.errors)
  for event in graph.events:
    if event not in graph.event_nodes: errors.append(f"event has no emitted UOp: {event}")
  for event,node in graph.event_nodes.items():
    if event.op == "ready":
      for role in graph.plan.roles:
        producer = graph.event_nodes.get(KernelStage1LifecycleEvent(event.phase,"produce",event.epoch,event.slot,role))
        if producer is None or producer not in node.backward_slice: errors.append(f"ready lacks {role} producer for epoch {event.epoch}")
    elif event.op == "consume":
      ready_phase = "prologue" if event.epoch == 0 else "body"
      ready = graph.event_nodes.get(KernelStage1LifecycleEvent(ready_phase,"ready",event.epoch,event.slot))
      if ready is None or ready not in node.backward_slice: errors.append(f"consume lacks ready dependency for epoch {event.epoch}")
    elif event.op == "release":
      for role in graph.plan.roles:
        consume = graph.event_nodes.get(KernelStage1LifecycleEvent(event.phase,"consume",event.epoch,event.slot,role))
        if consume is None or consume not in node.backward_slice: errors.append(f"release lacks {role} consumer for epoch {event.epoch}")
  if graph.plan.buffer_count == 2:
    for epoch in range(graph.k_tiles-1):
      release = graph.event_nodes[KernelStage1LifecycleEvent("body","release",epoch,graph.plan.slot_for_epoch(epoch))]
      next_ready = graph.event_nodes[KernelStage1LifecycleEvent("body","ready",epoch+1,graph.plan.slot_for_epoch(epoch+1))]
      if next_ready not in release.backward_slice: errors.append(f"body epoch {epoch} does not join sibling next producer")
    for epoch in range(2,graph.k_tiles):
      prior_release = graph.event_nodes[KernelStage1LifecycleEvent("body","release",epoch-2,graph.plan.slot_for_epoch(epoch-2))]
      for role in graph.plan.roles:
        producer = graph.event_nodes[KernelStage1LifecycleEvent("body","produce",epoch,graph.plan.slot_for_epoch(epoch),role)]
        if prior_release not in producer.backward_slice: errors.append(f"epoch {epoch} {role} producer can overwrite unreleased slot")
  regs = [u for u in graph.sink.toposort() if u.op is Ops.DEFINE_REG]
  ends = [u for u in graph.sink.toposort() if u.op is Ops.END]
  if regs != [graph.accumulator_reg]: errors.append("graph does not use exactly one shared DEFINE_REG accumulator")
  for epoch in range(graph.k_tiles-1):
    compute = graph.event_nodes[KernelStage1LifecycleEvent("body","consume",epoch,graph.plan.slot_for_epoch(epoch),"A")]
    release_node = graph.event_nodes[KernelStage1LifecycleEvent("body","release",epoch,graph.plan.slot_for_epoch(epoch))]
    if graph.accumulator_reg not in compute.backward_slice: errors.append(f"body epoch {epoch} WMMA does not read shared accumulator")
    acc_stores = [u for u in release_node.backward_slice if u.op is Ops.STORE and graph.accumulator_reg in u.src[0].backward_slice]
    if not acc_stores: errors.append(f"body epoch {epoch} release does not join accumulator store")
  if graph.k_tiles == 1:
    if graph.body_range is not None or graph.loop_end is not None or ends: errors.append("single-tile graph must not emit body RANGE/END")
  else:
    if graph.body_range is None or graph.body_range.op is not Ops.RANGE or graph.body_range.arg[-1] is not AxisType.REDUCE:
      errors.append("body accumulator does not use one REDUCE RANGE")
    if graph.loop_end is None or graph.loop_end.op is not Ops.END or graph.body_range not in graph.loop_end.src:
      errors.append("body accumulator does not close with RANGE/END")
    if ends != [graph.loop_end]: errors.append("graph does not contain exactly one loop END")
    drain_acc_reads = [u for u in graph.drain.backward_slice if u.op is Ops.INDEX and graph.accumulator_reg in u.backward_slice]
    if not drain_acc_reads or not any(graph.loop_end in u.backward_slice for u in drain_acc_reads):
      errors.append("drain accumulator read is not ordered after loop END")
  if any(u.op in (Ops.REDUCE,Ops.ADD) for u in graph.sink.toposort()): errors.append("synthetic lifecycle uses forbidden REDUCE/final ADD")
  return KernelStage1UOpProof(not errors,tuple(errors),lifecycle)
