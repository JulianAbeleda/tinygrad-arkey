"""Typed plans and ownership proofs for staged kernel pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
from extra.qk.compiler_policies import StoragePolicy

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp

PipelinePhase = Literal["prologue", "body", "drain"]
PipelineOp = Literal["produce", "ready", "consume", "release"]

def storage_policy_from_stage1(plan: "KernelStage1PipelinePlan") -> StoragePolicy:
  return StoragePolicy("lds", plan.buffer_count, plan.slot_bytes, plan.roles)


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
class KernelStage1ProducerStage:
  epoch: UOp
  slot: UOp
  role_nodes: tuple[UOp, UOp]
  ready: UOp


@dataclass(frozen=True)
class KernelStage1FragmentStage:
  epoch: UOp
  slot: UOp
  ready: UOp
  fragments: tuple[UOp, ...]


@dataclass(frozen=True)
class KernelStage1UOpGraph:
  plan: KernelStage1PipelinePlan
  k_tiles: int
  sink: UOp
  accumulator: tuple[UOp, ...]
  accumulator_reg: UOp|None
  accumulator_init: UOp|None
  body_range: UOp|None
  loop_end: UOp|None
  body_join: UOp|None
  prologue: KernelStage1ProducerStage
  body_producer: KernelStage1ProducerStage|None
  body_fragments: KernelStage1FragmentStage|None
  drain_fragments: KernelStage1FragmentStage
  drain: tuple[UOp, ...]
  subtile_count: int
  events: tuple[KernelStage1LifecycleEvent, ...]


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
                           produce:Callable[[UOp,UOp,UOp|None], KernelStage1ProducerStage],
                           fragments:Callable[[UOp,UOp,UOp], KernelStage1FragmentStage],
                           wmma:Callable[[KernelStage1FragmentStage,UOp,int], UOp], *, subtile_count:int=8,
                           accumulator_elements:int|None=None, accumulator_offset:UOp|None=None,
                           accumulator_contract:tuple[UOp,tuple[tuple[int,int],...]]|None=None,
                           body_range_id:int=9100, accumulator_id:int=9200) -> KernelStage1UOpGraph:
  if subtile_count <= 0: raise ValueError("subtile_count must be positive")
  events = stage1_lifecycle_events(plan,k_tiles)
  zero,last=UOp.const(dtypes.weakint,0),UOp.const(dtypes.weakint,k_tiles-1)
  prologue=produce(zero,zero,None)
  if k_tiles == 1:
    frag=fragments(last,zero,prologue.ready)
    drain=tuple(wmma(frag,UOp.const(dtypes.float.vec(8),0.0),i) for i in range(subtile_count))
    return KernelStage1UOpGraph(plan,k_tiles,UOp.sink(*drain,prologue.ready),drain,None,None,None,None,None,
      prologue,None,None,frag,drain,subtile_count,events)
  rng=UOp.range(k_tiles-1,body_range_id,AxisType.REDUCE); slot=rng%plan.buffer_count
  body_frag=fragments(rng,slot,prologue.ready)
  acc_elements=subtile_count*8 if accumulator_elements is None else accumulator_elements
  if acc_elements < subtile_count*8 or acc_elements % 8: raise ValueError("accumulator_elements must contain whole vec8 slices")
  reg=UOp.placeholder((acc_elements,),dtypes.float,accumulator_id,addrspace=AddrSpace.REG)
  init=reg.index(UOp.const(dtypes.weakint,0),dtype=dtypes.float.vec(acc_elements)).store(UOp.const(dtypes.float.vec(acc_elements),0.0))
  updates=[]
  for i in range(subtile_count):
    off=UOp.const(dtypes.weakint,i*8) if accumulator_offset is None else accumulator_offset+i*8
    if accumulator_contract is None:
      acc=reg.after(init,rng).index(off,dtype=dtypes.float.vec(8)); value=wmma(body_frag,acc,i)
      updates.append(reg.index(off,dtype=dtypes.float.vec(8)).store(value))
    else:
      elem,arg=accumulator_contract
      acc=UOp(Ops.CONTRACT,dtypes.float.vec(8),(reg.after(init,rng).index(off+elem).load(),),arg)
      value=UOp(Ops.UNROLL,dtypes.float,(wmma(body_frag,acc,i),),arg)
      updates.append(reg.index(off+elem).store(value))
  body_prod=produce(rng+1,(rng+1)%plan.buffer_count,None)
  join=UOp.barrier(UOp.group(*updates,*body_prod.role_nodes)).replace(tag=("pipeline_body_join",rng,body_prod.epoch,body_prod.slot))
  body_prod=KernelStage1ProducerStage(body_prod.epoch,body_prod.slot,body_prod.role_nodes,join)
  end=join.end(rng).replace(tag=("pipeline_body_end",rng))
  drain_frag=fragments(last,UOp.const(dtypes.weakint,plan.slot_for_epoch(k_tiles-1)),end)
  drain=[]
  for i in range(subtile_count):
    off=UOp.const(dtypes.weakint,i*8) if accumulator_offset is None else accumulator_offset+i*8
    acc=(reg.after(end).index(off,dtype=dtypes.float.vec(8)) if accumulator_contract is None else
         UOp(Ops.CONTRACT,dtypes.float.vec(8),(reg.after(end).index(off+accumulator_contract[0]).load(),),accumulator_contract[1]))
    drain.append(wmma(drain_frag,acc,i))
  drain=tuple(drain)
  return KernelStage1UOpGraph(plan,k_tiles,UOp.sink(*drain,end,prologue.ready),drain,reg,init,rng,end,join,
    prologue,body_prod,body_frag,drain_frag,drain,subtile_count,events)

def prove_stage1_uop_graph(graph:KernelStage1UOpGraph) -> KernelStage1UOpProof:
  lifecycle=prove_stage1_lifecycle(graph.plan,graph.k_tiles,graph.events); errors=list(lifecycle.errors)
  topo=graph.sink.toposort(); regs=[u for u in topo if u.op is Ops.DEFINE_REG]; ends=[u for u in topo if u.op is Ops.END]
  if graph.k_tiles == 1:
    if regs or ends or graph.body_range is not None: errors.append("single tile must not emit REG/RANGE/END")
  else:
    if graph.body_fragments.epoch is not graph.body_range or graph.body_fragments.slot.render() != (graph.body_range%graph.plan.buffer_count).render():
      errors.append("body fragment callback changed symbolic epoch/slot formula")
    if graph.body_producer.epoch.render() != (graph.body_range+1).render() or \
       graph.body_producer.slot.render() != ((graph.body_range+1)%graph.plan.buffer_count).render():
      errors.append("next producer callback changed symbolic epoch/slot formula")
    if regs != [graph.accumulator_reg] or graph.accumulator_reg.ptrdtype.size < graph.subtile_count*8 or graph.accumulator_reg.ptrdtype.size%8:
      errors.append("bad shared accumulator layout")
    if graph.body_range is None or graph.body_range.arg[-1] is not AxisType.REDUCE: errors.append("missing symbolic body range")
    if ends != [graph.loop_end] or graph.loop_end.src[0] is not graph.body_join: errors.append("END is not rooted at body join")
    stores=[u for u in graph.body_join.backward_slice if u.op is Ops.STORE and graph.accumulator_reg in u.src[0].backward_slice and
            (u.src[1].dtype == dtypes.float.vec(8) or u.src[1].op is Ops.UNROLL)]
    if len(stores) != graph.subtile_count: errors.append("body lacks one accumulator update per symbolic subtile")
    if any(node not in graph.body_join.backward_slice for node in graph.body_producer.role_nodes): errors.append("body join lacks sibling producer")
    for out in graph.drain:
      direct=any(u.op is Ops.INDEX and u.dtype == dtypes.float.vec(8) and graph.loop_end in u.backward_slice for u in out.backward_slice)
      contracted=any(u.op is Ops.CONTRACT and u.dtype == dtypes.float.vec(8) and graph.loop_end in u.backward_slice for u in out.backward_slice)
      if not (direct or contracted):
        errors.append("drain lacks vec8 accumulator read after END")
  if any(u.op is Ops.REDUCE for u in topo): errors.append("forbidden Ops.REDUCE")
  return KernelStage1UOpProof(not errors,tuple(errors),lifecycle)
