"""Typed plans and executable graph construction for staged kernel pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol
from tinygrad.codegen.opt.compiler_policies import PipelinePolicy, StoragePolicy

from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp

# The Q4 scheduler owns the output-tile loop, but WMMA fragments are a physical
# resource.  Keep this limit here (at the compiler boundary) rather than in a
# route selector or renderer so all scheduler-owned consumers get the same
# fail-closed admission rule.
PINNED_WMMA_VGPR_BUDGET = 192

def validate_scheduler_tile_loop_pressure(*, resident_accumulator_vgprs:int,
                                          resident_fragment_vgprs:int,
                                          transient_vgpr_reserve:int=0,
                                          pinned_vgpr_budget:int=PINNED_WMMA_VGPR_BUDGET) -> None:
  """Admit a fused scheduler tile loop without widening WMMA residency.

  A tile loop is legal when its accumulator and A/B fragment carriers fit in
  the pinned budget.  The loop trip count is intentionally absent: iterations
  must reuse these carriers, so adding output tiles cannot increase pressure.
  This is a compiler contract, not an AMD-specific lowering rule.
  """
  values = (resident_accumulator_vgprs, resident_fragment_vgprs, transient_vgpr_reserve, pinned_vgpr_budget)
  if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in values):
    raise ValueError("WMMA VGPR pressure values must be non-negative ints")
  if pinned_vgpr_budget != PINNED_WMMA_VGPR_BUDGET:
    raise ValueError(f"stage-1 WMMA VGPR budget is pinned at {PINNED_WMMA_VGPR_BUDGET}")
  required = resident_accumulator_vgprs + resident_fragment_vgprs + transient_vgpr_reserve
  # A schedule carrying transient work must stay strictly below the abstract
  # ceiling: equality leaves no allocator flexibility at the overlap point.
  # The zero-reserve form retains the original pinned-carrier contract.
  if required > pinned_vgpr_budget or (transient_vgpr_reserve and required == pinned_vgpr_budget):
    raise ValueError(f"scheduler tile loop requires {required} pinned VGPRs, budget is {pinned_vgpr_budget}")

class StorageCallbacks(Protocol):
  def producer(self, epoch: UOp, slot: UOp, reuse: UOp|None = None): ...
  def fragments(self, epoch: UOp, slot: UOp, ready: UOp|None = None): ...

@dataclass(frozen=True)
class Stage1StorageAdapter:
  callbacks: StorageCallbacks
  policy: StoragePolicy

  def producer(self, epoch: UOp, slot: UOp): return self.callbacks.producer(epoch, slot)
  def fragments(self, epoch: UOp, slot: UOp): return self.callbacks.fragments(epoch, slot)

  # Stage-1 lowering callbacks carry the lifecycle dependency in addition to
  # the symbolic epoch and slot.  Keep the two-argument accessors above for
  # callers that inspect raw storage callbacks, while exposing typed methods
  # for the graph builder.
  def producer_stage(self, epoch: UOp, slot: UOp, reuse: UOp|None) -> "KernelStage1ProducerStage":
    stage = self.callbacks.producer(epoch, slot, reuse)
    if not isinstance(stage, KernelStage1ProducerStage):
      raise TypeError("stage-1 producer callback must return KernelStage1ProducerStage")
    return stage

  def fragment_stage(self, epoch: UOp, slot: UOp, ready: UOp) -> "KernelStage1FragmentStage":
    stage = self.callbacks.fragments(epoch, slot, ready)
    if not isinstance(stage, KernelStage1FragmentStage):
      raise TypeError("stage-1 fragment callback must return KernelStage1FragmentStage")
    return stage

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


def pipeline_policy_from_candidate(pipeline: object) -> PipelinePolicy:
  """Resolve an admitted candidate's existing typed policy at one boundary.

  Legacy stage-1 admissions carry ``KernelStage1PipelinePlan`` directly; new
  policy-aware candidates expose ``pipeline_policy``. Register-resident
  candidates are recognized here but remain unsupported by the stage-1 LDS
  adapter, allowing callers to fail closed before allocating local storage.
  """
  if isinstance(pipeline, KernelStage1PipelinePlan):
    storage = storage_policy_from_stage1(pipeline)
    return PipelinePolicy.lds(buffer_count=storage.buffer_count, slot_bytes=storage.slot_bytes, stages=pipeline.stage_count)
  policy = getattr(pipeline, "pipeline_policy", None)
  if not isinstance(policy, PipelinePolicy):
    raise ValueError("candidate pipeline does not expose a typed PipelinePolicy")
  return policy


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
  body_readiness: Literal["legacy", "matching", "sequential"] = "legacy"
  accumulator_dtype: DType = dtypes.float


def build_stage1_uop_graph(plan:KernelStage1PipelinePlan, k_tiles:int,
                           produce:Callable[[UOp,UOp,UOp|None], KernelStage1ProducerStage],
                           fragments:Callable[[UOp,UOp,UOp], KernelStage1FragmentStage],
                           wmma:Callable[[KernelStage1FragmentStage,UOp,int], UOp], *, subtile_count:int=8,
                           accumulator_elements:int|None=None, accumulator_offset:UOp|None=None,
                           accumulator_contract:tuple[UOp,tuple[tuple[int,int],...]]|None=None,
                           body_range_id:int=9100, accumulator_id:int=9200,
                           body_readiness:Literal["legacy", "matching", "sequential"]="legacy",
                           accumulator_dtype:DType=dtypes.float) -> KernelStage1UOpGraph:
  if subtile_count <= 0: raise ValueError("subtile_count must be positive")
  if not isinstance(accumulator_dtype, DType) or accumulator_dtype.count != 1:
    raise ValueError("accumulator_dtype must be a scalar dtype")
  accumulator_vec_dtype=accumulator_dtype.vec(8)

  def accumulate(stage:KernelStage1FragmentStage, accumulator:UOp, subtile:int) -> UOp:
    value=wmma(stage,accumulator,subtile)
    if value.dtype != accumulator_vec_dtype:
      raise ValueError(f"mixed accumulator dtypes: expected {accumulator_vec_dtype}, got {value.dtype}")
    return value

  zero,last=UOp.const(dtypes.weakint,0),UOp.const(dtypes.weakint,k_tiles-1)
  prologue=produce(zero,zero,None)
  if k_tiles == 1:
    frag=fragments(last,zero,prologue.ready)
    drain=tuple(accumulate(frag,UOp.const(accumulator_vec_dtype,0),i) for i in range(subtile_count))
    return KernelStage1UOpGraph(plan,k_tiles,UOp.sink(*drain),drain,None,None,None,None,None,
      prologue,None,None,frag,drain,subtile_count,body_readiness,accumulator_dtype)
  rng=UOp.range(k_tiles-1,body_range_id,AxisType.REDUCE)
  # gfx1100 has no indirect VGPR addressing. A sequential register schedule
  # therefore carries a compile-time slot zero through the whole K loop.
  slot=UOp.const(dtypes.weakint, 0) if body_readiness == "sequential" else rng%plan.buffer_count
  if body_readiness == "legacy":
    body_frag=fragments(rng,slot,prologue.ready)
  elif body_readiness == "matching":
    # The body consumes the current slot while prefetching the next slot.  The
    # first current-slot value comes from the prologue, so keep that producer
    # in the readiness dependency as well; subsequent iterations are ordered by
    # the loop END rooted at the prior body join.
    pending_body_prod=produce(rng+1,(rng+1)%plan.buffer_count,prologue.ready)
    body_frag=fragments(rng,slot,pending_body_prod.ready)
  elif body_readiness == "sequential":
    # Consume the current tile before constructing the next producer. The
    # producer receives accumulator updates as its overwrite dependency below.
    body_frag=fragments(rng,slot,prologue.ready)
  else:
    raise ValueError(f"unsupported body readiness mode {body_readiness!r}")
  acc_elements=subtile_count*8 if accumulator_elements is None else accumulator_elements
  if acc_elements < subtile_count*8 or acc_elements % 8: raise ValueError("accumulator_elements must contain whole vec8 slices")
  reg=UOp.placeholder((acc_elements,),accumulator_dtype,accumulator_id,addrspace=AddrSpace.REG)
  init=reg.index(UOp.const(dtypes.weakint,0),dtype=accumulator_dtype.vec(acc_elements)).store(UOp.const(accumulator_dtype.vec(acc_elements),0))
  updates=[]
  for i in range(subtile_count):
    off=UOp.const(dtypes.weakint,i*8) if accumulator_offset is None else accumulator_offset+i*8
    if accumulator_contract is None:
      acc=reg.after(init,rng).index(off,dtype=accumulator_vec_dtype); value=accumulate(body_frag,acc,i)
      updates.append(reg.index(off,dtype=accumulator_vec_dtype).store(value))
    else:
      elem,arg=accumulator_contract
      acc=UOp(Ops.CONTRACT,accumulator_vec_dtype,(reg.after(init,rng).index(off+elem).load(),),arg)
      value=UOp(Ops.UNROLL,accumulator_dtype,(accumulate(body_frag,acc,i),),arg)
      updates.append(reg.index(off+elem).store(value))
  if body_readiness == "matching":
    body_prod=pending_body_prod
    join=UOp.barrier(UOp.group(*updates,*body_prod.role_nodes)).replace(tag=("pipeline_body_join",rng,body_prod.epoch,body_prod.slot))
    body_prod=KernelStage1ProducerStage(body_prod.epoch,body_prod.slot,body_prod.role_nodes,join)
  elif body_readiness == "sequential":
    # One physical stage: the next producer must be ordered after all current
    # accumulator stores before it can overwrite the same VGPR fragments.
    body_prod=produce(rng+1,slot,UOp.group(*updates))
    # Register producers may expose a BARRIER readiness node so a typed
    # targeted wait can sit beside their stores.  Keep the generic GROUP
    # verifier happy by joining the role stores and readiness at the barrier
    # seam instead of placing a BARRIER inside a GROUP.
    join=UOp.barrier(UOp.group(*updates,*body_prod.role_nodes), body_prod.ready).replace(
      tag=("pipeline_body_join",rng,body_prod.epoch,body_prod.slot))
  else:
    body_prod=produce(rng+1,(rng+1)%plan.buffer_count,None)
    join=UOp.barrier(UOp.group(*updates,*body_prod.role_nodes)).replace(tag=("pipeline_body_join",rng,body_prod.epoch,body_prod.slot))
    body_prod=KernelStage1ProducerStage(body_prod.epoch,body_prod.slot,body_prod.role_nodes,join)
  end=join.end(rng).replace(tag=("pipeline_body_end",rng))
  drain_frag=fragments(last,UOp.const(dtypes.weakint,plan.slot_for_epoch(k_tiles-1)),end)
  drain=[]
  for i in range(subtile_count):
    off=UOp.const(dtypes.weakint,i*8) if accumulator_offset is None else accumulator_offset+i*8
    acc=(reg.after(end).index(off,dtype=accumulator_vec_dtype) if accumulator_contract is None else
         UOp(Ops.CONTRACT,accumulator_vec_dtype,(reg.after(end).index(off+accumulator_contract[0]).load(),),accumulator_contract[1]))
    drain.append(accumulate(drain_frag,acc,i))
  drain=tuple(drain)
  return KernelStage1UOpGraph(plan,k_tiles,UOp.sink(*drain,end),drain,reg,init,rng,end,join,
    prologue,body_prod,body_frag,drain_frag,drain,subtile_count,body_readiness,accumulator_dtype)


def build_stage1_uop_graph_with_storage(adapter: Stage1StorageAdapter, plan: KernelStage1PipelinePlan, k_tiles: int,
                                        wmma: Callable[[KernelStage1FragmentStage,UOp,int], UOp], *, subtile_count: int = 8,
                                        accumulator_elements: int|None = None, accumulator_offset: UOp|None = None,
                                        accumulator_contract: tuple[UOp,tuple[tuple[int,int],...]]|None = None,
                                        body_range_id: int = 9100, accumulator_id: int = 9200,
                                        body_readiness: Literal["legacy", "matching", "sequential"] | None = None,
                                        accumulator_dtype: DType = dtypes.float) -> KernelStage1UOpGraph:
  """Build a stage-1 graph through a typed storage adapter.

  This is deliberately a wrapper around ``build_stage1_uop_graph``: existing
  callers retain the legacy callback entrypoint and therefore identical UOp
  output, while policy-aware lowering can swap storage implementations at one
  boundary.  The current stage-1 graph is an LDS plan, so fail closed if an
  adapter advertises a different storage contract.
  """
  if not isinstance(adapter, Stage1StorageAdapter): raise TypeError("expected Stage1StorageAdapter")
  mode = body_readiness or getattr(adapter.callbacks, "body_readiness", "legacy")
  if adapter.policy.kind == "global_register_resident":
    if mode not in ("matching", "sequential"): raise ValueError("register storage requires matching or sequential body readiness")
    if getattr(plan, "active_lds_bytes", None) != 0: raise ValueError("register storage plan must declare zero LDS")
  else:
    expected = storage_policy_from_stage1(plan)
    if adapter.policy != expected:
      raise ValueError(f"storage policy does not match stage-1 plan: expected {expected!r}, got {adapter.policy!r}")
  return build_stage1_uop_graph(plan, k_tiles, adapter.producer_stage, adapter.fragment_stage, wmma,
    subtile_count=subtile_count, accumulator_elements=accumulator_elements, accumulator_offset=accumulator_offset,
    accumulator_contract=accumulator_contract, body_range_id=body_range_id, accumulator_id=accumulator_id,
    body_readiness=mode, accumulator_dtype=accumulator_dtype)

def validate_stage1_uop_graph(graph:KernelStage1UOpGraph) -> tuple[str, ...]:
  """Return structural errors that must reject a stage-1 production candidate."""
  errors:list[str] = []
  if not isinstance(graph.accumulator_dtype, DType) or graph.accumulator_dtype.count != 1:
    errors.append("accumulator dtype metadata must be scalar")
    accumulator_vec_dtype = None
  else: accumulator_vec_dtype=graph.accumulator_dtype.vec(8)
  topo=graph.sink.toposort(); regs=[u for u in topo if u.op is Ops.DEFINE_REG]; ends=[u for u in topo if u.op is Ops.END]
  # Validate WMMA carrier/contract ABI while the graph is still structured;
  # malformed output contracts must not reach vector decomposition.
  from tinygrad.codegen.opt.kernel_lds import validate_precontract_wmma_abi
  for u in topo:
    if u.op is Ops.WMMA:
      try: validate_precontract_wmma_abi(u, context="stage1 pipeline")
      except ValueError as exc: errors.append(str(exc))
  if graph.k_tiles == 1:
    if graph.body_readiness == "legacy" and regs: errors.append("single tile must not emit REG/RANGE/END")
    if ends or graph.body_range is not None: errors.append("single tile must not emit REG/RANGE/END")
    if any(out.dtype != accumulator_vec_dtype for out in graph.drain): errors.append("single-tile drain has mixed accumulator dtype")
  else:
    expected_slot = UOp.const(dtypes.weakint, 0) if graph.body_readiness == "sequential" else graph.body_range%graph.plan.buffer_count
    if graph.body_fragments.epoch is not graph.body_range or graph.body_fragments.slot.render() != expected_slot.render():
      errors.append("body fragment callback changed symbolic epoch/slot formula")
    expected_producer_slot = UOp.const(dtypes.weakint, 0) if graph.body_readiness == "sequential" else (graph.body_range+1)%graph.plan.buffer_count
    if graph.body_producer.epoch.render() != (graph.body_range+1).render() or \
       graph.body_producer.slot.render() != expected_producer_slot.render():
      errors.append("next producer callback changed symbolic epoch/slot formula")
    if (graph.body_readiness == "legacy" and regs != [graph.accumulator_reg]) or \
       (graph.body_readiness in ("matching", "sequential") and graph.accumulator_reg not in regs) or \
       graph.accumulator_reg.ptrdtype.base != graph.accumulator_dtype or \
       graph.accumulator_reg.ptrdtype.size < graph.subtile_count*8 or graph.accumulator_reg.ptrdtype.size%8:
      errors.append("bad shared accumulator layout")
    expected_init_dtype=graph.accumulator_dtype.vec(graph.accumulator_reg.ptrdtype.size)
    if graph.accumulator_init.op is not Ops.STORE or graph.accumulator_init.src[0].dtype != expected_init_dtype or \
       graph.accumulator_init.src[1].dtype != expected_init_dtype:
      errors.append("bad typed accumulator initializer")
    if graph.body_range is None or graph.body_range.arg[-1] is not AxisType.REDUCE: errors.append("missing symbolic body range")
    if ends != [graph.loop_end] or graph.loop_end.src[0] is not graph.body_join: errors.append("END is not rooted at body join")
    stores=[u for u in graph.body_join.backward_slice if u is not graph.accumulator_init and u.op is Ops.STORE and graph.accumulator_reg in u.src[0].backward_slice and
            (u.src[1].dtype == accumulator_vec_dtype or (u.src[1].op is Ops.UNROLL and u.src[1].dtype == graph.accumulator_dtype))]
    if len(stores) != graph.subtile_count: errors.append("body lacks one accumulator update per symbolic subtile")
    if any(node not in graph.body_join.backward_slice for node in graph.body_producer.role_nodes): errors.append("body join lacks sibling producer")
    if graph.body_readiness == "sequential":
      # A one-buffer producer must carry the current accumulator stores as a
      # dependency. Merely grouping the two operations would permit an
      # overwrite before WMMA has consumed the old fragment.
      current_updates = tuple(u for u in graph.body_join.backward_slice
                              if u.op is Ops.STORE and graph.accumulator_reg in u.src[0].backward_slice)
      if not current_updates or any(not any(dep in node.backward_slice for dep in current_updates)
                                    for node in graph.body_producer.role_nodes):
        errors.append("sequential producer is not ordered after accumulator updates")
    for out in graph.drain:
      direct=any(u.op is Ops.INDEX and u.dtype == accumulator_vec_dtype and graph.loop_end in u.backward_slice for u in out.backward_slice)
      contracted=any(u.op is Ops.CONTRACT and u.dtype == accumulator_vec_dtype and graph.loop_end in u.backward_slice for u in out.backward_slice)
      if not (direct or contracted):
        errors.append(f"drain lacks {accumulator_vec_dtype} accumulator read after END")
      if out.dtype != accumulator_vec_dtype: errors.append("drain result has mixed accumulator dtype")
  if any(u.op is Ops.REDUCE for u in topo): errors.append("forbidden Ops.REDUCE")
  return tuple(errors)
