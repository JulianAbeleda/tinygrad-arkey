"""Typed plans and ownership proofs for staged kernel pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Literal, Protocol, TypeVar
from tinygrad.codegen.opt.compiler_policies import PipelinePolicy, StoragePolicy

from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp

PipelinePhase = Literal["prologue", "body", "drain"]
PipelineOp = Literal["produce", "ready", "consume", "release"]
HierarchicalLifetime = Literal["outer_epoch", "inner_phase"]
HierarchicalOp = Literal["produce", "publish", "consume", "release"]

AttachmentT = TypeVar("AttachmentT")

@dataclass(frozen=True)
class DotUpdateRecurrencePlan:
  """Shape and carrier types for a grouped mixed-dtype recurrence."""
  persistent_dtype: DType
  dot_dtype: DType
  phase_count: int
  groups_per_phase: int
  dot_substeps: int
  dot_op: Ops = Ops.WMMA

  def __post_init__(self) -> None:
    if not isinstance(self.persistent_dtype, DType) or not isinstance(self.dot_dtype, DType):
      raise TypeError("recurrence dtypes must be DType instances")
    for name in ("phase_count", "groups_per_phase", "dot_substeps"):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} must be a positive int")
    if not isinstance(self.dot_op, Ops): raise TypeError("dot_op must be an Ops member")

  @property
  def group_count(self) -> int: return self.phase_count * self.groups_per_phase

  @property
  def total_dot_count(self) -> int: return self.group_count * self.dot_substeps

@dataclass(frozen=True)
class DotUpdateAttachment(Generic[AttachmentT]):
  """Typed update sidecar and the graph values whose use is mandatory."""
  value: AttachmentT
  dependencies: tuple[UOp, ...] = ()

  def __post_init__(self) -> None:
    if not isinstance(self.dependencies, tuple) or any(not isinstance(x, UOp) for x in self.dependencies):
      raise TypeError("attachment dependencies must be an immutable tuple of UOps")

@dataclass(frozen=True)
class DotUpdateGroupContext:
  phase: int
  group: int
  ordinal: int
  predecessor_update: UOp | None

@dataclass(frozen=True)
class DotUpdateGroupRecord(Generic[AttachmentT]):
  context: DotUpdateGroupContext
  dot_zero: UOp
  dots: tuple[UOp, ...]
  persistent_before: UOp
  attachment: DotUpdateAttachment[AttachmentT]
  update: UOp

@dataclass(frozen=True)
class DotUpdateRecurrenceGraph(Generic[AttachmentT]):
  plan: DotUpdateRecurrencePlan
  initial_persistent: UOp
  groups: tuple[DotUpdateGroupRecord[AttachmentT], ...]
  result: UOp
  sink: UOp

@dataclass(frozen=True)
class DotUpdateRecurrenceProof:
  passed: bool
  errors: tuple[str, ...]
  dot_count: int
  update_count: int

def prove_dot_update_recurrence(graph: DotUpdateRecurrenceGraph[AttachmentT]) -> DotUpdateRecurrenceProof:
  """Prove exact grouped reset/update topology, failing closed on detached values."""
  errors: list[str] = []
  if not isinstance(graph, DotUpdateRecurrenceGraph): raise TypeError("expected DotUpdateRecurrenceGraph")
  plan, groups = graph.plan, graph.groups
  if not isinstance(groups, tuple): errors.append("groups must be an immutable tuple")
  if graph.initial_persistent.dtype != plan.persistent_dtype: errors.append("initial persistent accumulator dtype drift")
  if len(groups) != plan.group_count: errors.append(f"expected {plan.group_count} groups, got {len(groups)}")
  previous = graph.initial_persistent
  recorded_dots: list[UOp] = []
  recorded_updates: list[UOp] = []
  for ordinal, record in enumerate(groups):
    if not isinstance(record, DotUpdateGroupRecord): errors.append(f"group {ordinal}: invalid record"); continue
    phase, group = divmod(ordinal, plan.groups_per_phase)
    if record.context != DotUpdateGroupContext(phase, group, ordinal, None if ordinal == 0 else recorded_updates[-1]):
      errors.append(f"group {ordinal}: context or update ordering drift")
    if record.persistent_before is not previous: errors.append(f"group {ordinal}: persistent recurrence is detached")
    if record.persistent_before.dtype != plan.persistent_dtype or record.update.dtype != plan.persistent_dtype:
      errors.append(f"group {ordinal}: persistent accumulator dtype drift")
    if record.dot_zero.op is not Ops.NOOP or record.dot_zero.dtype != plan.dot_dtype or \
       record.dot_zero.arg != ("dot_zero", ordinal) or len(record.dot_zero.src) != 1 or \
       record.dot_zero.src[0].op is not Ops.CONST or record.dot_zero.src[0].arg != 0:
      errors.append(f"group {ordinal}: dot accumulator does not start at a fresh typed zero")
    if ordinal and any(record.dot_zero is old.dot_zero for old in groups[:ordinal]):
      errors.append(f"group {ordinal}: dot accumulator zero reused across groups")
    if not isinstance(record.dots, tuple) or len(record.dots) != plan.dot_substeps:
      errors.append(f"group {ordinal}: expected {plan.dot_substeps} dot results")
    prior = record.dot_zero
    for substep, dot in enumerate(record.dots):
      if not isinstance(dot, UOp) or dot.op is not plan.dot_op: errors.append(f"group {ordinal} dot {substep}: wrong dot operation")
      elif dot.dtype != plan.dot_dtype: errors.append(f"group {ordinal} dot {substep}: dot dtype drift")
      if isinstance(dot, UOp) and prior not in dot.src: errors.append(f"group {ordinal} dot {substep}: recurrence is not directly chained")
      if ordinal and substep == 0 and previous not in dot.backward_slice:
        errors.append(f"group {ordinal}: first dot is not ordered after the preceding update")
      prior = dot
    if record.dots and record.dots[-1] not in record.update.src:
      errors.append(f"group {ordinal}: update is detached from final dot")
    if record.persistent_before not in record.update.src:
      errors.append(f"group {ordinal}: update is detached from persistent accumulator")
    for dependency in record.attachment.dependencies:
      if dependency not in record.update.backward_slice: errors.append(f"group {ordinal}: update is detached from attachment")
    recorded_dots.extend(record.dots); recorded_updates.append(record.update); previous = record.update
  if graph.result is not previous: errors.append("result is detached from final update")
  topo = graph.sink.toposort()
  actual_dots = tuple(x for x in topo if x.op is plan.dot_op)
  if len(actual_dots) != plan.total_dot_count or set(actual_dots) != set(recorded_dots):
    errors.append(f"expected exactly {plan.total_dot_count} dot nodes, got {len(actual_dots)}")
  if any(update not in topo for update in recorded_updates): errors.append("an update is detached from the recurrence sink")
  return DotUpdateRecurrenceProof(not errors, tuple(errors), len(actual_dots), len(recorded_updates))

def build_dot_update_recurrence(plan: DotUpdateRecurrencePlan, initial_persistent: UOp,
                                attachments: tuple[DotUpdateAttachment[AttachmentT], ...],
                                dot: Callable[[DotUpdateGroupContext, int, UOp], UOp],
                                update: Callable[[DotUpdateGroupContext, UOp, UOp, DotUpdateAttachment[AttachmentT]], UOp],
                                *, verify: bool = True) -> DotUpdateRecurrenceGraph[AttachmentT]:
  """Build zero -> dot chain -> immediate persistent update for every planned group."""
  if not isinstance(plan, DotUpdateRecurrencePlan): raise TypeError("expected DotUpdateRecurrencePlan")
  if initial_persistent.dtype != plan.persistent_dtype: raise ValueError("initial persistent accumulator dtype drift")
  if not isinstance(attachments, tuple) or len(attachments) != plan.group_count or \
     any(not isinstance(x, DotUpdateAttachment) for x in attachments):
    raise ValueError(f"attachments must contain exactly {plan.group_count} typed sidecars")
  if not callable(dot) or not callable(update): raise TypeError("dot and update callbacks must be callable")
  persistent, records = initial_persistent, []
  for ordinal, attachment in enumerate(attachments):
    phase, group = divmod(ordinal, plan.groups_per_phase)
    context = DotUpdateGroupContext(phase, group, ordinal, None if ordinal == 0 else persistent)
    dot_zero, dot_results = UOp(Ops.NOOP, plan.dot_dtype, (UOp.const(plan.dot_dtype, 0),), ("dot_zero", ordinal)), []
    dot_acc = dot_zero
    for substep in range(plan.dot_substeps):
      dot_acc = dot(context, substep, dot_acc)
      if not isinstance(dot_acc, UOp): raise TypeError("dot callback must return a UOp")
      dot_results.append(dot_acc)
    updated = update(context, persistent, dot_acc, attachment)
    if not isinstance(updated, UOp): raise TypeError("update callback must return a UOp")
    records.append(DotUpdateGroupRecord(context, dot_zero, tuple(dot_results), persistent, attachment, updated))
    persistent = updated
  graph = DotUpdateRecurrenceGraph(plan, initial_persistent, tuple(records), persistent, UOp.sink(persistent))
  if verify and not (proof := prove_dot_update_recurrence(graph)).passed:
    raise ValueError("invalid dot/update recurrence: " + "; ".join(proof.errors))
  return graph

@dataclass(frozen=True)
class HierarchicalPipelineRole:
  """An operand role and the scope for which one production remains live."""
  name: str
  lifetime: HierarchicalLifetime

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name: raise ValueError("role name must be a non-empty string")
    if self.lifetime not in ("outer_epoch", "inner_phase"): raise ValueError("unsupported hierarchical role lifetime")

@dataclass(frozen=True)
class HierarchicalKernelPipelinePlan:
  """Generic two-level lifetime contract, independent of storage or allocation."""
  persistent: HierarchicalPipelineRole
  overwriteable: HierarchicalPipelineRole
  phase_count: int = 2

  def __post_init__(self) -> None:
    if not isinstance(self.persistent, HierarchicalPipelineRole) or self.persistent.lifetime != "outer_epoch":
      raise ValueError("persistent role must have outer_epoch lifetime")
    if not isinstance(self.overwriteable, HierarchicalPipelineRole) or self.overwriteable.lifetime != "inner_phase":
      raise ValueError("overwriteable role must have inner_phase lifetime")
    if self.persistent.name == self.overwriteable.name: raise ValueError("hierarchical roles must be distinct")
    if not isinstance(self.phase_count, int) or isinstance(self.phase_count, bool) or self.phase_count <= 0:
      raise ValueError("phase_count must be a positive int")

  @property
  def roles(self) -> tuple[HierarchicalPipelineRole, HierarchicalPipelineRole]:
    return self.persistent, self.overwriteable

@dataclass(frozen=True)
class HierarchicalLifecycleEvent:
  op: HierarchicalOp
  role: str
  phase: int | None

@dataclass(frozen=True)
class HierarchicalLifecycleProof:
  passed: bool
  errors: tuple[str, ...]
  produced: tuple[tuple[str, int | None], ...]
  consumed: tuple[tuple[str, int], ...]
  barriers: tuple[tuple[Literal["publish", "release"], int], ...]

def hierarchical_lifecycle_events(plan: HierarchicalKernelPipelinePlan) -> tuple[HierarchicalLifecycleEvent, ...]:
  """Return the sole admitted asymmetric lifecycle; publish/release are the uniform barriers."""
  if not isinstance(plan, HierarchicalKernelPipelinePlan): raise TypeError("expected HierarchicalKernelPipelinePlan")
  events = [HierarchicalLifecycleEvent("produce", plan.persistent.name, None)]
  for phase in range(plan.phase_count):
    events.extend((HierarchicalLifecycleEvent("produce", plan.overwriteable.name, phase),
                   HierarchicalLifecycleEvent("publish", plan.overwriteable.name, phase),
                   HierarchicalLifecycleEvent("consume", plan.persistent.name, phase),
                   HierarchicalLifecycleEvent("consume", plan.overwriteable.name, phase),
                   HierarchicalLifecycleEvent("release", plan.overwriteable.name, phase)))
  events.append(HierarchicalLifecycleEvent("release", plan.persistent.name, None))
  return tuple(events)

def prove_hierarchical_lifecycle(plan: HierarchicalKernelPipelinePlan,
                                 events: tuple[HierarchicalLifecycleEvent, ...]) -> HierarchicalLifecycleProof:
  """Fail-closed proof of the exact two-level role lifetime and barrier protocol."""
  if not isinstance(plan, HierarchicalKernelPipelinePlan): raise TypeError("expected HierarchicalKernelPipelinePlan")
  errors: list[str] = []
  if not isinstance(events, tuple): errors.append("events must be an immutable tuple")
  actual = events if isinstance(events, tuple) else tuple(events)
  expected = hierarchical_lifecycle_events(plan)
  for index in range(max(len(actual), len(expected))):
    if index >= len(actual): errors.append(f"event {index}: missing {expected[index].op} for role {expected[index].role}")
    elif index >= len(expected): errors.append(f"event {index}: unexpected extra event {actual[index]!r}")
    elif not isinstance(actual[index], HierarchicalLifecycleEvent): errors.append(f"event {index}: wrong event type")
    elif actual[index] != expected[index]:
      want, got = expected[index], actual[index]
      errors.append(f"event {index}: expected {want.op} role {want.role} phase {want.phase}, got {got.op} role {got.role} phase {got.phase}")

  produced = tuple((event.role, event.phase) for event in actual
                   if isinstance(event, HierarchicalLifecycleEvent) and event.op == "produce")
  consumed = tuple((event.role, event.phase) for event in actual
                   if isinstance(event, HierarchicalLifecycleEvent) and event.op == "consume" and event.phase is not None)
  barriers = tuple((event.op, event.phase) for event in actual
                   if isinstance(event, HierarchicalLifecycleEvent) and event.op in ("publish", "release") and
                   event.role == plan.overwriteable.name and event.phase is not None)
  return HierarchicalLifecycleProof(not errors, tuple(errors), produced, consumed, barriers)

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

@dataclass(frozen=True)
class SchedulerOutputTileLoop:
  """A scheduler-owned output loop with one resident WMMA carrier set.

  ``tile_count`` is deliberately a loop trip count, not an unroll factor.  The
  owner callback receives the symbolic tile index so output addressing can be
  carried by the compiled program; it must not replicate the host graph for
  each output tile.
  """
  tile_count: int
  loop_id: int = 9300
  resident_accumulator_vgprs: int = 128
  resident_fragment_vgprs: int = 64

  def __post_init__(self) -> None:
    if not isinstance(self.tile_count, int) or isinstance(self.tile_count, bool) or self.tile_count <= 0:
      raise ValueError("output tile loop count must be a positive int")
    if not isinstance(self.loop_id, int) or isinstance(self.loop_id, bool) or self.loop_id < 0:
      raise ValueError("output tile loop id must be a non-negative int")
    validate_scheduler_tile_loop_pressure(resident_accumulator_vgprs=self.resident_accumulator_vgprs,
                                          resident_fragment_vgprs=self.resident_fragment_vgprs)

@dataclass(frozen=True)
class SchedulerOutputTileIndices:
  """The symbolic coordinates owned by a tiled output producer."""
  m: UOp
  n: UOp
  group: UOp

def build_scheduler_output_tile_owner(m_plan: SchedulerOutputTileLoop, n_plan: SchedulerOutputTileLoop,
                                      group_plan: SchedulerOutputTileLoop,
                                      owner: Callable[[SchedulerOutputTileIndices], UOp]) -> UOp:
  """Lower one real M/N/group owner, without host-side tile replication.

  The callback must construct the complete producer graph from these RANGE
  values.  This deliberately admits only memory/WMMA owners: a marker graph
  can no longer make the scheduler-loop contract appear implemented.
  """
  plans = (m_plan, n_plan, group_plan)
  if any(not isinstance(x, SchedulerOutputTileLoop) for x in plans):
    raise TypeError("scheduler output tile plans must be SchedulerOutputTileLoop instances")
  if not callable(owner): raise TypeError("scheduler output tile owner must be callable")
  ranges = tuple(UOp.range(p.tile_count, p.loop_id, AxisType.LOOP) for p in plans)
  value = owner(SchedulerOutputTileIndices(*ranges))
  if not isinstance(value, UOp): raise TypeError("scheduler output tile owner must return a UOp")
  nodes = value.toposort()
  effects = tuple(x for x in nodes if x.op in (Ops.LOAD, Ops.STORE, Ops.WMMA))
  if not effects:
    raise ValueError("scheduler output tile owner must lower tensor loads, WMMA, or stores")
  for axis in ranges:
    if not any(axis in x.backward_slice_with_self for x in effects):
      raise ValueError(f"scheduler output tile index {axis.arg[0]} is detached from owner effects")
  # Every dynamic INDEX must be rooted in the supplied owner axes or in a
  # compile-time constant.  Silently accepting another RANGE would place
  # addressing outside the scheduler's ownership proof.
  allowed = set(ranges)
  for idx in (x for x in nodes if x.op is Ops.INDEX):
    dynamic = idx.src[1:]
    if any(y.op is not Ops.CONST and not (y.ranges.keys() <= allowed) for y in dynamic):
      raise ValueError("unsupported dynamic indexing outside scheduler tile ownership")
  body = value
  for axis in reversed(ranges):
    body = body if axis in body.ended_ranges else body.end(axis)
  return UOp.sink(body)

def build_scheduler_output_tile_loop(plan: SchedulerOutputTileLoop,
                                     owner: Callable[[UOp], UOp]) -> UOp:
  """Build one symbolic loop that owns all output tiles.

  This is intentionally effect-transparent: producer readiness and barriers
  remain dependencies of the owner's returned UOp, including packed-Q4
  producer contracts.  The loop only supplies output ownership and therefore
  cannot widen WMMA residency as ``tile_count`` grows.
  """
  if not isinstance(plan, SchedulerOutputTileLoop): raise TypeError("expected SchedulerOutputTileLoop")
  tile = UOp.range(plan.tile_count, plan.loop_id, AxisType.LOOP)
  value = owner(tile)
  if not isinstance(value, UOp): raise TypeError("output tile owner must return a UOp")
  # RANGE being merely a data dependency is not enough: postrange/linearize
  # only emits a loop when an effectful body is closed with END(range).  Close
  # the owner seam here, after the callback has built its complete dependency
  # graph.  In particular this keeps readiness barriers and packed-Q4 loads in
  # the body slice instead of accidentally making them host-side prerequisites.
  # ``end`` is deliberately applied to the callback result (rather than to
  # individual stores): producer graphs may contain several stores and their
  # sibling ordering must remain owned by the callback.
  return UOp.sink(value if tile in value.ended_ranges else value.end(tile))

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
  body_readiness: Literal["legacy", "matching", "sequential"] = "legacy"
  accumulator_dtype: DType = dtypes.float


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

  events = stage1_lifecycle_events(plan,k_tiles)
  zero,last=UOp.const(dtypes.weakint,0),UOp.const(dtypes.weakint,k_tiles-1)
  prologue=produce(zero,zero,None)
  if k_tiles == 1:
    frag=fragments(last,zero,prologue.ready)
    drain=tuple(accumulate(frag,UOp.const(accumulator_vec_dtype,0),i) for i in range(subtile_count))
    return KernelStage1UOpGraph(plan,k_tiles,UOp.sink(*drain),drain,None,None,None,None,None,
      prologue,None,None,frag,drain,subtile_count,events,body_readiness,accumulator_dtype)
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
    prologue,body_prod,body_frag,drain_frag,drain,subtile_count,events,body_readiness,accumulator_dtype)


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

def prove_stage1_uop_graph(graph:KernelStage1UOpGraph) -> KernelStage1UOpProof:
  lifecycle=prove_stage1_lifecycle(graph.plan,graph.k_tiles,graph.events); errors=list(lifecycle.errors)
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
  return KernelStage1UOpProof(not errors,tuple(errors),lifecycle)
