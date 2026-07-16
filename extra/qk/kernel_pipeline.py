"""Typed plans and ownership proofs for staged kernel pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Literal, TypeVar
from tinygrad.codegen.opt.kernel_pipeline import PINNED_WMMA_VGPR_BUDGET, validate_scheduler_tile_loop_pressure

from tinygrad.dtype import DType
from tinygrad.uop.ops import AxisType, Ops, UOp

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
