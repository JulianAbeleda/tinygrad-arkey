"""Fail-closed graph for the source-pinned llama.cpp Q8_1 MMA recurrence.

This is proof-only structure.  It consumes an already-built hierarchical LDS
stage and deliberately has no dispatch or runtime integration surface.
"""
from __future__ import annotations

import ast, inspect, textwrap
from dataclasses import dataclass

from tinygrad import dtypes
from extra.qk.kernel_lds import (HierarchicalPackedRecordGroup, HierarchicalPackedRecordStage,
  validate_precontract_wmma_abi, validate_rdna3_wmma_descriptor)
from tinygrad.uop.ops import AxisType, Ops, UOp

from extra.qk.mmq_llama_record_producers import is_record_producer_instance_dependency, record_producer_instance_witnesses


LLAMA_SOURCE_COMMIT = "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
LLAMA_SOURCE = "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh"
LLAMA_SOURCE_ANCHOR = "vec_dot_q8_1_q8_1_mma"


@dataclass(frozen=True)
class LlamaOracleGroupRecurrence:
  ordinal: int
  phase: int
  group: int
  k: int
  fragments: tuple[tuple[UOp, UOp], tuple[UOp, UOp]]
  dm: tuple[UOp, ...]
  ds: UOp
  zero: UOp
  wmmas: tuple[UOp, UOp]
  previous: tuple[UOp, ...]
  update: tuple[UOp, ...]


@dataclass(frozen=True)
class LlamaOraclePhaseRecurrence:
  phase: int
  producer: UOp
  publish: UOp
  groups: tuple[LlamaOracleGroupRecurrence, ...]
  release: UOp


@dataclass(frozen=True)
class LlamaOracleRecurrenceGraph:
  stage: HierarchicalPackedRecordStage
  initial: tuple[UOp, ...]
  phases: tuple[LlamaOraclePhaseRecurrence, ...]
  consumer_seam: UOp

  @property
  def groups(self) -> tuple[LlamaOracleGroupRecurrence, ...]:
    return tuple(g for p in self.phases for g in p.groups)

  def export_accumulators(self) -> tuple[UOp, ...]:
    """Expose the eight scalar lane chains while leaving subtile_n symbolic."""
    return self.groups[-1].update


@dataclass(frozen=True)
class LlamaOracleRecurrenceProof:
  passed: bool
  errors: tuple[str, ...]


def _wmma_arg(stage:HierarchicalPackedRecordStage) -> tuple:
  contracts = {x.role:x for x in stage.contracts}
  caxes = tuple(UOp.range(2, 1700+i, AxisType.UPCAST) for i in range(3))
  axes = (contracts[stage.descriptor.plan.persistent.name].arg,
          contracts[stage.descriptor.plan.overwriteable.name].arg,
          tuple((x.arg[0], 2) for x in caxes))
  tc = stage.tc
  return (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, "gfx1100", tc.threads, axes, ())


def _renderer_signed_operand_contract() -> bool:
  """Read the ISA renderer's executable call contract, never infer signs from the instruction name."""
  from tinygrad.renderer.isa.amd import lower_inst
  tree = ast.parse(textwrap.dedent(inspect.getsource(lower_inst)))
  for call in (x for x in ast.walk(tree) if isinstance(x, ast.Call)):
    if isinstance(call.func, ast.Name) and call.func.id == "v_wmma_i32_16x16x16_iu8":
      neg = next((x.value for x in call.keywords if x.arg == "neg"), None)
      return isinstance(neg, ast.Constant) and neg.value == 3
  return False


def _fragment_at(stage:HierarchicalPackedRecordStage, publish:UOp, group:HierarchicalPackedRecordGroup, role:str, offset:int) -> UOp:
  base = group.persistent_byte_address if role == stage.descriptor.plan.persistent.name else group.overwriteable_byte_address
  contract = next(x for x in stage.contracts if x.role == role)
  ordered = stage.allocation.after(publish)
  load = ordered.index(base+offset, dtype=dtypes.char).replace(
    tag=("llama_oracle_fragment_load", role, group.phase, group.group, group.persistent_k+offset)).load()
  return UOp(Ops.CONTRACT, dtypes.char.vec(16), (load,), contract.arg,
             tag=("llama_oracle_fragment", role, group.phase, group.group, offset))


def _sidecars(stage:HierarchicalPackedRecordStage, publish:UOp,
              group:HierarchicalPackedRecordGroup) -> tuple[tuple[UOp, ...], UOp]:
  persistent, overwriteable = stage.descriptor.plan.persistent.name, stage.descriptor.plan.overwriteable.name
  ordered = stage.allocation.after(publish)
  dm_side = [x for x in group.sidecars if x.role == persistent and x.value.dtype == dtypes.half.vec(2)]
  ds_side = [x for x in group.sidecars if x.role == overwriteable and x.value.dtype == dtypes.half.vec(2)]
  if len(dm_side) != 1 or len(ds_side) != 1:
    raise ValueError("each K32 group requires exactly one persistent dm field and one overwriteable ds half2 sidecar")

  def load_half2(role:str, field:str, address:UOp, output_element:int|None=None) -> UOp:
    # Scalar loads stacked into the ABI's half2 avoid an unshaped vector LOAD,
    # which is not a legal full-program UOp.
    return UOp(Ops.STACK, dtypes.half.vec(2), tuple(ordered.index(address+i*2, dtype=dtypes.half).replace(
      tag=("llama_oracle_sidecar", role, field, group.phase, group.group, output_element, i)).load() for i in range(2)))

  # mmq.cuh indexes tile_x.dm at tile_C::get_i(l): each accumulator
  # element has a distinct Q4 row.  The generic stage sidecar is fragment-row
  # owned (lane%16), so derive the J-major C/A row directly from the typed
  # persistent record layout.  Q8 ds remains B-row/lane owned and constant
  # across the eight accumulator elements.
  region_name = next(x.region for x in stage.regions if x.role == persistent)
  region = stage.geometry.lds_region(region_name)
  if region.records is None: raise ValueError("persistent sidecar region lacks a record layout")
  field = region.records.component(dm_side[0].field)
  group_offset_num = group.persistent_k * field.size_bytes
  if group_offset_num % stage.descriptor.outer_k:
    raise ValueError("persistent dm group offset does not divide outer K")
  group_offset = group_offset_num // stage.descriptor.outer_k
  subtiles_m, rem = divmod(stage.geometry.tile[0], stage.geometry.waves[0]*16)
  if rem or subtiles_m <= 0: raise ValueError("persistent dm rows do not divide wave ownership")
  dms = []
  for element in range(8):
    row = (stage.threads.wave_m*subtiles_m+stage.subtile_m)*16 + stage.threads.lane//16 + element*2
    address = UOp.const(dtypes.weakint, region.base+field.offset_bytes+group_offset)+row*region.records.stride_bytes
    dms.append(load_half2(persistent, dm_side[0].field, address, element))
  ds = load_half2(overwriteable, ds_side[0].field, ds_side[0].byte_address)
  return tuple(dms), ds


def _strip_value_ordering(value: UOp) -> UOp:
  """Remove only source-pointer ordering carriers from one producer value.

  LDS transaction order remains on the store address.  Keeping the value free
  of prior stores makes the typed producer witness local to its own store.
  """
  rebuilt: dict[UOp, UOp] = {}
  for node in value.toposort():
    if node.op is Ops.AFTER:
      # The producer witness is intentionally the AFTER dependency.  Preserve
      # that typed node, while dropping only AFTER carriers whose dependency is
      # an earlier LDS STORE.
      if is_record_producer_instance_dependency(node) or any(is_record_producer_instance_dependency(x) for x in node.src[1:]):
        rebuilt[node] = node.replace(src=tuple(rebuilt.get(x, x) for x in node.src))
      else:
        rebuilt[node] = rebuilt[node.src[0]]
      continue
    src = tuple(rebuilt.get(x, x) for x in node.src)
    rebuilt[node] = node if src == node.src else node.replace(src=src)
  return rebuilt[value]


def _localize_b_producer(producer: UOp) -> UOp:
  replacements = {}
  for store in producer.toposort():
    if store.op is not Ops.STORE or not isinstance(store.tag, tuple) or len(store.tag) < 2 or store.tag[1] != "B":
      continue
    value = store.src[1]
    localized = _strip_value_ordering(value)
    witnesses = record_producer_instance_witnesses(localized)
    if len(witnesses) != 1:
      raise ValueError("Q8 B store must carry exactly one producer witness after composition")
    replacements[store] = store.replace(src=(store.src[0], localized), arg=witnesses[0])
  return producer.substitute(replacements) if replacements else producer


def build_llama_oracle_recurrence(stage:HierarchicalPackedRecordStage) -> LlamaOracleRecurrenceGraph:
  """Build exactly one K256 recurrence, while replacing release/overwrite dependencies in this proof graph."""
  if not isinstance(stage, HierarchicalPackedRecordStage): raise TypeError("expected HierarchicalPackedRecordStage")
  validate_rdna3_wmma_descriptor(stage.tc)
  if (stage.descriptor.outer_k, stage.descriptor.phase_k, stage.descriptor.group_k,
      stage.descriptor.plan.phase_count, stage.descriptor.groups_per_phase) != (256, 128, 32, 2, 4):
    raise ValueError("llama oracle recurrence requires exactly K256 = 2 phases * 4 K32 groups")
  if stage.tc.dtype_in != dtypes.char or stage.tc.dtype_out != dtypes.int:
    raise ValueError("llama oracle recurrence requires the tinygrad RDNA3 signed-char -> int32 descriptor")
  arg = _wmma_arg(stage)
  initial = tuple(UOp.const(dtypes.float, 0.0).replace(tag=("llama_oracle_initial_scalar_lane", lane)) for lane in range(8))
  phases, ordinal = [], 0
  previous = initial
  prior_release = None
  for phase_index, source_phase in enumerate(stage.phases):
    # Replace the native stage release inside the actual phase producer.  This producer is then the one published and
    # observed by every load below; there is no detached AFTER wrapper kept only as proof metadata.
    producer = source_phase.producer if phase_index == 0 else source_phase.producer.substitute(
      {stage.phases[phase_index-1].release: prior_release})
    producer = _localize_b_producer(producer)
    publish = UOp.barrier(UOp.group(stage.persistent_producer, producer)).replace(tag=("llama_oracle_publish", phase_index))
    records = []
    for source_group in source_phase.groups:
      a = tuple(_fragment_at(stage, publish, source_group, stage.descriptor.plan.persistent.name, x) for x in (0, 16))
      b = tuple(_fragment_at(stage, publish, source_group, stage.descriptor.plan.overwriteable.name, x) for x in (0, 16))
      dm, ds = _sidecars(stage, publish, source_group)
      zero = UOp.const(dtypes.int.vec(8), 0).replace(tag=("llama_oracle_fresh_i32_zero", ordinal))
      first = UOp(Ops.WMMA, dtypes.int.vec(8), (a[0], b[0], zero), arg,
                  tag=("llama_oracle_wmma", ordinal, 0, source_group.persistent_k))
      second = UOp(Ops.WMMA, dtypes.int.vec(8), (a[1], b[1], first), arg,
                   tag=("llama_oracle_wmma", ordinal, 1, source_group.persistent_k+16))
      # mmq.cuh vec_dot_q8_1_q8_1_mma: sum += dm.x*ds.x*C + dm.y*ds.y.
      update = []
      for i in range(8):
        scale = dm[i].src[0].cast(dtypes.float) * ds.src[0].cast(dtypes.float)
        bias = dm[i].src[1].cast(dtypes.float) * ds.src[1].cast(dtypes.float)
        # Preserve llama's source recurrence and rounding order:
        # (previous + scale*C) + bias.
        update.append((previous[i] + scale*second.gep(i).cast(dtypes.float) + bias).replace(
          tag=("llama_oracle_float_update", ordinal, i)))
      update = tuple(update)
      records.append(LlamaOracleGroupRecurrence(ordinal, phase_index, source_group.group, source_group.persistent_k,
                                                ((a[0], b[0]), (a[1], b[1])), dm, ds, zero, (first, second), previous, update))
      previous, ordinal = update, ordinal+1
    release = UOp(Ops.BARRIER, dtypes.void, records[-1].update).replace(
      tag=("llama_oracle_release_after_fourth_update", phase_index))
    phases.append(LlamaOraclePhaseRecurrence(phase_index, producer, publish, tuple(records), release))
    prior_release = release
  consumer_seam = UOp(Ops.BARRIER, dtypes.void, (phases[-1].release,)+previous).replace(
    tag=("llama_oracle_subsequent_epoch_or_consumer_seam", 256))
  graph = LlamaOracleRecurrenceGraph(stage, initial, tuple(phases), consumer_seam)
  proof = prove_llama_oracle_recurrence(graph)
  if not proof.passed: raise ValueError("invalid llama oracle recurrence: " + "; ".join(proof.errors))
  return graph


def prove_llama_oracle_recurrence(graph:LlamaOracleRecurrenceGraph) -> LlamaOracleRecurrenceProof:
  """Prove exact topology and algebra; unknown or malformed structure is rejected."""
  errors: list[str] = []
  if not isinstance(graph, LlamaOracleRecurrenceGraph): raise TypeError("expected LlamaOracleRecurrenceGraph")
  stage = graph.stage
  try: validate_rdna3_wmma_descriptor(stage.tc)
  except (TypeError, ValueError) as exc: errors.append(str(exc))
  if stage.tc.dtype_in != dtypes.char or stage.tc.dtype_out != dtypes.int: errors.append("descriptor signed int8 contract mismatch")
  if not _renderer_signed_operand_contract(): errors.append("AMD ISA renderer no longer asserts signed semantics for both operands")
  if len(graph.phases) != 2 or len(graph.groups) != 8: errors.append("K256 must contain exactly eight K32 groups")
  expected_previous = graph.initial
  for pi, phase in enumerate(graph.phases):
    if phase.phase != pi or len(phase.groups) != 4: errors.append(f"phase {pi}: requires exactly four groups")
    expected_producer = stage.phases[pi].producer if pi == 0 else stage.phases[pi].producer.substitute(
      {stage.phases[pi-1].release: graph.phases[pi-1].release})
    expected_producer = _localize_b_producer(expected_producer)
    if phase.producer is not expected_producer: errors.append(f"phase {pi}: producer is not rewired from recurrence release")
    expected_publish = UOp.barrier(UOp.group(stage.persistent_producer, phase.producer)).replace(tag=("llama_oracle_publish", pi))
    if phase.publish is not expected_publish: errors.append(f"phase {pi}: publish does not consume rewired producer")
    for gi, rec in enumerate(phase.groups):
      ordinal, k = pi*4+gi, (pi*128+gi*32)
      if (rec.ordinal, rec.phase, rec.group, rec.k) != (ordinal, pi, gi, k): errors.append(f"group {ordinal}: ordinal/K mismatch")
      if rec.previous is not expected_previous: errors.append(f"group {ordinal}: recurrence predecessor mismatch")
      if rec.zero.op is not Ops.CONST or rec.zero.dtype != dtypes.int.vec(8) or rec.zero.arg != 0 or rec.zero.tag != ("llama_oracle_fresh_i32_zero", ordinal):
        errors.append(f"group {ordinal}: missing fresh int32 vec8 zero")
      first, second = rec.wmmas
      for si, node in enumerate((first, second)):
        try: validate_precontract_wmma_abi(node, context=f"group {ordinal} substep {si}")
        except ValueError as exc: errors.append(str(exc))
        expected_offset = k+si*16
        if node.tag != ("llama_oracle_wmma", ordinal, si, expected_offset): errors.append(f"group {ordinal}: substep/K offset mismatch")
        if node.src[:2] != rec.fragments[si]: errors.append(f"group {ordinal}: fragment wiring mismatch")
        if any(phase.publish not in fragment.backward_slice for fragment in rec.fragments[si]):
          errors.append(f"group {ordinal}: fragments do not consume rewired publish")
      if not isinstance(rec.dm, tuple) or len(rec.dm) != 8 or any(x.dtype != dtypes.half.vec(2) for x in rec.dm):
        errors.append(f"group {ordinal}: requires one persistent dm half2 per C element")
        continue
      if any(phase.publish not in dm.backward_slice for dm in rec.dm) or phase.publish not in rec.ds.backward_slice:
        errors.append(f"group {ordinal}: sidecars do not consume rewired publish")
      if first.src[2] is not rec.zero or second.src[2] is not first: errors.append(f"group {ordinal}: WMMA chain mismatch")
      expected_update = []
      for i in range(8):
        scale = rec.dm[i].src[0].cast(dtypes.float) * rec.ds.src[0].cast(dtypes.float)
        bias = rec.dm[i].src[1].cast(dtypes.float) * rec.ds.src[1].cast(dtypes.float)
        expected_update.append((rec.previous[i] + scale*second.gep(i).cast(dtypes.float) + bias).replace(
          tag=("llama_oracle_float_update", ordinal, i)))
      expected_update = tuple(expected_update)
      if rec.update != expected_update: errors.append(f"group {ordinal}: float recurrence algebra mismatch")
      expected_previous = rec.update
    if phase.groups:
      if phase.release.op is not Ops.BARRIER or any(x not in phase.release.backward_slice for x in phase.groups[3].update):
        errors.append(f"phase {pi}: release does not depend on completed fourth float update")
      if pi+1 < len(graph.phases) and phase.release not in graph.phases[pi+1].producer.backward_slice:
        errors.append(f"phase {pi}: producer actually feeding next phase does not depend on release")
  expected_seam = UOp(Ops.BARRIER, dtypes.void, (graph.phases[-1].release,)+expected_previous).replace(
    tag=("llama_oracle_subsequent_epoch_or_consumer_seam", 256))
  if graph.consumer_seam is not expected_seam or graph.phases[-1].release not in graph.consumer_seam.backward_slice:
    errors.append("final release does not order the actual subsequent epoch/consumer seam")
  return LlamaOracleRecurrenceProof(not errors, tuple(errors))


__all__ = ["LLAMA_SOURCE_COMMIT", "LLAMA_SOURCE", "LLAMA_SOURCE_ANCHOR", "LlamaOracleGroupRecurrence",
           "LlamaOraclePhaseRecurrence", "LlamaOracleRecurrenceGraph", "LlamaOracleRecurrenceProof",
           "build_llama_oracle_recurrence", "prove_llama_oracle_recurrence"]
