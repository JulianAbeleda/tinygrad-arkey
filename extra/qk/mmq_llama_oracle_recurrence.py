"""fp16-WMMA/fp32-accumulate K32-group recurrence graph, modeled on the hand
kernel's ``compute0``/``decode_group`` (implementation plan PART I.2/I.4,
``extra/qk/prefill/wmma.py:501-654`` ``build_gemm_lds2_q4k``).

This is proof-only structure.  It consumes an already-built hierarchical LDS
stage (now sized to exactly one K32 group -- see mmq_llama_candidate_plan.py
``_geometry()``) and deliberately has no dispatch or runtime integration
surface.

Unlike the retired int8-MMQ recurrence this file replaced, there is no
post-WMMA ``dm``/``ds`` scale+bias correction: the fp16 WMMA accumulate is
already the numerically-final fp32 partial sum for this K32 group
(``d*sc*code-dmin*mn`` is folded into the decode that fills LDS, not applied
here), so ``update = previous + wmma_result`` (wmma.py I.5).
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import dtypes
from extra.qk.kernel_lds import (HierarchicalPackedRecordGroup, HierarchicalPackedRecordStage,
  validate_precontract_wmma_abi, validate_rdna3_wmma_descriptor)
from tinygrad.uop.ops import AxisType, Ops, UOp


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


def _fragment_at(stage:HierarchicalPackedRecordStage, publish:UOp, group:HierarchicalPackedRecordGroup, role:str, offset:int) -> UOp:
  base = group.persistent_byte_address if role == stage.descriptor.plan.persistent.name else group.overwriteable_byte_address
  contract = next(x for x in stage.contracts if x.role == role)
  ordered = stage.allocation.after(publish)
  esz = stage.tc.dtype_in.itemsize
  load = ordered.index(base+offset*esz, dtype=stage.tc.dtype_in).replace(
    tag=("llama_oracle_fragment_load", role, group.phase, group.group, group.persistent_k+offset)).load()
  return UOp(Ops.CONTRACT, stage.tc.dtype_in.vec(16), (load,), contract.arg,
             tag=("llama_oracle_fragment", role, group.phase, group.group, offset))


def build_llama_oracle_recurrence(stage:HierarchicalPackedRecordStage) -> LlamaOracleRecurrenceGraph:
  """Build exactly one K32 recurrence (one hand-kernel decode_group/compute0 step)."""
  if not isinstance(stage, HierarchicalPackedRecordStage): raise TypeError("expected HierarchicalPackedRecordStage")
  validate_rdna3_wmma_descriptor(stage.tc)
  if (stage.descriptor.outer_k, stage.descriptor.phase_k, stage.descriptor.group_k,
      stage.descriptor.plan.phase_count, stage.descriptor.groups_per_phase) != (32, 32, 32, 1, 1):
    raise ValueError("llama oracle recurrence requires exactly one K32 = 1 phase * 1 group")
  if stage.tc.dtype_in != dtypes.half or stage.tc.dtype_out != dtypes.float:
    raise ValueError("llama oracle recurrence requires the tinygrad RDNA3 fp16 -> fp32 descriptor")
  arg = _wmma_arg(stage)
  initial = tuple(UOp.const(dtypes.float, 0.0).replace(tag=("llama_oracle_initial_scalar_lane", lane)) for lane in range(8))
  phases, ordinal = [], 0
  previous = initial
  for phase_index, source_phase in enumerate(stage.phases):
    # Exactly one phase at this K32 granularity: no cross-phase release
    # rewiring is needed (that only mattered for the retired K256/K128
    # multi-phase int8 design).
    producer = source_phase.producer
    publish = UOp.barrier(UOp.group(stage.persistent_producer, producer)).replace(tag=("llama_oracle_publish", phase_index))
    records = []
    for source_group in source_phase.groups:
      a = tuple(_fragment_at(stage, publish, source_group, stage.descriptor.plan.persistent.name, x) for x in (0, 16))
      b = tuple(_fragment_at(stage, publish, source_group, stage.descriptor.plan.overwriteable.name, x) for x in (0, 16))
      zero = UOp.const(dtypes.float.vec(8), 0.0).replace(tag=("llama_oracle_fresh_f32_zero", ordinal))
      first = UOp(Ops.WMMA, dtypes.float.vec(8), (a[0], b[0], zero), arg,
                  tag=("llama_oracle_wmma", ordinal, 0, source_group.persistent_k))
      second = UOp(Ops.WMMA, dtypes.float.vec(8), (a[1], b[1], first), arg,
                   tag=("llama_oracle_wmma", ordinal, 1, source_group.persistent_k+16))
      # wmma.py I.5: no post-WMMA dm/ds correction -- the fp16x fp16 -> fp32
      # WMMA accumulate is already the final partial sum for this K32 group.
      update = tuple((previous[i] + second.gep(i)).replace(tag=("llama_oracle_float_update", ordinal, i)) for i in range(8))
      records.append(LlamaOracleGroupRecurrence(ordinal, phase_index, source_group.group, source_group.persistent_k,
                                                  ((a[0], b[0]), (a[1], b[1])), zero, (first, second), previous, update))
      previous, ordinal = update, ordinal+1
    release = UOp(Ops.BARRIER, dtypes.void, records[-1].update).replace(
      tag=("llama_oracle_release_after_fourth_update", phase_index))
    phases.append(LlamaOraclePhaseRecurrence(phase_index, producer, publish, tuple(records), release))
  consumer_seam = UOp(Ops.BARRIER, dtypes.void, (phases[-1].release,)+previous).replace(
    tag=("llama_oracle_subsequent_epoch_or_consumer_seam", 32))
  graph = LlamaOracleRecurrenceGraph(stage, initial, tuple(phases), consumer_seam)
  proof = prove_llama_oracle_recurrence(graph)
  if not proof.passed: raise ValueError("invalid llama oracle recurrence: " + "; ".join(proof.errors))
  return graph


def prove_llama_oracle_recurrence(graph:LlamaOracleRecurrenceGraph) -> LlamaOracleRecurrenceProof:
  """Prove exact topology and fp32-accumulate algebra; unknown or malformed structure is rejected."""
  errors: list[str] = []
  if not isinstance(graph, LlamaOracleRecurrenceGraph): raise TypeError("expected LlamaOracleRecurrenceGraph")
  stage = graph.stage
  try: validate_rdna3_wmma_descriptor(stage.tc)
  except (TypeError, ValueError) as exc: errors.append(str(exc))
  if stage.tc.dtype_in != dtypes.half or stage.tc.dtype_out != dtypes.float: errors.append("descriptor fp16 contract mismatch")
  if len(graph.phases) != 1 or len(graph.groups) != 1: errors.append("K32 must contain exactly one phase and one group")
  expected_previous = graph.initial
  for pi, phase in enumerate(graph.phases):
    if phase.phase != pi or len(phase.groups) != stage.descriptor.groups_per_phase:
      errors.append(f"phase {pi}: requires exactly {stage.descriptor.groups_per_phase} groups")
    if phase.producer is not stage.phases[pi].producer: errors.append(f"phase {pi}: producer is not the stage's own phase producer")
    expected_publish = UOp.barrier(UOp.group(stage.persistent_producer, phase.producer)).replace(tag=("llama_oracle_publish", pi))
    if phase.publish is not expected_publish: errors.append(f"phase {pi}: publish does not consume the phase producer")
    for gi, rec in enumerate(phase.groups):
      ordinal, k = pi*stage.descriptor.groups_per_phase+gi, pi*stage.descriptor.phase_k+gi*stage.descriptor.group_k
      if (rec.ordinal, rec.phase, rec.group, rec.k) != (ordinal, pi, gi, k): errors.append(f"group {ordinal}: ordinal/K mismatch")
      if rec.previous is not expected_previous: errors.append(f"group {ordinal}: recurrence predecessor mismatch")
      if rec.zero.op is not Ops.CONST or rec.zero.dtype != dtypes.float.vec(8) or rec.zero.arg != 0.0 or \
         rec.zero.tag != ("llama_oracle_fresh_f32_zero", ordinal):
        errors.append(f"group {ordinal}: missing fresh fp32 vec8 zero")
      first, second = rec.wmmas
      for si, node in enumerate((first, second)):
        try: validate_precontract_wmma_abi(node, context=f"group {ordinal} substep {si}")
        except ValueError as exc: errors.append(str(exc))
        expected_offset = k+si*16
        if node.tag != ("llama_oracle_wmma", ordinal, si, expected_offset): errors.append(f"group {ordinal}: substep/K offset mismatch")
        if node.src[:2] != rec.fragments[si]: errors.append(f"group {ordinal}: fragment wiring mismatch")
        if any(phase.publish not in fragment.backward_slice for fragment in rec.fragments[si]):
          errors.append(f"group {ordinal}: fragments do not consume the phase publish")
      if first.src[2] is not rec.zero or second.src[2] is not first: errors.append(f"group {ordinal}: WMMA chain mismatch")
      expected_update = tuple((rec.previous[i] + second.gep(i)).replace(tag=("llama_oracle_float_update", ordinal, i)) for i in range(8))
      if rec.update != expected_update: errors.append(f"group {ordinal}: float recurrence algebra mismatch")
      expected_previous = rec.update
    if phase.groups:
      last = phase.groups[-1]
      if phase.release.op is not Ops.BARRIER or any(x not in phase.release.backward_slice for x in last.update):
        errors.append(f"phase {pi}: release does not depend on the completed final group update")
  expected_seam = UOp(Ops.BARRIER, dtypes.void, (graph.phases[-1].release,)+expected_previous).replace(
    tag=("llama_oracle_subsequent_epoch_or_consumer_seam", 32))
  if graph.consumer_seam is not expected_seam or graph.phases[-1].release not in graph.consumer_seam.backward_slice:
    errors.append("final release does not order the actual subsequent epoch/consumer seam")
  return LlamaOracleRecurrenceProof(not errors, tuple(errors))


__all__ = ["LLAMA_SOURCE_COMMIT", "LLAMA_SOURCE", "LLAMA_SOURCE_ANCHOR", "LlamaOracleGroupRecurrence",
           "LlamaOraclePhaseRecurrence", "LlamaOracleRecurrenceGraph", "LlamaOracleRecurrenceProof",
           "build_llama_oracle_recurrence", "prove_llama_oracle_recurrence"]
