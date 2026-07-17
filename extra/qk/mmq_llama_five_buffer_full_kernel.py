"""Full-grid emission seam for the source-pinned five-buffer llama MMQ graph."""
from __future__ import annotations

from dataclasses import dataclass, replace

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.kernel_writeback import (WMMAWritebackDescriptor, WMMAWritebackLayout,
  WMMAWritebackProof, build_wmma_writeback)
from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_five_buffer_graph import (FiveBufferEpochOffsets, LlamaFiveBufferGraph,
  build_llama_five_buffer_graph, five_buffer_parameters)
from extra.qk.mmq_llama_full_kernel import order_wmma_behind_lane_drain
from extra.qk.mmq_llama_oracle_epoch import build_llama_oracle_epoch_stage_five_buffer
from extra.qk.mmq_llama_oracle_recurrence import LlamaOracleRecurrenceGraph, build_llama_oracle_recurrence
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT, SOURCE_ANCHORS


SCHEMA = "tinygrad.mmq_llama_five_buffer_full_kernel.v1"
RESOURCE_BLOCKER = "AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget; Inc 0 has no spills"


@dataclass(frozen=True)
class FullGridTopology:
  grid: tuple[int, int, int]
  local_size: tuple[int, int, int] = (256, 1, 1)
  waves: tuple[int, int] = (8, 1)
  wave_size: int = 32
  lds_bytes: int = 57856


@dataclass(frozen=True)
class LlamaFiveBufferFullKernel:
  proof_graph: LlamaFiveBufferGraph
  topology: FullGridTopology
  sink: UOp
  owner_coordinates: frozenset[tuple[int, int]]
  source_commit: str
  source_anchors: tuple[tuple[str, str], ...]
  blocker: str = RESOURCE_BLOCKER
  program: UOp|None = None
  emitted: bool = False

  def __post_init__(self) -> None:
    if self.source_commit != LLAMA_SOURCE_COMMIT: raise ValueError("source identity drift")
    if self.emitted != (self.program is not None): raise ValueError("emitted must match successful to_program")
    if tuple(x.slot for x in self.proof_graph.parameters) != tuple(range(5)): raise ValueError("ABI must be exactly slots 0..4")
    if len(self.owner_coordinates) != self.proof_graph.facts.m*self.proof_graph.facts.n:
      raise ValueError("full grid must own every output exactly once")

  def epoch_offsets(self, tile_m:int, tile_n:int, epoch:int) -> FiveBufferEpochOffsets:
    facts = self.proof_graph.facts
    if not (0 <= tile_m < facts.m//128 and 0 <= tile_n < facts.n//128 and 0 <= epoch < facts.k//256):
      raise ValueError("tile/epoch outside full grid")
    m0, n0, records = tile_m*128, tile_n*128, epoch*2
    return FiveBufferEpochOffsets((n0*(facts.k//256)+epoch)*36,
      (records*facts.m+m0)*128, (records*facts.m+m0)*4, (records*facts.m+m0)*4)


def _phase_order_replacements(recurrence:LlamaOracleRecurrenceGraph, phase_index:int) -> dict[UOp, UOp]:
  """Order the four K32 groups while one Q8 phase is resident in LDS."""
  replacements:dict[UOp, UOp] = {}
  prior_drain:tuple[UOp, ...]|None = None
  for group in recurrence.phases[phase_index].groups:
    first = group.wmmas[0]
    if prior_drain is not None:
      drain = tuple(x.substitute(replacements) for x in prior_drain)
      inputs = tuple(UOp(Ops.BITCAST, x.substitute(replacements).dtype,
                         (x.substitute(replacements),)).after(*drain) for x in first.src)
      replacements[first] = first.replace(src=inputs)
    release = UOp(Ops.BARRIER, dtypes.void,
                  tuple(x.substitute(replacements) for x in group.update)).replace(
                    tag=("llama_five_buffer_phase_major_group_release", phase_index, group.ordinal))
    prior_drain = tuple(group.update) + (release,)
  return replacements


def _instantiate_phase_subtiles(recurrence:LlamaOracleRecurrenceGraph, phase_index:int, publish:UOp,
                                seeds:tuple[tuple[UOp, ...], ...]|None=None) -> tuple[tuple[UOp, ...], ...]:
  """Instantiate arithmetic/fragments only; the phase producer and publish stay shared."""
  phase, subtile = recurrence.phases[phase_index], recurrence.stage.subtile_n
  order = _phase_order_replacements(recurrence, phase_index)
  ordered_final = tuple(x.substitute(order) for x in phase.groups[-1].update)
  results:list[tuple[UOp, ...]] = []
  prior_drains:tuple[UOp, ...]|None = None
  for element in range(8):
    substitutions = {subtile: UOp.const(dtypes.weakint, element), phase.publish: publish}
    if seeds is not None:
      substitutions.update({old: new for old, new in zip(phase.groups[0].previous, seeds[element])})
    lanes = tuple(x.substitute(substitutions) for x in ordered_final)
    if prior_drains is not None:
      head = phase.groups[0].wmmas[0].substitute(substitutions)
      guarded = head.replace(src=tuple(UOp(Ops.BITCAST, s.dtype, (s,)).after(*prior_drains) for s in head.src))
      lanes = tuple(x.substitute({head: guarded}) for x in lanes)
    results.append(lanes)
    prior_drains = lanes
  return tuple(results)


def _phase_major_accumulator_vectors(recurrence:LlamaOracleRecurrenceGraph) -> tuple[UOp, ...]:
  """Expand subtiles phase-major so each Q8 panel is staged and published exactly once."""
  phase0 = _instantiate_phase_subtiles(recurrence, 0, recurrence.phases[0].publish)
  phase0_release = UOp(Ops.BARRIER, dtypes.void, tuple(x for lanes in phase0 for x in lanes)).replace(
    tag=("llama_five_buffer_phase_major_global_release", 0))
  phase1_producer = recurrence.phases[1].producer.substitute({recurrence.phases[0].release: phase0_release})
  phase1_publish = UOp.barrier(UOp.group(recurrence.stage.persistent_producer, phase1_producer)).replace(
    tag=("llama_oracle_publish", 1))
  phase1 = _instantiate_phase_subtiles(recurrence, 1, phase1_publish, phase0)
  return tuple(UOp(Ops.STACK, dtypes.float.vec(8), lanes) for lanes in phase1)


def _legacy_accumulator_vectors(values:tuple[UOp, ...], subtile:UOp, chain_head:UOp) -> tuple[UOp, ...]:
  """Retain the existing multi-epoch fallback while the phase-major prototype targets K256."""
  vectors, prior_drains = [], None
  for element in range(8):
    sub = {subtile: UOp.const(dtypes.weakint, element)}
    lanes = tuple(lane.substitute(sub) for lane in values)
    if prior_drains is not None:
      head = chain_head.substitute(sub)
      guarded = head.replace(src=tuple(UOp(Ops.BITCAST, s.dtype, (s,)).after(*prior_drains) for s in head.src))
      lanes = tuple(lane.substitute({head: guarded}) for lane in lanes)
    vectors.append(UOp(Ops.STACK, dtypes.float.vec(8), lanes))
    prior_drains = lanes
  return tuple(vectors)


def _full_grid_sink(m:int, n:int, k:int) -> UOp:
  params = five_buffer_parameters(m, n, k)
  output, q4, values, scales, sums = tuple(UOp.param(x.slot, x.dtype.ptr(x.size)) for x in params)
  block_n, block_m, local = UOp.special(n//128, "gidx0"), UOp.special(m//128, "gidx1"), UOp.special(256, "lidx0")
  wave_m, wave_n, lane = local//32, UOp.const(dtypes.weakint, 0), local%32
  previous = tuple(UOp.const(dtypes.float, 0.0) for _ in range(8))
  final, recurrences = None, []
  for epoch in range(k//256):
    records = epoch*2
    recurrence = build_llama_oracle_recurrence(build_llama_oracle_epoch_stage_five_buffer(q4, values, scales, sums,
      q4_word_offset=(block_n*128*(k//256)+epoch)*36,
      values_offset=(records*m+block_m*128)*128,
      scales_offset=(records*m+block_m*128)*4, sums_offset=(records*m+block_m*128)*4))
    exported = recurrence.export_accumulators()
    joined = tuple(previous[i] + (exported[i]-recurrence.initial[i]) for i in range(8))
    if final is not None: joined = tuple(x.after(final.consumer_seam) for x in joined)
    previous, final = joined, recurrence
    recurrences.append((epoch, recurrence))
  assert final is not None
  if len(recurrences) == 1:
    accumulators = _phase_major_accumulator_vectors(final)
  else:
    # Multi-epoch phase-major state carry is not implemented yet. Retain the existing exact fallback: order each K32
    # WMMA behind the preceding lane drain before the subtile substitution, then serialize the concrete subtiles.
    replacements, _ = order_wmma_behind_lane_drain(tuple(recurrences), "llama_five_buffer_full_grid_epoch_release")
    previous = tuple(x.substitute(replacements) for x in previous)
    chain_head = recurrences[0][1].groups[0].wmmas[0]
    accumulators = _legacy_accumulator_vectors(previous, final.stage.subtile_n, chain_head)
  plan = llama_mmq_candidate_plan()
  desc = WMMAWritebackDescriptor(plan.geometry, plan.tensor_core, dtypes.float, 8,
    # Oracle A rows are Q4/N and B rows are Q8/M, so row-major output[M,N]
    # is col * N + row in the tensor-core coordinate vocabulary.
    WMMAWritebackLayout("col", "row", n), None, True)
  writeback = build_wmma_writeback(WMMAWritebackProof.prove(desc), destination=output,
    accumulators=accumulators, wave_m=wave_m, wave_n=wave_n, lane=lane)
  # Exact aligned tiles need no predicates: grid origins are folded into the row-major destination base.
  tile_base = block_m*128*n + block_n*128
  prior = None
  for store in writeback.stores:
    pointer = output.index(tile_base + store.src[0].src[1], ptr=True)
    # Order the stores through the pointer only.  A same-dtype BITCAST on the value is a no-op that codegen folds
    # away, so an effect order hung on it lands on the scalar FP32 update underneath -- AFTER(ADD, STORE), which
    # spec_program rejects.  The pointer's INDEX is a real movement value and carries the order to the sink.
    value = store.src[1]
    if prior is not None: pointer = pointer.after(prior)
    prior = pointer.store(value).replace(tag=store.tag)
  assert prior is not None
  closed = prior.end(*prior.ranges)
  sink = UOp(Ops.SINK, dtypes.void, (closed,), KernelInfo(name="mmq_llama_five_buffer_full_grid", opts_to_apply=()))
  if sink.ranges: raise ValueError("full-grid callback ranges leaked past stores")
  return sink


def build_llama_five_buffer_full_kernel(m:int, n:int, k:int) -> LlamaFiveBufferFullKernel:
  if m % 128 or n % 128: raise ValueError("M and N must be divisible by 128; this milestone has no tails")
  proof = build_llama_five_buffer_graph(m, n, k)
  topology = FullGridTopology((n//128, m//128, 1))
  local = {(r, c) for r in range(128) for c in range(128)}
  owners = frozenset((tm*128+r, tn*128+c) for tm in range(m//128) for tn in range(n//128) for r, c in local)
  return LlamaFiveBufferFullKernel(proof, topology, _full_grid_sink(m, n, k), owners,
    LLAMA_SOURCE_COMMIT, tuple(sorted(SOURCE_ANCHORS.items())))


def compile_llama_five_buffer_full_kernel(kernel:LlamaFiveBufferFullKernel, target:str="AMD:ISA:gfx1100") -> LlamaFiveBufferFullKernel:
  """Claim emission only after the spill-free compiler accepts the final PROGRAM."""
  if not isinstance(kernel, LlamaFiveBufferFullKernel): raise TypeError("expected full-grid kernel")
  try: program = to_program(kernel.sink, AMDISARenderer(Target.parse(target)))
  except NotImplementedError as exc:
    if RESOURCE_BLOCKER not in str(exc): raise
    return kernel
  return replace(kernel, program=program, emitted=True, blocker="")


__all__ = ["RESOURCE_BLOCKER", "SCHEMA", "FullGridTopology", "LlamaFiveBufferFullKernel",
  "build_llama_five_buffer_full_kernel", "compile_llama_five_buffer_full_kernel"]
