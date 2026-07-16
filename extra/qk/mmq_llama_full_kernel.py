"""Bounded, production-shaped assembly graph for source-pinned llama MMQ.

This module joins all contracts that can currently be joined without claiming
that the resulting description is an executable tinygrad custom kernel.  The
compiler-observed executable blocker is recorded explicitly in ``blocker``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Callable

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_lds import contract_symbolic_upcast, lower_symbolic_barrier_dependencies
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_oracle_epoch import build_llama_oracle_epoch_stage
from extra.qk.mmq_llama_oracle_recurrence import LlamaOracleRecurrenceGraph, build_llama_oracle_recurrence
from extra.qk.mmq_llama_runtime_contract import (ConventionalRuntimeContract, ConventionalTile, GlobalAddresses, PhysicalEpochBinding,
  Grid3D, LLAMA_SOURCE_COMMIT, MMQExtents, MMQStrides, MMQTile, SOURCE_ANCHORS)

SCHEMA = "tinygrad.mmq_llama_full_kernel_graph.v1"
BLOCKER = (
  "the bounded compile probe represents one wave's eight symbolic subtiles and eight scalar lanes with destination-relative "
  "indices, not a launch grid. The generic pointer-rooted value/address effect carrier clears AFTER(CAST(int), STORE). "
  "Contracting scalar dependencies before effect barriers clears the SPEC=1 UNROLL(float) / STACK(float.vec(8)) verifier "
  "blocker while retaining 16 symbolic WMMAs. The writeback now carries each contracted lane through a typed BITCAST movement "
  "and orders that producer after the preceding lane store; this is the explicit store-to-next-producer progressive drain. "
  "Instruction selection now proves the structurally composed B fragment from its typed producer coordinates and native effect "
  "order, so progressive-C reuse collapses the 64 logical C runs to one physical drain lease without treating tags as authority. "
  "The bounded executable graph orders every subsequent K32 fragment address after the preceding eight FP32 lane updates, "
  "forming a compiler-native accumulator/drain chain while preserving all 16 WMMAs. "
  "The exact remaining bounded-probe blocker is the later spill-free allocation gate: "
  "'AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget; Inc 0 has no spills'. "
  "The probe has no grid-derived "
  "M/N addressing, edge "
  "predicates, or executable tail handling. No "
  "genuine full-grid binary emits, so execution, correctness, resource, emission, and routing gates remain unclaimed"
)


def scheduler_valid_callback_sink(store:UOp, *, name:str="mmq_llama_full_kernel") -> UOp:
  """Close callback execution ranges at STORE while keeping barrier dependencies inside an opaque CALL body.

  Closing the SINK itself changes it into an END-valued function and exposes internal AFTER(BARRIER)
  nodes to the host scheduler.  Closing the effectful write instead preserves the in-kernel barriers and
  leaves SINK as the opaque custom-kernel body expected by UOp.call.
  """
  if store.op is not Ops.STORE: raise TypeError("scheduler-valid callback seam requires the final Ops.STORE")
  closed = store.end(*store.ranges)
  sink = closed.sink(arg=KernelInfo(name=name, opts_to_apply=()))
  if sink.ranges: raise ValueError("callback execution ranges leaked past the final STORE")
  if not any(x.op is Ops.BARRIER for x in sink.toposort()): raise ValueError("callback lost its barrier visibility contract")
  return sink


def scalar_writeback_lane(graph:"LlamaFullKernelGraph", slot:int, lane:int) -> UOp:
  """Return one scalar lane chain; slot is validated but remains the symbolic subtile range."""
  if not 0 <= slot < 8 or not 0 <= lane < 8: raise ValueError("writeback slot/lane must be in [0,8)")
  scalar = UOp.const(dtypes.float, 0.0)
  for epoch in graph.body.epochs:
    value = epoch.recurrence.groups[-1].update[lane]
    scalar = scalar + (value - epoch.recurrence.initial[lane])
  return scalar


def _bounded_accumulator_drain(graph:"LlamaFullKernelGraph") -> tuple[UOp, ...]:
  """Order each K32 WMMA behind the preceding FP32 lane drain.

  The recurrence deliberately keeps the integer WMMA chain and the eight FP32
  lane chains as separate algebraic dependencies.  Without this executable
  ordering, a legal topological schedule issues all 16 WMMAs before consuming
  any C lanes, retaining every C carrier (and its address calculation) until
  the end.  Put the ordering on the next fragment's structural LDS base so
  each complete vec8 result is drained into the bounded FP32 accumulators
  before the following fragment may be consumed.  The address path is used
  because the WMMA C input must remain the renderer's exact constant carrier.
  """
  replacements:dict[UOp, UOp] = {}
  prior_drain:tuple[UOp, ...]|None = None
  final_release:UOp|None = None
  for epoch in graph.body.epochs:
    for group in epoch.recurrence.groups:
      first = group.wmmas[0]
      if prior_drain is not None:
        drain = tuple(x.substitute(replacements) for x in prior_drain)
        # The release is deliberately carried by all three WMMA inputs.  A
        # C-only dependency orders arithmetic, but leaves the prior A/B
        # fragments addressable from the final store and defeats the bounded
        # operand lifetime proof.  Typed no-op movement keeps the oracle ABI
        # intact while making the epoch boundary visible to the scheduler.
        inputs = tuple(UOp(Ops.BITCAST, x.substitute(replacements).dtype,
                           (x.substitute(replacements),)).after(*drain) for x in first.src[:2])
        seed = UOp(Ops.BITCAST, first.src[2].substitute(replacements).dtype,
                   (first.src[2].substitute(replacements),)).after(*drain)
        replacements[first] = first.replace(src=(inputs[0], inputs[1], seed))
      release = UOp(Ops.BARRIER, dtypes.void,
                    tuple(x.substitute(replacements) for x in group.update)).replace(
                      tag=("llama_full_kernel_bounded_epoch_release", epoch.ordinal, group.ordinal))
      # Keep the scalar updates as explicit dependencies in addition to the
      # tagged release marker; this preserves the oracle's recurrence witness
      # for structural consumers while the barrier supplies the lifetime seam.
      prior_drain = tuple(group.update) + (release,)
      final_release = release
  result = tuple(x.substitute(replacements) for x in graph.body.epochs[-1].accumulators)
  if final_release is not None:
    result = tuple(UOp(Ops.BITCAST, x.dtype, (x,)).after(final_release) for x in result)
  return result


def bounded_final_release(graph:"LlamaFullKernelGraph", destination:UOp) -> UOp:
  """Progressively produce and store each contracted lane, returning the ordered final STORE."""
  if not graph.body.epochs: raise ValueError("full callback requires at least one K epoch")
  recurrence = graph.body.epochs[-1].recurrence
  if destination.dtype != dtypes.float.ptr(destination.dtype.size): raise TypeError("destination must be a float pointer")
  if destination.dtype.size < 64: raise ValueError("bounded destination must cover 8 subtiles x 8 lanes")
  subtile = recurrence.stage.subtile_n
  prior = None
  for lane, value in enumerate(_bounded_accumulator_drain(graph)):
    value = contract_symbolic_upcast(lower_symbolic_barrier_dependencies(value, subtile), subtile)
    # AFTER is legal on movement values, not directly on CONTRACT.  The no-op
    # bitcast is therefore the typed effect carrier that makes the previous
    # store a dependency of the next lane producer as well as its destination.
    value = UOp(Ops.BITCAST, value.dtype, (value,))
    if prior is not None: value = value.after(prior)
    pointer = destination if prior is None else destination.after(prior)
    store = pointer.index(UOp.const(dtypes.weakint, lane*8), dtype=dtypes.float.vec(8)).store(value).replace(
      tag=("llama_full_kernel_bounded_final_release", "destination_subtile_vector", lane))
    prior = store
  assert prior is not None
  return prior


@dataclass(frozen=True)
class LlamaFullKernelFacts:
  """Dense-matmul facts: Q4 rows N, Q8 rows M, reduction K."""
  m: int
  n: int
  k: int

  def __post_init__(self) -> None:
    for name in ("m", "n", "k"):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name.upper()} must be positive")
    if self.k % 256: raise ValueError("K must be a multiple of the exact K256 epoch")


@dataclass(frozen=True)
class ScannedTargetFacts:
  """Facts supplied by runtime device scanning; none are selection labels."""
  backend: str
  architecture: str
  wave_size: int
  max_workgroup_threads: int
  lds_bytes: int
  signed_i8_wmma: bool
  total_vram_bytes: int
  free_vram_bytes: int
  provenance: str = "runtime_scan"

  def __post_init__(self) -> None:
    if not all(isinstance(x, str) and x for x in (self.backend, self.architecture, self.provenance)):
      raise ValueError("scanned target strings must be non-empty")
    for name in ("wave_size", "max_workgroup_threads", "lds_bytes", "total_vram_bytes"):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} must be positive")
    if not isinstance(self.free_vram_bytes, int) or isinstance(self.free_vram_bytes, bool) or self.free_vram_bytes < 0:
      raise ValueError("free_vram_bytes must be non-negative")
    if self.free_vram_bytes > self.total_vram_bytes: raise ValueError("free VRAM cannot exceed total VRAM")
    if not isinstance(self.signed_i8_wmma, bool): raise ValueError("signed_i8_wmma must be boolean")


@dataclass(frozen=True)
class LlamaFullKernelEpoch:
  ordinal: int
  kb0: int
  addresses: GlobalAddresses
  recurrence: LlamaOracleRecurrenceGraph
  previous: UOp
  accumulator: UOp
  binding: PhysicalEpochBinding
  previous_accumulators: tuple[UOp, ...]
  accumulators: tuple[UOp, ...]


@dataclass(frozen=True)
class LlamaFullKernelTileBody:
  """One logical grid tile; it is reused by the launch grid, not unrolled over M/N."""
  logical_index_axes: tuple[str, str, str]
  representative_tile: ConventionalTile
  epochs: tuple[LlamaFullKernelEpoch, ...]
  accumulator_dtype: object
  output_owner_count: int
  writeback_count: int
  tail_predicated: bool
  identity_ids: bool


@dataclass(frozen=True)
class LlamaFullKernelGraph:
  facts: LlamaFullKernelFacts
  target: ScannedTargetFacts
  runtime: ConventionalRuntimeContract
  body: LlamaFullKernelTileBody
  candidate_identity: str
  source_commit: str
  source_anchors: tuple[tuple[str, str], ...]
  allocated_shapes: tuple[tuple[str, tuple[int, ...], str], ...]
  blocker: str
  custom_kernel: Callable | None = None
  emitted: bool = False
  routed: bool = False

  def __post_init__(self) -> None:
    if self.source_commit != LLAMA_SOURCE_COMMIT: raise ValueError("source identity drift")
    if self.emitted != (self.custom_kernel is not None): raise ValueError("emitted must truthfully match callable existence")
    if self.routed and not self.emitted: raise ValueError("routing requires an emitted callable")
    if self.blocker and (self.emitted or self.routed or self.custom_kernel is not None):
      raise ValueError("blocked graph cannot claim a callable, emission, or routing")
    if self.body.writeback_count != self.facts.m*self.facts.n or self.body.output_owner_count != self.body.writeback_count:
      raise ValueError("each valid output must have exactly one owner and writeback")
    if any(shape == (self.facts.n, self.facts.k) and kind.startswith("dequant") for kind, shape, _ in self.allocated_shapes):
      raise ValueError("hidden full dequantized [N,K] allocation")

  @property
  def grid(self) -> Grid3D: return self.runtime.grid

  def identity(self) -> str:
    row = {"schema": SCHEMA, "facts": (self.facts.m, self.facts.n, self.facts.k),
      # Capacity/free bytes are live feasibility inputs, not kernel capability or structural identity.
      "target": (self.target.backend, self.target.architecture, self.target.wave_size, self.target.max_workgroup_threads,
                 self.target.lds_bytes, self.target.signed_i8_wmma),
      "candidate": self.candidate_identity, "source_commit": self.source_commit,
      "anchors": self.source_anchors, "grid": (self.grid.x, self.grid.y, self.grid.z), "blocker": self.blocker}
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

  def program(self) -> Callable:
    raise RuntimeError(self.blocker)


def _runtime(facts:LlamaFullKernelFacts) -> ConventionalRuntimeContract:
  # Units follow mmq.cuh: Q4 row stride is blocks, Q8 addresses are uint32 records, destination is float elements.
  q4_blocks = facts.k//256
  q8_records = facts.k//32
  return ConventionalRuntimeContract(
    MMQTile(128, 128, 256, 256, 64, 36),
    MMQExtents(facts.k, facts.n, facts.m, facts.m, facts.m, 1, 1, 1, 1),
    MMQStrides(q4_blocks, facts.n*q4_blocks, facts.m*q8_records*36, facts.m*facts.n,
               facts.n*q4_blocks, facts.m*q8_records*36, facts.m*facts.n, facts.n))


def build_llama_full_kernel_graph(m:int, n:int, k:int, *, target:ScannedTargetFacts) -> LlamaFullKernelGraph:
  """Build the bounded full-launch graph, failing closed at the executable seam."""
  facts = LlamaFullKernelFacts(m, n, k)
  runtime, plan = _runtime(facts), llama_mmq_candidate_plan()
  if target.provenance != "runtime_scan": raise ValueError("target capability must come from runtime scanning")
  if (target.backend.lower(), target.architecture.lower(), target.wave_size) != ("amd", "gfx1100", 32):
    raise ValueError("scanned target does not match the source-pinned gfx1100 wave32 candidate")
  if target.max_workgroup_threads < plan.geometry.threads or target.lds_bytes < plan.geometry.lds_bytes or not target.signed_i8_wmma:
    raise ValueError("scanned target lacks candidate workgroup/LDS/signed-i8-WMMA capability")
  # Exact physical types are retained.  These are packed source buffers, never dense/dequantized weights.
  q4 = UOp.param(0, dtypes.uint32.ptr(max(128*36, n*(k//256)*36)))
  q8 = UOp.param(1, dtypes.uint8.ptr(max(2*128*144, m*(k//32)*36)))
  representative = runtime.conventional_tile(0, 0, 0)
  previous_accumulators = tuple(UOp.const(dtypes.float, 0.0).replace(
    tag=("llama_full_kernel_initial_scalar_lane", lane)) for lane in range(8))
  epochs = []
  for ordinal, kb0 in enumerate(runtime.k_epoch_starts):
    binding = representative.bind_epoch(kb0, runtime.tile, runtime.extents)
    recurrence = build_llama_oracle_recurrence(build_llama_oracle_epoch_stage(
      q4, q8, q4_word_offset=binding.q4_word_offset, q8_byte_offset=binding.q8_byte_offset))
    # This is the actual FP32 recurrence dependency. Address rebasing remains the explicit executable blocker above.
    # Express the epoch as a delta before joining it.  Direct graph substitution is not safe here because tinygrad
    # interns the vector-zero seed with other constants in the producer graph.
    bound = recurrence.export_accumulators()
    accumulators = tuple((previous_accumulators[lane] + (value - recurrence.initial[lane])).replace(
      tag=("llama_full_kernel_fp32_epoch_join", ordinal, lane)) for lane, value in enumerate(bound))
    if ordinal: accumulators = tuple(value.after(epochs[-1].recurrence.consumer_seam) for value in accumulators)
    epochs.append(LlamaFullKernelEpoch(ordinal, kb0, binding.addresses, recurrence, previous_accumulators[0],
      accumulators[0], binding, previous_accumulators, accumulators))
    previous_accumulators = accumulators
  body = LlamaFullKernelTileBody(("block_x", "block_y", "block_z"), representative, tuple(epochs),
    dtypes.float, m*n, m*n, m % 128 != 0 or n % 128 != 0, True)
  allocations = (("packed_q4", (n, k//256, 36), "uint32"),
                 ("packed_q8_1_ds4", (m, k//32, 36), "uint8"),
                 ("output", (m, n), "float32"),
                 ("lds_bounded_tile", (plan.geometry.lds_bytes,), "uint8"))
  return LlamaFullKernelGraph(facts, target, runtime, body, plan.identity(), LLAMA_SOURCE_COMMIT,
    tuple(sorted(SOURCE_ANCHORS.items())), allocations, BLOCKER)


__all__ = ["BLOCKER", "LlamaFullKernelEpoch", "LlamaFullKernelFacts", "LlamaFullKernelGraph", "ScannedTargetFacts",
           "LlamaFullKernelTileBody", "SCHEMA", "bounded_final_release", "build_llama_full_kernel_graph", "scalar_writeback_lane",
           "scheduler_valid_callback_sink"]
