"""Full-grid emission seam for the source-pinned five-buffer llama MMQ graph."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any

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
from extra.qk.mmq_llama_group_chain import group_major_accumulator_vectors
from extra.qk.mmq_llama_oracle_epoch import build_llama_oracle_group_stage
from extra.qk.mmq_llama_oracle_recurrence import LlamaOracleRecurrenceGraph, build_llama_oracle_recurrence
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT, SOURCE_ANCHORS
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import AMD_ISA_TARGET


SCHEMA = "tinygrad.mmq_llama_five_buffer_full_kernel.v1"
RESOURCE_BLOCKER = "AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget; Inc 0 has no spills"


@dataclass(frozen=True)
class FullGridTopology:
  grid: tuple[int, int, int]
  local_size: tuple[int, int, int] = (256, 1, 1)
  waves: tuple[int, int] = (8, 1)
  wave_size: int = 32
  # Was a duplicate hardcoded 57856 (the int8 ids/q8/q4 arena size), independent
  # of extra/qk/mmq_llama_candidate_plan.py's authoritative geometry.  Derive it
  # instead, so this topology always tracks the candidate plan's LDS arena size
  # (16384 for the fp16 two-region bring-up geometry).
  lds_bytes: int = field(default_factory=lambda: llama_mmq_candidate_plan().geometry.lds_bytes)


@dataclass(frozen=True)
class FullGridOwnerCoordinates:
  """Lazy ownership manifest for large M/N grids (avoids multi-million tuple sets)."""
  m: int
  n: int

  def __len__(self) -> int: return self.m * self.n
  def __contains__(self, coordinate: object) -> bool:
    return (isinstance(coordinate, tuple) and len(coordinate) == 2 and
            all(isinstance(x, int) for x in coordinate) and
            0 <= coordinate[0] < self.m and 0 <= coordinate[1] < self.n)


@dataclass(frozen=True)
class LlamaFiveBufferFullKernel:
  proof_graph: LlamaFiveBufferGraph
  topology: FullGridTopology
  sink: UOp
  owner_coordinates: frozenset[tuple[int, int]] | FullGridOwnerCoordinates
  source_commit: str
  source_anchors: tuple[tuple[str, str], ...]
  epoch_offset: int|None = None
  blocker: str = RESOURCE_BLOCKER
  program: UOp|None = None
  emitted: bool = False

  def __post_init__(self) -> None:
    if self.source_commit != LLAMA_SOURCE_COMMIT: raise ValueError("source identity drift")
    if self.emitted != (self.program is not None): raise ValueError("emitted must match successful to_program")
    if tuple(x.slot for x in self.proof_graph.parameters) != tuple(range(3)): raise ValueError("ABI must be exactly slots 0..2")
    if self.epoch_offset is not None and not 0 <= self.epoch_offset < self.proof_graph.facts.k//256:
      raise ValueError("compile-time epoch offset is outside the full-role buffers")
    if len(self.owner_coordinates) != self.proof_graph.facts.m*self.proof_graph.facts.n:
      raise ValueError("full grid must own every output exactly once")

  def epoch_offsets(self, tile_m:int, tile_n:int, epoch:int) -> FiveBufferEpochOffsets:
    facts = self.proof_graph.facts
    if not (0 <= tile_m < facts.m//128 and 0 <= tile_n < facts.n//128 and 0 <= epoch < facts.k//256):
      raise ValueError("tile/epoch outside full grid")
    m0, n0 = tile_m*128, tile_n*128
    return FiveBufferEpochOffsets((n0*(facts.k//256)+epoch)*36, m0*facts.k)


@dataclass(frozen=True)
class LlamaFiveBufferEpochOffsetFamily:
  """One full-role ABI and one compile-time-offset K256 kernel per epoch."""
  proof_graph: LlamaFiveBufferGraph
  topology: FullGridTopology
  variants: tuple[LlamaFiveBufferFullKernel, ...]

  def __post_init__(self) -> None:
    expected = tuple(range(self.proof_graph.facts.k//256))
    if tuple(variant.epoch_offset for variant in self.variants) != expected:
      raise ValueError("epoch-offset family must contain every full-role K256 epoch exactly once")
    if any(variant.proof_graph is not self.proof_graph or variant.topology != self.topology for variant in self.variants):
      raise ValueError("epoch-offset variants must share one full-role ABI and topology")

  @property
  def emitted(self) -> bool: return all(variant.emitted for variant in self.variants)

  @property
  def programs(self) -> tuple[UOp, ...]:
    if not self.emitted: raise RuntimeError("epoch-offset family has not emitted every PROGRAM")
    return tuple(variant.program for variant in self.variants if variant.program is not None)


def _full_grid_sink(m:int, n:int, k:int, *, accumulate: bool = False, epoch_offset: int|None = None) -> UOp:
  params = five_buffer_parameters(m, n, k)
  output, q4, activation = tuple(UOp.param(x.slot, x.dtype.ptr(x.size)) for x in params)
  block_n, block_m, local = UOp.special(n//128, "gidx0"), UOp.special(m//128, "gidx1"), UOp.special(256, "lidx0")
  wave_m, wave_n, lane = local//32, UOp.const(dtypes.weakint, 0), local%32
  if epoch_offset is not None and not 0 <= epoch_offset < k//256:
    raise ValueError("compile-time epoch offset is outside the full-role buffers")
  # One K256 epoch is eight chained K32-group recurrences (implementation plan
  # PART V.2 phase-1 bring-up note): the old int8 2-phase-x-4-group "phase
  # major" staging is retired -- see mmq_llama_group_chain.py.
  recurrences = []
  for epoch in range(k//256) if epoch_offset is None else (epoch_offset,):
    q4_word_offset = (block_n*128*(k//256)+epoch)*36
    activation_element_offset = (block_m*128)*k
    for group_index in range(8):
      stage = build_llama_oracle_group_stage(q4, activation,
        q4_word_offset=q4_word_offset, q4_row_stride_words=(k//256)*36, group_index=group_index,
        k_base=epoch*256+group_index*32, activation_element_offset=activation_element_offset, k_total=k)
      recurrences.append(build_llama_oracle_recurrence(stage))
  accumulators = group_major_accumulator_vectors(tuple(recurrences))
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
    # Keep the canonical INDEX address free of AFTER wrappers. Ordering on an
    # address can be stripped by AMD's linear-dependency pass (turning a
    # dynamic tile address into v0); carry ordering on the STORE pointer, while
    # a K-tiled accumulation LOAD is ordered separately.
    base_pointer = output.index(tile_base + store.src[0].src[1], ptr=True)
    pointer = base_pointer
    value = store.src[1]
    if accumulate:
      # K-tiled adapters launch this 256-wide kernel repeatedly.  The first
      # epoch overwrites output; subsequent epochs add into the prior FP32
      # tile in-place while preserving the same owner/order proof.
      prior_value = base_pointer.load()
      if prior is not None: prior_value = prior_value.after(prior)
      value = prior_value.cast(dtypes.float) + value
    if prior is not None: pointer = pointer.after(prior)
    prior = pointer.store(value).replace(tag=store.tag)
  assert prior is not None
  closed = prior.end(*prior.ranges)
  sink = UOp(Ops.SINK, dtypes.void, (closed,), KernelInfo(name="mmq_llama_five_buffer_full_grid_accumulate" if accumulate else "mmq_llama_five_buffer_full_grid", opts_to_apply=()))
  if sink.ranges: raise ValueError("full-grid callback ranges leaked past stores")
  return sink


def build_llama_five_buffer_full_kernel(m:int, n:int, k:int, *, accumulate: bool = False) -> LlamaFiveBufferFullKernel:
  if m % 128 or n % 128: raise ValueError("M and N must be divisible by 128; this milestone has no tails")
  proof = build_llama_five_buffer_graph(m, n, k)
  topology = FullGridTopology((n//128, m//128, 1))
  local = {(r, c) for r in range(128) for c in range(128)}
  owners = (frozenset((tm*128+r, tn*128+c) for tm in range(m//128) for tn in range(n//128) for r, c in local)
            if m*n <= 1_000_000 else FullGridOwnerCoordinates(m, n))
  return LlamaFiveBufferFullKernel(proof, topology, _full_grid_sink(m, n, k, accumulate=accumulate), owners,
    LLAMA_SOURCE_COMMIT, tuple(sorted(SOURCE_ANCHORS.items())))


def build_llama_five_buffer_epoch_offset_family(m:int, n:int, k:int) -> LlamaFiveBufferEpochOffsetFamily:
  """Build K256 kernels that address one epoch inside the same full-role base buffers."""
  if m % 128 or n % 128 or k % 256:
    raise ValueError("epoch-offset family requires aligned M/N and K divisible by 256")
  proof = build_llama_five_buffer_graph(m, n, k)
  topology = FullGridTopology((n//128, m//128, 1))
  local = {(r, c) for r in range(128) for c in range(128)}
  owners = (frozenset((tm*128+r, tn*128+c) for tm in range(m//128) for tn in range(n//128) for r, c in local)
            if m*n <= 1_000_000 else FullGridOwnerCoordinates(m, n))
  variants = tuple(LlamaFiveBufferFullKernel(
    proof, topology, _full_grid_sink(m, n, k, accumulate=True, epoch_offset=epoch), owners,
    LLAMA_SOURCE_COMMIT, tuple(sorted(SOURCE_ANCHORS.items())), epoch_offset=epoch)
    for epoch in range(k//256))
  return LlamaFiveBufferEpochOffsetFamily(proof, topology, variants)


def compile_llama_five_buffer_full_kernel(kernel:LlamaFiveBufferFullKernel, target:str=AMD_ISA_TARGET) -> LlamaFiveBufferFullKernel:
  """Claim emission only after the spill-free compiler accepts the final PROGRAM."""
  if not isinstance(kernel, LlamaFiveBufferFullKernel): raise TypeError("expected full-grid kernel")
  try: program = to_program(kernel.sink, AMDISARenderer(Target.parse(target)))
  except NotImplementedError as exc:
    # Two distinct AMDISARenderer resource gates observed at this fp16 K32-group
    # geometry: the regalloc spill-decision gate (RESOURCE_BLOCKER, "Inc 0 has
    # no spills") and an earlier ISA-selection register-stage-lease pool
    # overflow ("AMD:ISA vgpr lease exceeds virtual pool") that fires before
    # spill accounting is even reached.  Both are the renderer failing closed
    # on register pressure it cannot satisfy spill-free for this geometry, not
    # a crash in this stack -- treat both as the same fail-closed blocker
    # rather than letting the second one propagate as an uncaught exception.
    if RESOURCE_BLOCKER not in str(exc) and not str(exc).startswith("AMD:ISA "): raise
    return kernel
  return replace(kernel, program=program, emitted=True, blocker="")


def compile_llama_five_buffer_epoch_offset_family(family:LlamaFiveBufferEpochOffsetFamily,
                                                  target:str=AMD_ISA_TARGET) -> LlamaFiveBufferEpochOffsetFamily:
  """Emit every static-offset PROGRAM; compilation remains CPU-only."""
  if not isinstance(family, LlamaFiveBufferEpochOffsetFamily): raise TypeError("expected epoch-offset family")
  return replace(family, variants=tuple(compile_llama_five_buffer_full_kernel(variant, target) for variant in family.variants))


def bind_llama_five_buffer_epoch_offset_calls(family:LlamaFiveBufferEpochOffsetFamily,
                                              buffers:tuple[Any, Any, Any],
                                              *, output_is_zeroed: bool) -> Any:
  """Bind every emitted variant to the same three full-role buffer identities (output/q4/activation)."""
  if not isinstance(family, LlamaFiveBufferEpochOffsetFamily): raise TypeError("expected epoch-offset family")
  if len(buffers) != 3: raise ValueError("epoch-offset family requires exactly three full-role buffers")
  if output_is_zeroed is not True:
    raise ValueError("first accumulating epoch requires an explicitly zeroed full-role output")
  for value, parameter in zip(buffers, family.proof_graph.parameters):
    shape = getattr(value, "shape", None)
    if getattr(value, "dtype", None) != parameter.dtype or not isinstance(shape, tuple) or prod(shape) != parameter.size:
      raise ValueError(f"full-role buffer {parameter.name!r} differs from the family ABI")
  output, inputs = buffers[0], buffers[1:]
  for program in family.programs:
    output = output.custom_kernel(*inputs, fxn=lambda *_args, program=program: program)[0]
  return output


__all__ = ["AMD_ISA_TARGET", "RESOURCE_BLOCKER", "SCHEMA", "FullGridTopology", "FullGridOwnerCoordinates", "LlamaFiveBufferFullKernel",
  "LlamaFiveBufferEpochOffsetFamily", "bind_llama_five_buffer_epoch_offset_calls",
  "build_llama_five_buffer_epoch_offset_family", "build_llama_five_buffer_full_kernel",
  "compile_llama_five_buffer_epoch_offset_family", "compile_llama_five_buffer_full_kernel"]
