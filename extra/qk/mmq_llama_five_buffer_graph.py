"""Bounded five-buffer composition of the source-pinned llama MMQ oracle.

This is a one-tile proof graph, not a launch grid or an emitted kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import dtypes
from tinygrad.uop.ops import UOp

from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_full_kernel import (LlamaFullKernelFacts, bounded_final_release,
  scheduler_valid_callback_sink)
from extra.qk.mmq_llama_oracle_epoch import build_llama_oracle_epoch_stage_five_buffer
from extra.qk.mmq_llama_oracle_recurrence import LlamaOracleRecurrenceGraph, build_llama_oracle_recurrence
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT, SOURCE_ANCHORS


SCHEMA = "tinygrad.mmq_llama_five_buffer_graph.v1"
BLOCKER = ("bounded one-tile source-pinned oracle proof only; no grid addressing, executable emission, "
           "dispatch routing, edge handling, or full-grid correctness claim")


@dataclass(frozen=True)
class FiveBufferParameter:
  slot: int
  name: str
  dtype: object
  size: int
  physical_shape: tuple[int, ...]


@dataclass(frozen=True)
class FiveBufferEpochOffsets:
  """Element offsets in the declared split physical arrays."""
  q4: int
  values: int
  scales: int
  sums: int


@dataclass(frozen=True)
class FiveBufferEpoch:
  ordinal: int
  k0: int
  offsets: FiveBufferEpochOffsets
  recurrence: LlamaOracleRecurrenceGraph
  previous_accumulators: tuple[UOp, ...]
  accumulators: tuple[UOp, ...]


@dataclass(frozen=True)
class FiveBufferTileBody:
  tile_m: int
  tile_n: int
  epochs: tuple[FiveBufferEpoch, ...]


@dataclass(frozen=True)
class LlamaFiveBufferGraph:
  facts: LlamaFullKernelFacts
  parameters: tuple[FiveBufferParameter, ...]
  body: FiveBufferTileBody
  candidate_identity: str
  source_commit: str
  source_anchors: tuple[tuple[str, str], ...]
  allocated_shapes: tuple[tuple[str, tuple[int, ...], str], ...]
  blocker: str = BLOCKER
  custom_kernel: None = None
  emitted: bool = False
  routed: bool = False

  def __post_init__(self) -> None:
    if self.source_commit != LLAMA_SOURCE_COMMIT: raise ValueError("source identity drift")
    if tuple(x.slot for x in self.parameters) != tuple(range(5)): raise ValueError("five-buffer ABI slots must be exactly 0..4")
    if self.custom_kernel is not None or self.emitted or self.routed: raise ValueError("proof graph cannot claim emission or routing")
    if any(kind.startswith("dense") or kind.startswith("dequant") for kind, _, _ in self.allocated_shapes):
      raise ValueError("dense/dequantized allocation is forbidden")

  def program(self):
    raise RuntimeError(self.blocker)


def five_buffer_parameters(m:int, n:int, k:int) -> tuple[FiveBufferParameter, ...]:
  """Return the immutable output/Q4/Q8-values/scales/original-sums ABI."""
  facts = LlamaFullKernelFacts(m, n, k)
  if facts.m % 128 or facts.n % 128: raise ValueError("M and N must be complete 128-row proof tiles")
  return (
    FiveBufferParameter(0, "output", dtypes.float32, m*n, (m, n)),
    FiveBufferParameter(1, "q4", dtypes.uint32, n*(k//256)*36, (n, k//256, 36)),
    FiveBufferParameter(2, "q8_values", dtypes.int8, (k//128)*m*128, (k//128, m, 128)),
    FiveBufferParameter(3, "q8_scales", dtypes.float32, (k//128)*m*4, (k//128, m, 4)),
    FiveBufferParameter(4, "q8_original_sums", dtypes.float32, (k//128)*m*4, (k//128, m, 4)),
  )


def _offsets(facts:LlamaFullKernelFacts, tile_m:int, tile_n:int, epoch:int) -> FiveBufferEpochOffsets:
  m0, n0, records = tile_m*128, tile_n*128, epoch*2
  return FiveBufferEpochOffsets(
    q4=(n0*(facts.k//256)+epoch)*36,
    values=(records*facts.m+m0)*128,
    scales=(records*facts.m+m0)*4,
    sums=(records*facts.m+m0)*4)


def build_llama_five_buffer_graph(m:int, n:int, k:int, *, tile_m:int=0, tile_n:int=0) -> LlamaFiveBufferGraph:
  """Compose one aligned logical tile across all K256 epochs, failing closed outside it."""
  facts, plan = LlamaFullKernelFacts(m, n, k), llama_mmq_candidate_plan()
  params = five_buffer_parameters(m, n, k)
  if not isinstance(tile_m, int) or isinstance(tile_m, bool) or not isinstance(tile_n, int) or isinstance(tile_n, bool):
    raise ValueError("tile coordinates must be integers")
  if tile_m < 0 or tile_n < 0 or (tile_m+1)*128 > m or (tile_n+1)*128 > n:
    raise ValueError("bounded proof tile is outside the aligned physical buffers")
  sources = tuple(UOp.param(x.slot, x.dtype.ptr(x.size)) for x in params)
  previous = tuple(UOp.const(dtypes.float, 0.0).replace(tag=("llama_five_buffer_initial_lane", lane)) for lane in range(8))
  epochs:list[FiveBufferEpoch] = []
  for ordinal in range(k//256):
    offsets = _offsets(facts, tile_m, tile_n, ordinal)
    stage = build_llama_oracle_epoch_stage_five_buffer(sources[1], sources[2], sources[3], sources[4],
      q4_word_offset=offsets.q4, values_offset=offsets.values,
      scales_offset=offsets.scales, sums_offset=offsets.sums)
    recurrence = build_llama_oracle_recurrence(stage)
    exported = recurrence.export_accumulators()
    accumulators = tuple((previous[lane] + (value-recurrence.initial[lane])).replace(
      tag=("llama_five_buffer_fp32_epoch_join", ordinal, lane)) for lane, value in enumerate(exported))
    if epochs: accumulators = tuple(x.after(epochs[-1].recurrence.consumer_seam) for x in accumulators)
    epochs.append(FiveBufferEpoch(ordinal, ordinal*256, offsets, recurrence, previous, accumulators))
    previous = accumulators
  allocations = tuple((x.name, x.physical_shape, x.dtype.name) for x in params) + \
                (("lds_bounded_tile", (plan.geometry.lds_bytes,), "uint8"),)
  return LlamaFiveBufferGraph(facts, params, FiveBufferTileBody(tile_m, tile_n, tuple(epochs)),
    plan.identity(), LLAMA_SOURCE_COMMIT, tuple(sorted(SOURCE_ANCHORS.items())), allocations)


def build_llama_five_buffer_bounded_sink(graph:LlamaFiveBufferGraph) -> UOp:
  """Build only the existing 8xvec8 progressive proof drain into ABI slot zero."""
  if not isinstance(graph, LlamaFiveBufferGraph): raise TypeError("expected LlamaFiveBufferGraph")
  output = UOp.param(0, dtypes.float32.ptr(graph.parameters[0].size))
  output_base = graph.body.tile_m*128*graph.facts.n + graph.body.tile_n*128
  store = bounded_final_release(graph, output.index(UOp.const(dtypes.weakint, output_base), ptr=True))  # type: ignore[arg-type]
  return scheduler_valid_callback_sink(store, name="mmq_llama_five_buffer_bounded_proof")


__all__ = ["BLOCKER", "SCHEMA", "FiveBufferEpoch", "FiveBufferEpochOffsets", "FiveBufferParameter",
  "FiveBufferTileBody", "LlamaFiveBufferGraph", "build_llama_five_buffer_bounded_sink",
  "build_llama_five_buffer_graph", "five_buffer_parameters"]
