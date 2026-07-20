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
from extra.qk.mmq_llama_group_chain import chain_group_stage
from extra.qk.mmq_llama_oracle_epoch import build_llama_oracle_group_stage
from extra.qk.mmq_llama_oracle_recurrence import LlamaOracleRecurrenceGraph, build_llama_oracle_recurrence
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT, SOURCE_ANCHORS


SCHEMA = "tinygrad.mmq_llama_five_buffer_graph.v1"
BLOCKER = ("bounded one-tile source-pinned oracle proof only; no grid addressing, executable emission, "
           "dispatch routing, edge handling, or full-grid correctness claim")
# Historical name: this ABI was five split int8-MMQ buffers before the
# fp16-dequant-in-register primitive (implementation plan PART II.6).  It is
# now output + q4 (still packed, decoded in-register) + one plain fp16
# activation buffer -- three slots -- but the "five_buffer" module/class names
# are kept to avoid a repo-wide rename cascade.


@dataclass(frozen=True)
class FiveBufferParameter:
  slot: int
  name: str
  dtype: object
  size: int
  physical_shape: tuple[int, ...]


@dataclass(frozen=True)
class FiveBufferEpochOffsets:
  """Element offsets in the declared physical arrays for one K256 epoch."""
  q4: int
  activation: int


@dataclass(frozen=True)
class FiveBufferEpoch:
  ordinal: int
  k0: int
  offsets: FiveBufferEpochOffsets
  recurrences: tuple[LlamaOracleRecurrenceGraph, ...]
  previous_accumulators: tuple[UOp, ...]
  accumulators: tuple[UOp, ...]

  @property
  def recurrence(self) -> LlamaOracleRecurrenceGraph:
    """Last K32-group recurrence of this epoch (for consumer-seam ordering)."""
    return self.recurrences[-1]


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
    if tuple(x.slot for x in self.parameters) != tuple(range(3)): raise ValueError("fp16 ABI slots must be exactly 0..2")
    if self.custom_kernel is not None or self.emitted or self.routed: raise ValueError("proof graph cannot claim emission or routing")
    if any(kind.startswith("dense") or kind.startswith("dequant") for kind, _, _ in self.allocated_shapes):
      raise ValueError("dense/dequantized allocation is forbidden")

  def program(self):
    raise RuntimeError(self.blocker)


def five_buffer_parameters(m:int, n:int, k:int) -> tuple[FiveBufferParameter, ...]:
  """Return the immutable output/Q4/activation ABI (implementation plan PART II.6).

  Q4 stays packed (still decoded in-register); the three split int8-MMQ Q8
  buffers (values/scales/original-sums) collapse into one plain fp16
  activation buffer -- three slots total.
  """
  facts = LlamaFullKernelFacts(m, n, k)
  if facts.m % 128 or facts.n % 128: raise ValueError("M and N must be complete 128-row proof tiles")
  return (
    FiveBufferParameter(0, "output", dtypes.float32, m*n, (m, n)),
    FiveBufferParameter(1, "q4", dtypes.uint32, n*(k//256)*36, (n, k//256, 36)),
    FiveBufferParameter(2, "activation", dtypes.half, m*k, (m, k)),
  )


def _offsets(facts:LlamaFullKernelFacts, tile_m:int, tile_n:int, epoch:int) -> FiveBufferEpochOffsets:
  m0, n0 = tile_m*128, tile_n*128
  return FiveBufferEpochOffsets(q4=(n0*(facts.k//256)+epoch)*36, activation=m0*facts.k)


def build_llama_five_buffer_graph(m:int, n:int, k:int, *, tile_m:int=0, tile_n:int=0) -> LlamaFiveBufferGraph:
  """Compose one aligned logical tile across all K256 epochs, failing closed outside it."""
  facts, plan = LlamaFullKernelFacts(m, n, k), llama_mmq_candidate_plan()
  params = five_buffer_parameters(m, n, k)
  if not isinstance(tile_m, int) or isinstance(tile_m, bool) or not isinstance(tile_n, int) or isinstance(tile_n, bool):
    raise ValueError("tile coordinates must be integers")
  if tile_m < 0 or tile_n < 0 or (tile_m+1)*128 > m or (tile_n+1)*128 > n:
    raise ValueError("bounded proof tile is outside the aligned physical buffers")
  sources = tuple(UOp.param(x.slot, x.dtype.ptr(x.size)) for x in params)
  states, prior_release = None, None
  epochs:list[FiveBufferEpoch] = []
  for ordinal in range(k//256):
    offsets = _offsets(facts, tile_m, tile_n, ordinal)
    recurrences = []
    previous = states
    for group_index in range(8):
      stage = build_llama_oracle_group_stage(sources[1], sources[2],
        q4_word_offset=offsets.q4, q4_row_stride_words=(facts.k//256)*36, group_index=group_index,
        k_base=ordinal*256+group_index*32, activation_element_offset=offsets.activation, k_total=facts.k)
      recurrence = build_llama_oracle_recurrence(stage)
      recurrences.append(recurrence)
      states, prior_release = chain_group_stage(recurrence, ordinal*8+group_index, states, prior_release)
    accumulators = states
    epochs.append(FiveBufferEpoch(ordinal, ordinal*256, offsets, tuple(recurrences), previous, accumulators))
  allocations = tuple((x.name, x.physical_shape, x.dtype.name) for x in params) + \
                (("lds_bounded_tile", (plan.geometry.lds_bytes,), "uint8"),)
  return LlamaFiveBufferGraph(facts, params, FiveBufferTileBody(tile_m, tile_n, tuple(epochs)),
    plan.identity(), LLAMA_SOURCE_COMMIT, tuple(sorted(SOURCE_ANCHORS.items())), allocations)


def build_llama_five_buffer_bounded_sink(graph:LlamaFiveBufferGraph) -> UOp:
  """Retired for this phase.

  The old epilogue drain (``bounded_final_release``/``order_wmma_behind_lane_drain``
  in ``mmq_llama_full_kernel.py``) assumes one recurrence per K256 epoch (the
  int8 2-phase x 4-group stage).  Phase-1/2 shrank the stage granularity to one
  K32 group (``mmq_llama_candidate_plan.py`` ``_geometry()``), so a K256 epoch
  is now eight chained ``LlamaOracleRecurrenceGraph`` objects
  (``FiveBufferEpoch.recurrences``), not one.  The full-grid kernel
  (``mmq_llama_five_buffer_full_kernel.py``) does not use this bounded-proof
  epilogue -- it builds its own writeback directly off ``group_major_accumulator_vectors``.
  Rewiring this smaller bounded-tile proof path to the new per-group epilogue
  is out of scope for this phase; call sites should build off
  ``mmq_llama_five_buffer_full_kernel.py`` instead.
  """
  raise NotImplementedError(
    "build_llama_five_buffer_bounded_sink is retired: phase-1/2 moved to per-K32-group recurrences "
    "(FiveBufferEpoch.recurrences), and the old single-recurrence-per-epoch bounded_final_release epilogue "
    "was not rewired for it -- use mmq_llama_five_buffer_full_kernel.py's full-grid sink instead")


__all__ = ["BLOCKER", "SCHEMA", "FiveBufferEpoch", "FiveBufferEpochOffsets", "FiveBufferParameter",
  "FiveBufferTileBody", "LlamaFiveBufferGraph", "build_llama_five_buffer_bounded_sink",
  "build_llama_five_buffer_graph", "five_buffer_parameters"]
