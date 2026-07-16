"""Fail-closed contract for a fused packed-Q4 MMQ tile.

This module owns the *logical* fused tile boundary only.  The actual graph is
handed to the existing generated Q4_K/int8-WMMA Tensor lowering; no backend
assembly, selector, or route is defined here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.kernel_pipeline import SchedulerOutputTileLoop
from tinygrad.uop.ops import UOp
from tinygrad.uop.ops import KernelLDSWindow, KernelTileGeometry

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS
from extra.qk.prefill_int8_wmma_spec import (
  Q4KInt8WMMATiledPrefillSpec, emit_q4k_int8_wmma_tiled_scheduler_tensor,
)
from extra.qk.dynamic_tile_owner import dynamic_store, own_dynamic_tiles

FUSED_Q4K_MMQ_CONTRACT = "q4k-fused-mmq-tile-v2"

# These are deliberately local contract facts.  The fused owner must not make
# route selection depend on this module, and an unknown role shape stays
# rejected until it has its own evidence.
QWEN3_14B_FUSED_ROLE_SHAPES = (
  (512, 1024, 5120),   # attn_kv
  (512, 5120, 5120),   # attn_qo
  (512, 5120, 17408),  # ffn_down
  (512, 17408, 5120),  # ffn_gate_up
)


@dataclass(frozen=True)
class FusedQ4KMMQTileSpec:
  """Logical role shape plus bounded M/N/K tile geometry.

  The phases are descriptive compiler-contract facts, not an assembly
  schedule.  Full role shapes are lowered as a loop of 16x16 output tiles and
  8 Q8_1 groups per K tile; no full [groups,M,N] RAW is admitted.
  """
  m: int = 16
  n: int = 16
  k: int = 256
  m_tile: int = 16
  n_tile: int = 16
  group_tile: int = 8
  wmma_shape: tuple[int, int, int] = (16, 16, 16)
  lifecycle: Literal["decode_lds_wmma_correct_writeback"] = "decode_lds_wmma_correct_writeback"
  extension: str | None = None

  def validate(self) -> None:
    if (self.m, self.n, self.k) != (16, 16, 256) and (self.m, self.n, self.k) not in QWEN3_14B_FUSED_ROLE_SHAPES:
      raise NotImplementedError("fused MMQ admits only the bounded tile or exact Qwen3-14B role shapes")
    if self.wmma_shape != (16, 16, 16):
      raise ValueError("the canonical fused tile requires WMMA 16x16x16")
    if self.extension is not None:
      raise NotImplementedError(f"fused MMQ extension {self.extension!r} has no lowering")
    if (self.m_tile, self.n_tile) != (16, 16) or self.group_tile != 8:
      raise NotImplementedError("fused MMQ requires bounded 16x16x8 logical M/N/K tiles")
    if self.k % Q8_1_BLOCK_ELEMS or self.k // Q8_1_BLOCK_ELEMS < self.group_tile:
      raise ValueError("K must contain complete Q8_1 groups and one bounded K tile")

  @property
  def words_shape(self) -> tuple[int]:
    return (self.n * (self.k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,)

  @property
  def xq_shape(self) -> tuple[int, int]: return (self.m, self.k)

  @property
  def xscales_shape(self) -> tuple[int, int]: return (self.m, self.k // Q8_1_BLOCK_ELEMS)

  @property
  def live_raw_elems(self) -> int: return self.m_tile * self.n_tile * self.group_tile

  def compiler_geometry(self) -> KernelTileGeometry:
    """Return the shared resource geometry consumed by compiler evidence."""
    self.validate()
    # The windows are an explicit contract placeholder for LDS staging.  The
    # generated lowering remains responsible for materializing legal storage.
    return KernelTileGeometry((self.m, self.n, self.wmma_shape[2]), (1, 1), 32, 32,
      (KernelLDSWindow("A", 0, 16 * 16 * 2, 64),
       KernelLDSWindow("B", 16 * 16 * 2, 2 * 16 * 16 * 2, 64)))


def emit_fused_q4k_mmq_tile(words: Tensor, xq: Tensor, xscales: Tensor,
                            spec: FusedQ4KMMQTileSpec = FusedQ4KMMQTileSpec()) -> Tensor:
  """Emit the bounded fused logical tile through the existing Tensor pipeline."""
  spec.validate()
  if tuple(words.shape) != spec.words_shape or tuple(xq.shape) != spec.xq_shape or tuple(xscales.shape) != spec.xscales_shape:
    raise ValueError(f"operands must have shapes {spec.words_shape}, {spec.xq_shape}, {spec.xscales_shape}")
  lowered = Q4KInt8WMMATiledPrefillSpec(n=spec.n, k=spec.k, m=spec.m,
    wmma_m=16, wmma_n=16, wmma_k=16, m_tile=spec.m_tile, n_tile=spec.n_tile, group_tile=spec.group_tile,
    role="fused_q4k_mmq", implementation="direct_tiled_wmma_v0")
  return emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, xscales, lowered)


def build_fused_q4k_mmq_dynamic_owner(words: Tensor, xq: Tensor, xscales: Tensor,
                                       output: Tensor, *, tile_count: int = 2,
                                       loop_id: int = 9600) -> UOp:
  """Build one generated owner for packed-Q4 output tiles.

  The operands are flat backing stores for ``tile_count`` 16x16 tiles.  The
  callback runs once in Python; Q4 words, Q8 activations, Q8 scales, and the
  indexed output writeback all retain the symbolic scheduler tile in their
  address calculations.  This is intentionally a graph builder (rather than
  a ``Tensor`` result): the output store is the owned effect.
  """
  spec = FusedQ4KMMQTileSpec()
  spec.validate()
  expected = (spec.words_shape[0], spec.xq_shape[0] * spec.xq_shape[1],
              spec.xscales_shape[0] * spec.xscales_shape[1], spec.m * spec.n)
  if tuple(words.shape) != (tile_count * expected[0],) or tuple(xq.shape) != (tile_count * expected[1],) \
      or tuple(xscales.shape) != (tile_count * expected[2],) or tuple(output.shape) != (tile_count * expected[3],):
    raise ValueError("dynamic fused-Q4 owner expects one flat backing store per tile")
  if words.device != xq.device or words.device != xscales.device or words.device != output.device:
    raise ValueError("dynamic fused-Q4 owner operands must share a device")
  plan = SchedulerOutputTileLoop(tile_count, loop_id=loop_id)
  try:
    return own_dynamic_tiles(plan, words, xq, xscales, output,
      weight_rows=expected[0], activation_rows=expected[1], scale_rows=expected[2], output_rows=expected[3], row_width=1,
      weight_stride=expected[0], activation_stride=expected[1], scale_stride=expected[2],
          output_stride=expected[3],
          body=lambda tile: dynamic_store(output, tile.output_indices,
            emit_fused_q4k_mmq_tile(tile.weights, tile.activation.reshape(spec.xq_shape),
                                    tile.scales.reshape(spec.xscales_shape)).reshape(-1)))
  except (NotImplementedError, ValueError, TypeError, RuntimeError) as e:
    raise NotImplementedError(f"dynamic fused-Q4 indexing cannot lower: {e}") from e


def fused_q4k_mmq_admitted(*, compile_evidence: bool = False, correctness_evidence: bool = False,
                           spec: FusedQ4KMMQTileSpec | None = None) -> bool:
  """Admission gate; both independent evidence records are required."""
  if spec is not None:
    try: spec.validate()
    except (NotImplementedError, ValueError): return False
  return bool(compile_evidence and correctness_evidence)


__all__ = ["FUSED_Q4K_MMQ_CONTRACT", "QWEN3_14B_FUSED_ROLE_SHAPES", "FusedQ4KMMQTileSpec",
           "emit_fused_q4k_mmq_tile", "build_fused_q4k_mmq_dynamic_owner", "fused_q4k_mmq_admitted"]
