"""Descriptor-driven Q4_K x Q8_1 MMQ graph lowering.

The descriptor owns candidate geometry; the generated WMMA lowering owns the
Tensor graph and backend instruction selection.  This module intentionally
does not contain a schedule, route, or ISA implementation.
"""
from __future__ import annotations

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec
from extra.qk.prefill_int8_wmma_spec import (
  Q4KInt8WMMAPrefillSpec, emit_q4k_int8_wmma_prefill_tensor,
  Q4KInt8WMMATiledPrefillSpec, emit_q4k_int8_wmma_tiled_lifecycle_tensor,
)


def emit_q4k_q8_mmq_prefill(words: Tensor, xq: Tensor, xscales: Tensor,
                            spec: Q4KQ8MMQPrefillSpec) -> Tensor:
  """Emit the descriptor-shaped graph without compiling or dispatching it."""
  spec.validate()
  if spec.output_layout != "tokens_rows":
    raise ValueError("MMQ lowering only emits the canonical tokens_rows ABI layout")
  if any(size % tile for size, tile in ((spec.m, spec.tile_m), (spec.n, spec.tile_n),
                                        (spec.k, spec.tile_k))):
    raise ValueError("descriptor shape must be divisible by tile geometry")
  expected_words = spec.n * (spec.k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK
  expected_scales = (spec.m, spec.k // Q8_1_BLOCK_ELEMS)
  if tuple(words.shape) != (expected_words,):
    raise ValueError(f"words shape must be {(expected_words,)}, got {tuple(words.shape)}")
  if tuple(xq.shape) != (spec.m, spec.k):
    raise ValueError(f"xq shape must be {(spec.m, spec.k)}, got {tuple(xq.shape)}")
  if tuple(xscales.shape) != expected_scales:
    raise ValueError(f"xscales shape must be {expected_scales}, got {tuple(xscales.shape)}")
  if words.dtype != dtypes.uint32 or xq.dtype != dtypes.int8 or xscales.dtype != dtypes.float32:
    raise ValueError("MMQ operands have unsupported dtypes")
  if not (words.device == xq.device == xscales.device):
    raise ValueError("MMQ operands must be on the same device")

  # Use the generated tiled lifecycle whenever the candidate describes WMMA
  # tiles.  All tile/group ownership remains data in the translated spec.
  if spec.m % 16 == 0 and spec.n % 16 == 0 and spec.tile_m % 16 == 0 and spec.tile_n % 16 == 0:
    tiled = Q4KInt8WMMATiledPrefillSpec(n=spec.n, k=spec.k, m=spec.m, role=spec.role,
      m_tile=spec.tile_m, n_tile=spec.tile_n, group_tile=spec.tile_k // Q8_1_BLOCK_ELEMS)
    return emit_q4k_int8_wmma_tiled_lifecycle_tensor(words.contiguous(), xq.contiguous(),
                                                       xscales.contiguous(), tiled)

  # Small graph/oracle shapes still use the same generated primitive, without
  # inventing a vector pointer base or a backend schedule.
  generated = Q4KInt8WMMAPrefillSpec(n=spec.n, k=spec.k, m=spec.m, role=spec.role,
                                     wmma_m=spec.m if spec.m < 16 else 16,
                                     wmma_n=16, wmma_k=16, n_tile=max(16, spec.tile_n))
  return emit_q4k_int8_wmma_prefill_tensor(words.contiguous(), xq.contiguous(),
                                           xscales.contiguous(), generated, vectorized=False)


__all__ = ["Q4KQ8MMQPrefillSpec", "emit_q4k_q8_mmq_prefill"]
