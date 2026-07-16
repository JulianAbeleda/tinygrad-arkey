"""Fail-closed contract for a fused packed-Q4 MMQ tile.

This module owns the *logical* fused tile boundary only.  The actual graph is
handed to the existing generated Q4_K/int8-WMMA Tensor lowering; no backend
assembly, selector, or route is defined here.
"""
from __future__ import annotations

from tinygrad import Tensor

from extra.qk.prefill_int8_wmma_spec import Q4KInt8WMMATiledPrefillSpec, emit_q4k_int8_wmma_tiled_scheduler_tensor
from extra.qk.q4k_fused_mmq_contract import FusedQ4KMMQTileSpec


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


def fused_q4k_mmq_admitted(*, compile_evidence: bool = False, correctness_evidence: bool = False,
                           spec: FusedQ4KMMQTileSpec | None = None) -> bool:
  """Admission gate; both independent evidence records are required."""
  if spec is not None:
    try: spec.validate()
    except (NotImplementedError, ValueError): return False
  return bool(compile_evidence and correctness_evidence)


__all__ = ["emit_fused_q4k_mmq_tile", "fused_q4k_mmq_admitted"]
