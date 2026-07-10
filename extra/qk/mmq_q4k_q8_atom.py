#!/usr/bin/env python3
"""Executable Q4_K x Q8_1 MMQ atom body for bounded 14B prefill gates.

This is the first runnable backend atom behind the hybrid MMQ boundary. It is
intentionally not wired into whole-prefill selection and it does not claim GPU
performance. The value of this slice is that the atom API is now executable,
spec-validated, lifecycle-attributed, and usable by the bounded harness before
the AMD kernel body replaces the reference execution core.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from extra.qk.mmq_atom_boundary import (
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION,
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
)
from extra.qk.mmq_lifecycle import MMQLifecycleRow, zero_counters
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec, q4k_q8_1_mmq_tile_reference

BACKEND_ATOM_ID = "q4k_q8_1_mmq_reference_backed_atom_v0"


@dataclass(frozen=True)
class Q4KQ8MMQAtomResult:
  output: np.ndarray
  lifecycle: MMQLifecycleRow
  backend_atom_id: str = BACKEND_ATOM_ID

  def to_json(self) -> dict[str, Any]:
    return {
      "backend_atom_id": self.backend_atom_id,
      "route_id": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
      "classification": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION,
      "output_shape": list(self.output.shape),
      "lifecycle": self.lifecycle.to_json(),
    }


def _tile_id(spec: Q4KQ81MMQTileSpec) -> str:
  return f"m{spec.m0}_n{spec.n0}_k{spec.k0}_kg{spec.effective_k_groups}"


def _lifecycle_for_spec(spec: Q4KQ81MMQTileSpec) -> MMQLifecycleRow:
  counters = zero_counters(
    activation_quant_epochs=1,
    activation_q8_1_global_writes=spec.tile_m * spec.effective_k_groups,
    activation_q8_1_reads=spec.tile_m * spec.effective_k_groups,
    packed_weight_global_loads=spec.tile_n * (spec.effective_k_groups * 32 // 256),
    scale_min_metadata_loads=spec.tile_n * (spec.effective_k_groups * 32 // 256),
    dot_accumulation_epochs=1,
    dot_ops_or_packed_dot_insts=spec.tile_m * spec.tile_n * spec.effective_k_groups,
    intermediate_global_writes=0,
    output_store_epochs=1,
    output_stores=spec.tile_m * spec.tile_n,
  )
  return MMQLifecycleRow(role=spec.role, tile_id=_tile_id(spec), counters=counters)


def run_q4k_q8_1_mmq_tile_with_lifecycle(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                                         spec: Q4KQ81MMQTileSpec) -> Q4KQ8MMQAtomResult:
  spec.validate()
  output = q4k_q8_1_mmq_tile_reference(q4k_bytes, xq, xscales, spec)
  return Q4KQ8MMQAtomResult(output=np.asarray(output, dtype=np.float32), lifecycle=_lifecycle_for_spec(spec))


def run_q4k_q8_1_mmq_tile(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                          spec: Q4KQ81MMQTileSpec) -> np.ndarray:
  return run_q4k_q8_1_mmq_tile_with_lifecycle(q4k_bytes, xq, xscales, spec).output
