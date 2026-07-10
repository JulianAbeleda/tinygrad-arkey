#!/usr/bin/env python3
"""Oracle model for llama.cpp's Q4_K MMQ cooperative tile ownership.

This is a translated structure oracle, not a production kernel and not vendored
CUDA. It mirrors the llama MMQ tile/writeback ownership so future R4 atoms can
compare against a stable cooperative-tile contract while numeric values still
come from the existing Q4_K x DS4 reference algebra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from extra.qk.layout import Q8_1_BLOCK_ELEMS
from extra.qk.mmq_q4k_q8_reference import (
  Q81MMQDS4Activation, Q4KQ81MMQTileSpec, Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
)


LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID = "llama_mmq_q4k_q8_1_coop_tile_oracle"
LLAMA_MMQ_CUH = "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh"


@dataclass(frozen=True)
class LlamaMMQOracleGeometry:
  mmq_x: int = 128
  mmq_y: int = 128
  iter_k: int = 256
  nwarps: int = 8
  warp_size: int = 32
  tile_c_i: int = 16
  tile_c_j: int = 16

  @property
  def tile_c_ne(self) -> int:
    return self.tile_c_i * self.tile_c_j

  def validate(self) -> None:
    if min(self.mmq_x, self.mmq_y, self.iter_k, self.nwarps, self.warp_size, self.tile_c_i, self.tile_c_j) <= 0:
      raise ValueError(f"invalid llama MMQ geometry: {self}")
    if self.nwarps * self.tile_c_i != self.mmq_y:
      raise ValueError(f"nwarps*tile_c_i must equal mmq_y, got {self.nwarps}*{self.tile_c_i}!={self.mmq_y}")
    if self.mmq_x % self.tile_c_j:
      raise ValueError(f"mmq_x={self.mmq_x} must be divisible by tile_c_j={self.tile_c_j}")
    if self.iter_k % 256:
      raise ValueError(f"iter_k={self.iter_k} must cover whole Q4_K blocks")

  def to_json(self) -> dict[str, Any]:
    return {
      "mmq_x": self.mmq_x, "mmq_y": self.mmq_y, "iter_k": self.iter_k, "nwarps": self.nwarps,
      "warp_size": self.warp_size, "tile_c_i": self.tile_c_i, "tile_c_j": self.tile_c_j,
      "tile_c_ne": self.tile_c_ne,
    }


@dataclass(frozen=True)
class LlamaMMQOracleResult:
  output: np.ndarray
  geometry: LlamaMMQOracleGeometry
  writeback_owners: tuple[dict[str, Any], ...]
  source_policy: dict[str, Any]
  backend_id: str = LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID

  def to_json(self) -> dict[str, Any]:
    return {
      "backend_id": self.backend_id,
      "oracle_only": True,
      "production_dispatch_changed": False,
      "default_route": "direct_packed",
      "geometry": self.geometry.to_json(),
      "writeback_owner_count": len(self.writeback_owners),
      "writeback_owners": list(self.writeback_owners),
      "source_policy": self.source_policy,
      "output_shape": list(self.output.shape),
    }


def llama_mmq_source_policy() -> dict[str, Any]:
  return {
    "mode": "translated_structure_oracle_do_not_bind_production",
    "source_clone": LLAMA_MMQ_CUH,
    "anchors": [
      "mul_mat_q_process_tile",
      "mmq_write_back_mma",
      "mmq_write_back_dp4a",
      "load_tiles_q4_K",
      "extern __shared__ int data_mul_mat_q[]",
    ],
    "vendored_cuda": False,
    "numeric_oracle": "extra.qk.mmq_q4k_q8_reference.q4k_q8_1_mmq_ds4_tile_reference",
  }


def llama_mma_writeback_owners(spec: Q4KQ81MMQTileSpec,
                               geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry()) -> tuple[dict[str, Any], ...]:
  spec.validate()
  geometry.validate()
  owners: list[dict[str, Any]] = []
  for warp_id in range(geometry.nwarps):
    m_start = warp_id * geometry.tile_c_i
    m_end = min(m_start + geometry.tile_c_i, spec.tile_m)
    if m_start >= spec.tile_m:
      continue
    for n_start in range(0, min(geometry.mmq_x, spec.tile_n), geometry.tile_c_j):
      n_end = min(n_start + geometry.tile_c_j, spec.tile_n)
      owners.append({
        "warp_id": warp_id,
        "m_range": [spec.m0 + m_start, spec.m0 + m_end],
        "n_range": [spec.n0 + n_start, spec.n0 + n_end],
        "tile_c": {"I": geometry.tile_c_i, "J": geometry.tile_c_j, "ne": geometry.tile_c_ne},
        "sum_index_model": "sum[(j0/tile_C::J)*tile_C::ne + l]",
        "writeback_model": "dst[ids_dst[j]*stride + i]",
      })
  return tuple(owners)


def run_llama_mmq_coop_tile_oracle(q4k_bytes: np.ndarray, q8_ds4: Q81MMQDS4Activation, spec: Q4KQ81MMQTileSpec,
                                   geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry()) -> LlamaMMQOracleResult:
  spec.validate()
  geometry.validate()
  if spec.activation_layout != Q8_1_MMQ_DS4_LAYOUT:
    raise ValueError(f"llama MMQ oracle requires activation_layout={Q8_1_MMQ_DS4_LAYOUT}, got {spec.activation_layout!r}")
  if spec.effective_k_groups * Q8_1_BLOCK_ELEMS > geometry.iter_k:
    raise ValueError("single-iteration oracle currently covers one llama MMQ ITER_K slice")
  if spec.tile_m > geometry.mmq_y or spec.tile_n > geometry.mmq_x:
    raise ValueError(f"oracle tile {(spec.tile_m, spec.tile_n)} exceeds llama geometry {(geometry.mmq_y, geometry.mmq_x)}")

  reference = q4k_q8_1_mmq_ds4_tile_reference(q4k_bytes, q8_ds4, spec)
  out = np.zeros_like(reference, dtype=np.float32)
  owners = llama_mma_writeback_owners(spec, geometry)
  for owner in owners:
    m0, m1 = owner["m_range"]
    n0, n1 = owner["n_range"]
    lm0, lm1 = m0 - spec.m0, m1 - spec.m0
    ln0, ln1 = n0 - spec.n0, n1 - spec.n0
    out[lm0:lm1, ln0:ln1] = reference[lm0:lm1, ln0:ln1]
  return LlamaMMQOracleResult(output=out.astype(np.float32), geometry=geometry, writeback_owners=owners,
                              source_policy=llama_mmq_source_policy())
