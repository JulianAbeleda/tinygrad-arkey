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


def llama_mma_sum_slot_mapping(spec: Q4KQ81MMQTileSpec,
                               geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry()) -> dict[str, Any]:
  """Static llama MMQ per-thread sum[] ownership model for AMD WMMA/MFMA writeback.

  This is a bounded research probe: it mirrors the source formulas around
  `sum[(j0/tile_C::J + n)*tile_C::ne + l]` and `mmq_write_back_mma`, but it
  does not claim that tinygrad has emitted or verified a production kernel.
  """
  spec.validate()
  geometry.validate()
  bounded_warps = (spec.tile_m + geometry.tile_c_i - 1) // geometry.tile_c_i
  tile_c_thread_elems = geometry.tile_c_i * geometry.tile_c_j // geometry.warp_size
  slots_per_thread = tile_c_thread_elems * ((spec.tile_n + geometry.tile_c_j - 1) // geometry.tile_c_j)

  slots: list[dict[str, Any]] = []
  covered: dict[tuple[int, int], dict[str, Any]] = {}
  duplicates: list[dict[str, Any]] = []
  for warp_id in range(min(geometry.nwarps, bounded_warps)):
    i0 = warp_id * geometry.tile_c_i
    for n_subtile in range(0, min(spec.tile_n, geometry.mmq_x), geometry.tile_c_j):
      slot_block = n_subtile // geometry.tile_c_j
      for lane_id in range(geometry.warp_size):
        for lane_l in range(tile_c_thread_elems):
          frag_linear = lane_l * geometry.warp_size + lane_id
          local_m = i0 + (frag_linear % geometry.tile_c_i)
          local_n = n_subtile + (frag_linear // geometry.tile_c_i)
          if local_m >= spec.tile_m or local_n >= spec.tile_n:
            continue
          slot = slot_block * tile_c_thread_elems + lane_l
          row = {
            "thread": {"warp_id": warp_id, "lane_id": lane_id, "threadIdx.y": warp_id, "threadIdx.x": lane_id},
            "sum_slot": slot,
            "slot_formula": "(j0/tile_C::J + n)*tile_C::ne + l",
            "l": lane_l,
            "fragment_linear": frag_linear,
            "local_m": local_m,
            "local_n": local_n,
            "m": spec.m0 + local_m,
            "n": spec.n0 + local_n,
            "tile_c": {"I": geometry.tile_c_i, "J": geometry.tile_c_j, "thread_elems": tile_c_thread_elems,
                       "layout": "DATA_LAYOUT_J_MAJOR"},
          }
          key = (row["m"], row["n"])
          if key in covered:
            duplicates.append({"first": covered[key], "duplicate": row})
          else:
            covered[key] = row
          slots.append(row)

  missing = [
    {"m": spec.m0 + m, "n": spec.n0 + n}
    for m in range(spec.tile_m) for n in range(spec.tile_n)
    if (spec.m0 + m, spec.n0 + n) not in covered
  ]
  return {
    "schema": "llama-mmq-asm-sum-slot-mapping-probe.v1",
    "candidate_id": "llama_mmq_r4_sum_slot_mapping_probe",
    "backend_atom_id": "research_only_sum_slot_static",
    "probe_kind": "sum_slot_accumulator_mapping",
    "status": "static_mapping_pass" if not missing and not duplicates else "static_mapping_fail",
    "research_only": True,
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "geometry": geometry.to_json(),
    "tile": {"m_tile": spec.tile_m, "n_tile": spec.tile_n, "m0": spec.m0, "n0": spec.n0,
             "bounded_warps": bounded_warps},
    "tile_c_thread_elems": tile_c_thread_elems,
    "slots_per_thread": slots_per_thread,
    "mapped_output_count": len(covered),
    "expected_output_count": spec.tile_m * spec.tile_n,
    "duplicate_store_count": len(duplicates),
    "missing_store_count": len(missing),
    "duplicates": duplicates,
    "missing": missing,
    "slots": tuple(slots),
    "tinygrad_asm_surface": {
      "representable_static_identity": True,
      "candidate_helper": "AMD_ISA_REG_ACCUM pinned DEFINE_REG element with compile-time index",
      "bounded_shapes": [{"M": 16, "N": 16, "K": 256}, {"M": 32, "N": 32, "K": 256}],
      "runtime_kernel_probe_status": "blocked_missing_physical_slot_introspection",
      "exact_missing_primitive_or_api": (
        "research-only AMD custom-kernel/ASM helper API that returns, for each generated output store, "
        "the source DEFINE_REG/sum[] element identity and physical VGPR or spill slot after AMD ISA lowering"
      ),
      "smallest_next_code_change": (
        "add an opt-in debug manifest from tinygrad.renderer.isa.amd for ACCUM_READ/ACCUM_WRITE and global_store "
        "instructions carrying UOp tag, pinned VGPR, thread/lane scope, and output index"
      ),
    },
    "source_anchors": [
      "float sum[mmq_x*mmq_y / (nwarps*warp_size)]",
      "sum[(j0/tile_C::J + n)*tile_C::ne + l]",
      "mmq_write_back_mma",
      "dst[ids_dst[j]*stride + i]",
    ],
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


def llama_mma_writeback_coverage(spec: Q4KQ81MMQTileSpec,
                                 geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry()) -> dict[str, Any]:
  owners = llama_mma_writeback_owners(spec, geometry)
  covered: dict[tuple[int, int], dict[str, Any]] = {}
  duplicates: list[dict[str, Any]] = []
  for fragment_id, owner in enumerate(owners):
    m0, m1 = owner["m_range"]
    n0, n1 = owner["n_range"]
    for m in range(m0, m1):
      for n in range(n0, n1):
        key = (m, n)
        point_owner = {
          "m": m,
          "n": n,
          "fragment_id": fragment_id,
          "warp_id": owner["warp_id"],
          "fragment_m_range": owner["m_range"],
          "fragment_n_range": owner["n_range"],
        }
        if key in covered:
          duplicates.append({"m": m, "n": n, "first": covered[key], "duplicate": point_owner})
        else:
          covered[key] = point_owner

  missing = [
    {"m": spec.m0 + m, "n": spec.n0 + n}
    for m in range(spec.tile_m) for n in range(spec.tile_n)
    if (spec.m0 + m, spec.n0 + n) not in covered
  ]
  return {
    "owner_fragment_count": len(owners),
    "covered_output_count": len(covered),
    "expected_output_count": spec.tile_m * spec.tile_n,
    "duplicate_store_count": len(duplicates),
    "missing_store_count": len(missing),
    "duplicates": duplicates,
    "missing": missing,
    "owners": owners,
  }


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
