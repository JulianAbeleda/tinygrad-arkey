#!/usr/bin/env python3
"""Bounded Q4_K/Q8_1 MMQ harness for the 14B ffn_gate_up candidate.

This is an opt-in diagnostic harness only. It does not bind the model prefill
route. The reference backend is runnable now; the atom backend is deliberately
fail-loud until the hand MMQ atom lands.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Literal

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q8_1_dequantize
from extra.qk.q4k_q8_fixture import (
  ACTIVATION_LAYOUT_MMQ_DS4, ACTIVATION_LAYOUT_ROW_MAJOR, Q8ActivationInputs, make_finite_q4k_bytes,
  make_q8_activation_inputs,
)
from extra.qk.mmq_q4k_q8_reference import (
  Q4KQ81MMQTileSpec, Q8_1_MMQ_DS4_BLOCK_ELEMS,
  Q8_1_MMQ_DS4_GROUPS_PER_BLOCK, Q8_1_MMQ_DS4_LAYOUT, Q8_1_ROW_MAJOR_LAYOUT, describe_q4k_q8_1_mmq_tile,
  q4k_q8_1_mmq_ds4_tile_reference, q4k_q8_1_mmq_tile_reference,
)
from extra.qk.mmq_llama_oracle import (
  LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID, llama_mmq_source_policy, run_llama_mmq_coop_tile_oracle,
)
from extra.qk.mmq_epoch_manifest_export import build_amd_isa_proof_manifest_bundle

ROLE = "ffn_gate_up"
M = 512
N = 17408
K = 5120
QUANT = "Q4_K"
ACTIVATION = "Q8_1"
CANDIDATE_ROUTE_ID = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
PUBLIC_LABEL = "hybrid_machine_search_mmq"
PRIMITIVE_CLASS = "compiler_primitive_spec_owned__hand_mmq_backend_atom"
COMPARATOR_ID = "direct_packed"
STAGED_DS4_BACKEND_ID = "q4k_q8_1_mmq_amd_staged_ds4_atom_v0"
AMD_DS4_WARP_BACKEND_ID = "q4k_q8_1_mmq_amd_ds4_warp_atom_v0"
AMD_DS4_DOT4X4_BACKEND_ID = "q4k_q8_1_mmq_amd_ds4_dot4x4_atom_v0"
AMD_DS4_LDS_SKELETON_BACKEND_ID = "q4k_q8_1_mmq_amd_ds4_lds_skeleton_atom_v0"
AMD_DS4_COOP_TILE_BACKEND_ID = "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"
FULL_GRID_BACKEND_ID = "q4k_q8_1_mmq_amd_isa_full_grid_v0"
LLAMA_MMQ_GEOMETRY = {"mmq_x": 128, "mmq_y": 128, "iter_k": 256, "nwarps": 8}
MMQ_DS4_BLOCK_ELEMS = Q8_1_MMQ_DS4_BLOCK_ELEMS
MMQ_DS4_GROUPS_PER_BLOCK = Q8_1_MMQ_DS4_GROUPS_PER_BLOCK


class MMQAtomUnavailableError(RuntimeError):
  pass


@dataclass(frozen=True)
class BoundedMMQConfig:
  m_tile: int = 16
  n_tile: int = 16
  k_groups: int = 8
  m_tiles: int = 1
  n_tiles: int = 1
  warmups: int = 0
  rounds: int = 1
  seed: int = 20260710
  backend: Literal["reference", "atom", "amd", "amd_warp", "amd_warp_batched", "amd_dot4_batched", "amd_dot4x4_batched", "direct_packed", "q4k_q8_1_mmq_amd_staged_ds4_atom_v0", "q4k_q8_1_mmq_amd_ds4_warp_atom_v0", "q4k_q8_1_mmq_amd_ds4_dot4x4_atom_v0", "q4k_q8_1_mmq_amd_ds4_lds_skeleton_atom_v0", "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0", "q4k_q8_1_mmq_amd_isa_full_grid_v0", "llama_mmq_q4k_q8_1_coop_tile_oracle"] = "reference"
  activation_layout: Literal["row_major_q8_1", "mmq_ds4"] = ACTIVATION_LAYOUT_ROW_MAJOR
  measure_direct_packed: bool = False
  writeback_mode: Literal["gated_matrix_v0", "direct_owner_v0"] = "gated_matrix_v0"

  @property
  def bounded_m(self) -> int:
    return self.m_tile * self.m_tiles

  @property
  def bounded_n(self) -> int:
    return self.n_tile * self.n_tiles

  @property
  def bounded_k(self) -> int:
    return self.k_groups * Q8_1_BLOCK_ELEMS

  def validate(self) -> None:
    if self.backend not in ("reference", "atom", "amd", "amd_warp", "amd_warp_batched", "amd_dot4_batched", "amd_dot4x4_batched", "direct_packed", STAGED_DS4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_COOP_TILE_BACKEND_ID, FULL_GRID_BACKEND_ID, LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID):
      raise ValueError(f"unknown backend={self.backend!r}")
    if self.activation_layout not in (ACTIVATION_LAYOUT_ROW_MAJOR, ACTIVATION_LAYOUT_MMQ_DS4):
      raise ValueError(f"unknown activation_layout={self.activation_layout!r}")
    if min(self.m_tile, self.n_tile, self.k_groups, self.m_tiles, self.n_tiles) <= 0:
      raise ValueError("tile sizes, tile counts, and k_groups must be positive")
    if self.warmups < 0 or self.rounds < 1: raise ValueError("warmups >= 0 and rounds >= 1 are required")
    if self.bounded_m > M or self.bounded_n > N or self.bounded_k > K:
      raise ValueError(f"bounded shape {(self.bounded_m, self.bounded_n, self.bounded_k)} exceeds role shape {(M, N, K)}")
    if self.bounded_k % Q4_K_BLOCK_ELEMS:
      raise ValueError(f"bounded K={self.bounded_k} must be Q4_K block aligned")
    if self.activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 and self.bounded_k % MMQ_DS4_BLOCK_ELEMS:
      raise ValueError(f"bounded K={self.bounded_k} must be MMQ DS4 block aligned")
    if self.backend == AMD_DS4_DOT4X4_BACKEND_ID and self.bounded_m % 4:
      raise ValueError(f"{AMD_DS4_DOT4X4_BACKEND_ID} requires bounded M to be a multiple of 4")
    if self.writeback_mode not in ("gated_matrix_v0", "direct_owner_v0"):
      raise ValueError(f"unknown writeback_mode={self.writeback_mode!r}")


def candidate_metadata(config: BoundedMMQConfig | None = None) -> dict[str, Any]:
  cfg = config or BoundedMMQConfig()
  cfg.validate()
  activation_layout = ACTIVATION_LAYOUT_MMQ_DS4 if cfg.backend in (STAGED_DS4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_COOP_TILE_BACKEND_ID, FULL_GRID_BACKEND_ID, LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID) else cfg.activation_layout
  return {
    "role": ROLE,
    "M": M,
    "N": N,
    "K": K,
    "quant": QUANT,
    "activation": ACTIVATION,
    "candidate_route_id": CANDIDATE_ROUTE_ID,
    "public_label": PUBLIC_LABEL,
    "primitive_class": PRIMITIVE_CLASS,
    "comparator_id": COMPARATOR_ID,
    "rollback": COMPARATOR_ID,
    "backend": cfg.backend,
    "backend_atom_id": cfg.backend if cfg.backend in (STAGED_DS4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_COOP_TILE_BACKEND_ID, FULL_GRID_BACKEND_ID, LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID) else None,
    "activation_layout": activation_layout,
    "bounded_shape": {"M": cfg.bounded_m, "N": cfg.bounded_n, "K": cfg.bounded_k},
    "tile": {"M": cfg.m_tile, "N": cfg.n_tile, "K_groups": cfg.k_groups},
    "writeback_mode": cfg.writeback_mode if cfg.backend == AMD_DS4_COOP_TILE_BACKEND_ID else None,
  }


def _source_hash() -> str:
  data = pathlib.Path(__file__).read_bytes()
  return hashlib.sha256(data).hexdigest()[:16]


def _atom_source_hash() -> str | None:
  try:
    import extra.qk.mmq_q4k_q8_atom as atom
    path = pathlib.Path(atom.__file__)
  except Exception:
    return None
  return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def _q4k_tile_loader_source_hash() -> str | None:
  try:
    import extra.qk.q4k_tile_loader as tile_loader
    path = pathlib.Path(tile_loader.__file__)
  except Exception:
    return None
  return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def _llama_mmq_oracle_source_hash() -> str | None:
  try:
    import extra.qk.mmq_llama_oracle as oracle
    path = pathlib.Path(oracle.__file__)
  except Exception:
    return None
  return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _run_reference_tile(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  return q4k_q8_1_mmq_tile_reference(q4k_bytes, xq, xscales, spec)


def _run_reference_tile_with_activation(q4k_bytes:np.ndarray, activation:Q8ActivationInputs,
                                        spec:Q4KQ81MMQTileSpec, activation_layout:str) -> np.ndarray:
  if activation_layout == ACTIVATION_LAYOUT_ROW_MAJOR:
    return _run_reference_tile(q4k_bytes, activation.row_values, activation.row_scales, spec)
  if activation_layout == ACTIVATION_LAYOUT_MMQ_DS4:
    if activation.ds4_activation is None or activation.q8_sums is None:
      raise ValueError("mmq_ds4 activation layout requires precomputed activation sums")
    return q4k_q8_1_mmq_ds4_tile_reference(q4k_bytes, activation.ds4_activation, spec)
  raise ValueError(f"unknown activation_layout={activation_layout!r}")


def _run_atom_tile(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  try:
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_tile
  except Exception as exc:
    raise MMQAtomUnavailableError(
      f"{CANDIDATE_ROUTE_ID} selected but extra.qk.mmq_q4k_q8_atom.run_q4k_q8_1_mmq_tile is unavailable"
    ) from exc
  return np.asarray(run_q4k_q8_1_mmq_tile(q4k_bytes, xq, xscales, spec), dtype=np.float32)


def _run_amd_tile(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  try:
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_tile_amd
  except Exception as exc:
    raise MMQAtomUnavailableError(
      f"{CANDIDATE_ROUTE_ID} selected but AMD UOp atom entrypoint is unavailable"
    ) from exc
  return np.asarray(run_q4k_q8_1_mmq_tile_amd(q4k_bytes, xq, xscales, spec).output, dtype=np.float32)


def _run_amd_warp_tile(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  try:
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_tile_amd_warp
  except Exception as exc:
    raise MMQAtomUnavailableError(
      f"{CANDIDATE_ROUTE_ID} selected but AMD warp atom entrypoint is unavailable"
    ) from exc
  return np.asarray(run_q4k_q8_1_mmq_tile_amd_warp(q4k_bytes, xq, xscales, spec).output, dtype=np.float32)


def _run_direct_packed(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, *, role:str=ROLE, device:str="AMD") -> np.ndarray:
  from extra.qk.mmq_q4k_q8_atom import _as_u32_words
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill, emit_q4k_packed_prefill_kernel
  m, k = xq.shape
  n = q4k_bytes.shape[0]
  spec = describe_q4k_packed_prefill(n, k, m, role=role, output_layout="direct_out")
  words = Tensor(_as_u32_words(q4k_bytes), dtype=dtypes.uint32, device=device).realize()
  x = q8_1_dequantize(Tensor(np.ascontiguousarray(xq.reshape(-1)), dtype=dtypes.int8, device=device),
                      Tensor(np.ascontiguousarray(xscales.reshape(-1)), dtype=dtypes.float32, device=device)).realize()
  out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    words, x, fxn=emit_q4k_packed_prefill_kernel(spec))[0].realize()
  return out.numpy().astype(np.float32)


def _run_direct_packed_tile(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  full = _run_direct_packed(q4k_bytes, xq, xscales, role=spec.role)
  return np.asarray(full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n], dtype=np.float32)


def _run_amd_warp_batched(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray) -> np.ndarray:
  try:
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_warp
  except Exception as exc:
    raise MMQAtomUnavailableError(
      f"{CANDIDATE_ROUTE_ID} selected but AMD warp batched atom entrypoint is unavailable"
    ) from exc
  return np.asarray(run_q4k_q8_1_mmq_bounded_amd_warp(q4k_bytes, xq, xscales, role=ROLE).output, dtype=np.float32)


def _run_amd_dot4_batched(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray) -> np.ndarray:
  try:
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_dot4
  except Exception as exc:
    raise MMQAtomUnavailableError(
      f"{CANDIDATE_ROUTE_ID} selected but AMD dot4 batched atom entrypoint is unavailable"
    ) from exc
  return np.asarray(run_q4k_q8_1_mmq_bounded_amd_dot4(q4k_bytes, xq, xscales, role=ROLE).output, dtype=np.float32)


def _run_amd_dot4x4_batched(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray) -> np.ndarray:
  try:
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_dot4x4
  except Exception as exc:
    raise MMQAtomUnavailableError(
      f"{CANDIDATE_ROUTE_ID} selected but AMD dot4x4 batched atom entrypoint is unavailable"
    ) from exc
  return np.asarray(run_q4k_q8_1_mmq_bounded_amd_dot4x4(q4k_bytes, xq, xscales, role=ROLE).output, dtype=np.float32)


def _run_staged_ds4_tile(q4k_bytes:np.ndarray, ds4:Any, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_staged_ds4_atom
  return np.asarray(run_q4k_q8_1_mmq_staged_ds4_atom(q4k_bytes, ds4, spec).output, dtype=np.float32)


def _run_amd_ds4_dot4x4(q4k_bytes:np.ndarray, ds4:Any) -> np.ndarray:
  from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_ds4_dot4x4
  return np.asarray(run_q4k_q8_1_mmq_bounded_amd_ds4_dot4x4(q4k_bytes, ds4, role=ROLE).output, dtype=np.float32)


def _run_amd_ds4_warp(q4k_bytes:np.ndarray, ds4:Any) -> np.ndarray:
  from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_ds4_warp
  return np.asarray(run_q4k_q8_1_mmq_bounded_amd_ds4_warp(q4k_bytes, ds4, role=ROLE).output, dtype=np.float32)


def _run_amd_ds4_lds_skeleton(q4k_bytes:np.ndarray, ds4:Any) -> np.ndarray:
  from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_ds4_lds_skeleton
  return np.asarray(run_q4k_q8_1_mmq_bounded_amd_ds4_lds_skeleton(q4k_bytes, ds4, role=ROLE).output, dtype=np.float32)


def _run_amd_ds4_coop_tile(q4k_bytes:np.ndarray, ds4:Any, writeback_mode:str="gated_matrix_v0") -> np.ndarray:
  from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile
  return np.asarray(run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile(
    q4k_bytes, ds4, role=ROLE, writeback_mode=writeback_mode).output, dtype=np.float32)


def coop_tile_blocked_translation_evidence(config:BoundedMMQConfig | None = None) -> dict[str, Any]:
  cfg = config or BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, backend=AMD_DS4_COOP_TILE_BACKEND_ID)
  return {
    "status": "bounded_numeric_pass",
    "backend": AMD_DS4_COOP_TILE_BACKEND_ID,
    "backend_atom_id": AMD_DS4_COOP_TILE_BACKEND_ID,
    "bounded_only": True,
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "attempted_shapes": [
      {"M": 8, "N": 8, "K": 256},
      {"M": 16, "N": 16, "K": 256},
      {"M": 16, "N": 16, "K": 512},
    ],
    "requested_bounded_shape": {"M": cfg.bounded_m, "N": cfg.bounded_n, "K": cfg.bounded_k},
    "coop_tile_atom_source_hash": _amd_ds4_coop_tile_atom_hash(
      BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, backend=AMD_DS4_COOP_TILE_BACKEND_ID)),
    "exact_blocker": "store_owner metadata is not attached to the emitted numeric graph; R4 owner proof remains separate",
    "blocked_translation_evidence": [
      "R3 q4k_q8_1_mmq_amd_ds4_lds_skeleton_atom_v0 stages DS4 q8 values through LOCAL memory and a barrier.",
      "R3 lifecycle marks shared_memory_staging=True but promotion_eligible=False and production_dispatch_changed=False.",
      "R4 lowered AMD ISA proof covers the 16x16 owner map as eight spill-free 32-store fragments.",
      "R5 bounded coop numeric atom emits and passes 16x16x256 DS4 correctness in run_bounded_harness.",
      "The emitted Tensor custom_kernel omits store_owner metadata; no production dispatch or route promotion is claimed.",
    ],
  }


def _amd_uop_hash(spec:Q4KQ81MMQTileSpec) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_atom_source_hash
    return amd_atom_source_hash(spec)
  except Exception:
    return None


def _amd_warp_uop_hash(spec:Q4KQ81MMQTileSpec) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_warp_atom_source_hash
    return amd_warp_atom_source_hash(spec)
  except Exception:
    return None


def _amd_warp_batched_uop_hash(config:BoundedMMQConfig) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_warp_batched_atom_source_hash
    return amd_warp_batched_atom_source_hash(config.bounded_m, config.bounded_n, config.bounded_k, ROLE)
  except Exception:
    return None


def _amd_dot4_batched_uop_hash(config:BoundedMMQConfig) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_dot4_batched_atom_source_hash
    return amd_dot4_batched_atom_source_hash(config.bounded_m, config.bounded_n, config.bounded_k, ROLE)
  except Exception:
    return None


def _amd_dot4x4_batched_uop_hash(config:BoundedMMQConfig) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_dot4x4_batched_atom_source_hash
    return amd_dot4x4_batched_atom_source_hash(config.bounded_m, config.bounded_n, config.bounded_k, ROLE)
  except Exception:
    return None


def _staged_ds4_atom_hash(spec:Q4KQ81MMQTileSpec) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import staged_ds4_atom_source_hash
    return staged_ds4_atom_source_hash(spec)
  except Exception:
    return None


def _amd_ds4_dot4x4_atom_hash(config:BoundedMMQConfig) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_ds4_dot4x4_atom_source_hash
    return amd_ds4_dot4x4_atom_source_hash(config.bounded_m, config.bounded_n, config.bounded_k, ROLE)
  except Exception:
    return None


def _amd_ds4_warp_atom_hash(config:BoundedMMQConfig) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_ds4_warp_atom_source_hash
    return amd_ds4_warp_atom_source_hash(config.bounded_m, config.bounded_n, config.bounded_k, ROLE)
  except Exception:
    return None


def _amd_ds4_lds_skeleton_atom_hash(config:BoundedMMQConfig) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_ds4_lds_skeleton_atom_source_hash
    return amd_ds4_lds_skeleton_atom_source_hash(config.bounded_m, config.bounded_n, config.bounded_k, ROLE)
  except Exception:
    return None


def _amd_ds4_coop_tile_atom_hash(config:BoundedMMQConfig) -> str | None:
  try:
    from extra.qk.mmq_q4k_q8_atom import amd_ds4_coop_tile_atom_source_hash
    return amd_ds4_coop_tile_atom_source_hash(
      config.bounded_m, config.bounded_n, config.bounded_k, ROLE, config.writeback_mode)
  except Exception:
    return None


def _time_full_output(runner, warmups:int, rounds:int) -> tuple[list[float], np.ndarray]:
  for _ in range(warmups): runner()
  samples_ms: list[float] = []
  last = None
  for _ in range(rounds):
    t0 = time.perf_counter()
    last = runner()
    samples_ms.append((time.perf_counter() - t0) * 1000.0)
  if last is None: raise ValueError("rounds must be >= 1")
  return samples_ms, np.asarray(last, dtype=np.float32)


def _fp32_accum_atol(k:int) -> float:
  return max(5e-4, 3e-6 * k)


def run_bounded_harness(config: BoundedMMQConfig) -> dict[str, Any]:
  config.validate()
  q4k_bytes = make_finite_q4k_bytes(config.bounded_n, config.bounded_k, config.seed)
  ds4_backends = (STAGED_DS4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID,
                  AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_COOP_TILE_BACKEND_ID, LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID)
  effective_activation_layout = ACTIVATION_LAYOUT_MMQ_DS4 if config.backend in ds4_backends else config.activation_layout
  activation = make_q8_activation_inputs(config.bounded_m, config.bounded_k, config.seed + 1, effective_activation_layout)
  xq, xscales = activation.row_values, activation.row_scales
  staged_ds4 = None
  oracle_tiles: list[dict[str, Any]] = []
  if config.backend in ds4_backends:
    if activation.ds4_activation is None:
      raise ValueError("DS4 backend requires canonical mmq_ds4 activation inputs")
    staged_ds4 = activation.ds4_activation
  runner = {"reference": _run_reference_tile, "atom": _run_atom_tile, "amd": _run_amd_tile,
            "amd_warp": _run_amd_warp_tile, "direct_packed": _run_direct_packed_tile}.get(config.backend)

  specs = [
    describe_q4k_q8_1_mmq_tile(role=ROLE, m=config.bounded_m, n=config.bounded_n, k=config.bounded_k,
                               m0=mt * config.m_tile, n0=nt * config.n_tile, m_tile=config.m_tile,
                               n_tile=config.n_tile, k_groups=config.k_groups,
                               activation_layout=Q8_1_MMQ_DS4_LAYOUT if effective_activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 else Q8_1_ROW_MAJOR_LAYOUT)
    for mt in range(config.m_tiles) for nt in range(config.n_tiles)
  ]

  reference_full = np.zeros((config.bounded_m, config.bounded_n), dtype=np.float32)
  for spec in specs:
    reference_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] = _run_reference_tile_with_activation(
      q4k_bytes, activation, spec, effective_activation_layout)
  direct_reference_full = None

  if config.backend in ("amd_warp_batched", "amd_dot4_batched", "amd_dot4x4_batched", "direct_packed"):
    full_runner = {
      "amd_warp_batched": lambda: _run_amd_warp_batched(q4k_bytes, xq, xscales),
      "amd_dot4_batched": lambda: _run_amd_dot4_batched(q4k_bytes, xq, xscales),
      "amd_dot4x4_batched": lambda: _run_amd_dot4x4_batched(q4k_bytes, xq, xscales),
      "direct_packed": lambda: _run_direct_packed(q4k_bytes, xq, xscales),
    }[config.backend]
    samples_ms, last_full = _time_full_output(full_runner, config.warmups, config.rounds)
    last_tiles = [last_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] for spec in specs]
  elif config.backend == STAGED_DS4_BACKEND_ID:
    assert staged_ds4 is not None
    for _ in range(config.warmups):
      for spec in specs: _run_staged_ds4_tile(q4k_bytes, staged_ds4, spec)

    samples_ms = []
    last_tiles = []
    for _ in range(config.rounds):
      t0 = time.perf_counter()
      last_tiles = [_run_staged_ds4_tile(q4k_bytes, staged_ds4, spec) for spec in specs]
      samples_ms.append((time.perf_counter() - t0) * 1000.0)
  elif config.backend == AMD_DS4_DOT4X4_BACKEND_ID:
    assert staged_ds4 is not None
    full_runner = lambda: _run_amd_ds4_dot4x4(q4k_bytes, staged_ds4)
    samples_ms, last_full = _time_full_output(full_runner, config.warmups, config.rounds)
    last_tiles = [last_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] for spec in specs]
  elif config.backend == AMD_DS4_WARP_BACKEND_ID:
    assert staged_ds4 is not None
    full_runner = lambda: _run_amd_ds4_warp(q4k_bytes, staged_ds4)
    samples_ms, last_full = _time_full_output(full_runner, config.warmups, config.rounds)
    last_tiles = [last_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] for spec in specs]
  elif config.backend == AMD_DS4_LDS_SKELETON_BACKEND_ID:
    assert staged_ds4 is not None
    full_runner = lambda: _run_amd_ds4_lds_skeleton(q4k_bytes, staged_ds4)
    samples_ms, last_full = _time_full_output(full_runner, config.warmups, config.rounds)
    last_tiles = [last_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] for spec in specs]
  elif config.backend == AMD_DS4_COOP_TILE_BACKEND_ID:
    assert staged_ds4 is not None
    try:
      full_runner = lambda: _run_amd_ds4_coop_tile(q4k_bytes, staged_ds4, config.writeback_mode)
      samples_ms, last_full = _time_full_output(full_runner, config.warmups, config.rounds)
    except RuntimeError as exc:
      raise MMQAtomUnavailableError(f"{AMD_DS4_COOP_TILE_BACKEND_ID} blocked_numeric_compute: {exc}") from exc
    last_tiles = [last_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] for spec in specs]
  elif config.backend == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID:
    assert staged_ds4 is not None
    for _ in range(config.warmups):
      for spec in specs: run_llama_mmq_coop_tile_oracle(q4k_bytes, staged_ds4, spec)

    samples_ms = []
    last_tiles = []
    for _ in range(config.rounds):
      t0 = time.perf_counter()
      results = [run_llama_mmq_coop_tile_oracle(q4k_bytes, staged_ds4, spec) for spec in specs]
      samples_ms.append((time.perf_counter() - t0) * 1000.0)
      last_tiles = [result.output for result in results]
      oracle_tiles = [result.to_json() for result in results]
  else:
    assert runner is not None
    for _ in range(config.warmups):
      for spec in specs:
        if config.backend == "reference":
          _run_reference_tile_with_activation(q4k_bytes, activation, spec, effective_activation_layout)
        else:
          runner(q4k_bytes, xq, xscales, spec)

    samples_ms = []
    last_tiles = []
    for _ in range(config.rounds):
      t0 = time.perf_counter()
      if config.backend == "reference":
        last_tiles = [_run_reference_tile_with_activation(q4k_bytes, activation, spec, effective_activation_layout) for spec in specs]
      else:
        last_tiles = [runner(q4k_bytes, xq, xscales, spec) for spec in specs]
      samples_ms.append((time.perf_counter() - t0) * 1000.0)

  reference_tiles = [reference_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] for spec in specs]
  max_abs = max(float(np.max(np.abs(got - ref))) for got, ref in zip(last_tiles, reference_tiles)) if last_tiles else 0.0
  atol = _fp32_accum_atol(config.bounded_k) if config.backend in ("amd", "amd_warp", "amd_warp_batched", "amd_dot4_batched", "amd_dot4x4_batched", "direct_packed", STAGED_DS4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_COOP_TILE_BACKEND_ID) else 2e-5
  if config.backend in ds4_backends:
    atol = max(atol, 1e-3)
  ok = max_abs <= atol
  direct_samples_ms, direct_max_abs, direct_status = None, None, "not_requested"
  if config.backend == "direct_packed":
    direct_samples_ms, direct_full = samples_ms, last_full
  elif config.measure_direct_packed:
    direct_samples_ms, direct_full = _time_full_output(lambda: _run_direct_packed(q4k_bytes, xq, xscales), config.warmups, config.rounds)
  if direct_samples_ms is not None:
    if effective_activation_layout == ACTIVATION_LAYOUT_MMQ_DS4:
      direct_reference_full = np.zeros((config.bounded_m, config.bounded_n), dtype=np.float32)
      for spec in specs:
        direct_reference_full[spec.m0:spec.m0+spec.tile_m, spec.n0:spec.n0+spec.tile_n] = _run_reference_tile(
          q4k_bytes, xq, xscales, spec)
    else:
      direct_reference_full = reference_full
    direct_max_abs = float(np.max(np.abs(direct_full - direct_reference_full)))
    direct_status = "PASS" if direct_max_abs <= _fp32_accum_atol(config.bounded_k) else "FAIL"
  if staged_ds4 is not None:
    ds4_json = staged_ds4.to_json()
    ds4_source = {
      STAGED_DS4_BACKEND_ID: "atom_q8_1_mmq_ds4_direct_carrier",
      AMD_DS4_WARP_BACKEND_ID: "amd_ds4_warp_gpu_direct_carrier",
      AMD_DS4_DOT4X4_BACKEND_ID: "amd_ds4_dot4x4_gpu_direct_carrier",
      AMD_DS4_LDS_SKELETON_BACKEND_ID: "amd_ds4_lds_skeleton_gpu_local_carrier",
      AMD_DS4_COOP_TILE_BACKEND_ID: "amd_ds4_coop_tile_gpu_local_owner_carrier",
      LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID: "llama_mmq_coop_tile_oracle_carrier",
    }[config.backend]
    layout_report = {
      "activation_layout": ACTIVATION_LAYOUT_MMQ_DS4,
      "activation_layout_source": ds4_source,
      "q8_values_shape": ds4_json["q8_values_shape"],
      "q8_scales_shape": ds4_json["q8_scales_shape"],
      "q8_sums_shape": ds4_json["q8_sums_shape"],
      "llama_mmq_geometry": LLAMA_MMQ_GEOMETRY,
      "uses_precomputed_activation_sums": True,
    }
  else:
    layout_report = {
      "activation_layout": effective_activation_layout,
      "activation_layout_source": activation.activation_layout_source,
      "q8_values_shape": activation.q8_values_shape,
      "q8_scales_shape": activation.q8_scales_shape,
      "q8_sums_shape": activation.q8_sums_shape,
      "llama_mmq_geometry": LLAMA_MMQ_GEOMETRY,
      "uses_precomputed_activation_sums": effective_activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 and config.backend == "reference",
    }
  report = {
    "schema": "q4k-q8-1-mmq-bounded-harness.v1",
    "metadata": {**candidate_metadata(config), **layout_report},
    **layout_report,
    "status": "PASS" if ok else "FAIL",
    "correctness": {"max_abs": max_abs, "atol": atol, "tiles": len(specs)},
    "timing": {
      "samples_ms": samples_ms,
      "min_ms": min(samples_ms),
      "median_ms": float(np.median(np.asarray(samples_ms, dtype=np.float64))),
      "comparator_id": COMPARATOR_ID,
      "comparator_status": "measured" if direct_samples_ms is not None else "named_not_measured",
      "direct_packed": None if direct_samples_ms is None else {
        "status": direct_status,
        "samples_ms": direct_samples_ms,
        "min_ms": min(direct_samples_ms),
        "median_ms": float(np.median(np.asarray(direct_samples_ms, dtype=np.float64))),
        "max_abs_vs_reference": direct_max_abs,
        "reference_activation_layout": ACTIVATION_LAYOUT_ROW_MAJOR if effective_activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 else effective_activation_layout,
        "atol": _fp32_accum_atol(config.bounded_k),
      },
    },
    "artifacts": {"harness_source_hash": _source_hash(),
                  "q4k_tile_loader_source_hash": _q4k_tile_loader_source_hash(),
                  "llama_mmq_oracle_source_hash": _llama_mmq_oracle_source_hash() if config.backend == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID else None,
                  "llama_mmq_oracle_source_policy": llama_mmq_source_policy() if config.backend == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID else None,
                  "llama_mmq_oracle_tiles": oracle_tiles if config.backend == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID else None,
                  "atom_source_hash": _atom_source_hash() if config.backend in ("atom", "amd", "amd_warp", "amd_warp_batched", "amd_dot4_batched", "amd_dot4x4_batched", STAGED_DS4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_COOP_TILE_BACKEND_ID) else None,
                  "amd_uop_hash": _amd_uop_hash(specs[0]) if config.backend == "amd" and specs else None,
                  "amd_warp_uop_hash": _amd_warp_uop_hash(specs[0]) if config.backend == "amd_warp" and specs else None,
                  "amd_warp_batched_uop_hash": _amd_warp_batched_uop_hash(config) if config.backend == "amd_warp_batched" else None,
                  "amd_dot4_batched_uop_hash": _amd_dot4_batched_uop_hash(config) if config.backend == "amd_dot4_batched" else None,
                  "amd_dot4x4_batched_uop_hash": _amd_dot4x4_batched_uop_hash(config) if config.backend == "amd_dot4x4_batched" else None,
                  "staged_ds4_atom_source_hash": _staged_ds4_atom_hash(specs[0]) if config.backend == STAGED_DS4_BACKEND_ID and specs else None,
                  "amd_ds4_warp_atom_source_hash": _amd_ds4_warp_atom_hash(config) if config.backend == AMD_DS4_WARP_BACKEND_ID else None,
                  "amd_ds4_dot4x4_atom_source_hash": _amd_ds4_dot4x4_atom_hash(config) if config.backend == AMD_DS4_DOT4X4_BACKEND_ID else None,
                  "amd_ds4_lds_skeleton_atom_source_hash": _amd_ds4_lds_skeleton_atom_hash(config) if config.backend == AMD_DS4_LDS_SKELETON_BACKEND_ID else None,
                  "amd_ds4_coop_tile_atom_source_hash": _amd_ds4_coop_tile_atom_hash(config) if config.backend == AMD_DS4_COOP_TILE_BACKEND_ID else None,
                  "emitted_binary_hash": _amd_ds4_coop_tile_atom_hash(config) if config.backend == AMD_DS4_COOP_TILE_BACKEND_ID else None},
    "blockers": (
      ["atom backend is reference-backed; AMD GPU atom body is not implemented"] if config.backend == "atom" else
      [
        "staged DS4 backend is reference-backed; no real AMD staged tile kernel is emitted",
        "cooperative multi-wave shared-memory staging is represented by lifecycle counters only",
        "no production dispatch or route promotion is claimed",
      ] if config.backend == STAGED_DS4_BACKEND_ID else []
    ),
  }
  return report


def bounded_candidate_id(config: BoundedMMQConfig) -> str:
  config.validate()
  return (
    f"{CANDIDATE_ROUTE_ID}."
    f"{config.backend}.m{config.bounded_m}.n{config.bounded_n}.k{config.bounded_k}."
    f"{config.activation_layout}"
  )


def build_bounded_candidate_result(config: BoundedMMQConfig) -> dict[str, Any]:
  """Research-only candidate result tying bounded numerics to proof export shape.

  This deliberately does not promote or bind a production MMQ route. If the
  selected path is oracle-only or lacks an emitted GPU candidate, the status is
  explicit and the exact blocker is carried in the artifact.
  """
  config.validate()
  candidate_id = bounded_candidate_id(config)
  proof_bundle = build_amd_isa_proof_manifest_bundle(candidate_id=candidate_id, kernel_name=config.backend, rows=[])
  shape = {"M": config.bounded_m, "N": config.bounded_n, "K": config.bounded_k}

  try:
    report = run_bounded_harness(config)
  except MMQAtomUnavailableError as exc:
    return {
      "schema": "q4k-q8-1-mmq-bounded-candidate-result.v1",
      "candidate_id": candidate_id,
      "candidate_route_id": CANDIDATE_ROUTE_ID,
      "backend": config.backend,
      "shape": shape,
      "research_only": True,
      "production_dispatch_changed": False,
      "default_route": COMPARATOR_ID,
      "status": "blocked_emitted_candidate_missing",
      "numeric_status": "not_run",
      "oracle_only": False,
      "exact_blocker": str(exc),
      "amd_isa_proof_bundle": proof_bundle,
      "harness_report": None,
    }

  numeric_status = report["status"]
  if config.backend == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID:
    return {
      "schema": "q4k-q8-1-mmq-bounded-candidate-result.v1",
      "candidate_id": candidate_id,
      "candidate_route_id": CANDIDATE_ROUTE_ID,
      "backend": config.backend,
      "shape": shape,
      "research_only": True,
      "production_dispatch_changed": False,
      "default_route": COMPARATOR_ID,
      "status": "oracle_only",
      "numeric_status": numeric_status,
      "oracle_only": True,
      "exact_blocker": "llama MMQ cooperative-tile path is a translated structure/numeric oracle only; no emitted AMD GPU candidate is claimed",
      "amd_isa_proof_bundle": proof_bundle,
      "harness_report": report,
    }

  emitted_binary_hash = report.get("artifacts", {}).get("emitted_binary_hash")
  if emitted_binary_hash is None and config.backend != "reference":
    blocker = "no emitted AMD GPU candidate binary/hash is present for this bounded MMQ backend"
    if report.get("blockers"):
      blocker = "; ".join(report["blockers"])
    return {
      "schema": "q4k-q8-1-mmq-bounded-candidate-result.v1",
      "candidate_id": candidate_id,
      "candidate_route_id": CANDIDATE_ROUTE_ID,
      "backend": config.backend,
      "shape": shape,
      "research_only": True,
      "production_dispatch_changed": False,
      "default_route": COMPARATOR_ID,
      "status": "blocked_emitted_candidate_missing",
      "numeric_status": numeric_status,
      "oracle_only": False,
      "exact_blocker": blocker,
      "amd_isa_proof_bundle": proof_bundle,
      "harness_report": report,
    }

  return {
    "schema": "q4k-q8-1-mmq-bounded-candidate-result.v1",
    "candidate_id": candidate_id,
    "candidate_route_id": CANDIDATE_ROUTE_ID,
    "backend": config.backend,
    "shape": shape,
    "research_only": True,
    "production_dispatch_changed": False,
    "default_route": COMPARATOR_ID,
    "status": "reference_only" if config.backend == "reference" else numeric_status,
    "numeric_status": numeric_status,
    "oracle_only": False,
    "exact_blocker": None,
    "amd_isa_proof_bundle": proof_bundle,
    "harness_report": report,
  }


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Bounded Q4_K/Q8_1 MMQ harness for 14B ffn_gate_up")
  ap.add_argument("--backend", choices=("reference", "atom", "amd", "amd_warp", "amd_warp_batched", "amd_dot4_batched", "amd_dot4x4_batched", "direct_packed", STAGED_DS4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_COOP_TILE_BACKEND_ID, LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID), default="reference")
  ap.add_argument("--activation-layout", choices=(ACTIVATION_LAYOUT_ROW_MAJOR, ACTIVATION_LAYOUT_MMQ_DS4), default=ACTIVATION_LAYOUT_ROW_MAJOR)
  ap.add_argument("--m-tile", type=int, default=16)
  ap.add_argument("--n-tile", type=int, default=16)
  ap.add_argument("--k-groups", type=int, default=8)
  ap.add_argument("--m-tiles", type=int, default=1)
  ap.add_argument("--n-tiles", type=int, default=1)
  ap.add_argument("--warmups", type=int, default=0)
  ap.add_argument("--rounds", type=int, default=1)
  ap.add_argument("--seed", type=int, default=20260710)
  ap.add_argument("--measure-direct-packed", action="store_true")
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  report = run_bounded_harness(BoundedMMQConfig(m_tile=args.m_tile, n_tile=args.n_tile, k_groups=args.k_groups,
                                               m_tiles=args.m_tiles, n_tiles=args.n_tiles, warmups=args.warmups,
                                               rounds=args.rounds, seed=args.seed, backend=args.backend,
                                               activation_layout=args.activation_layout,
                                               measure_direct_packed=args.measure_direct_packed))
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
