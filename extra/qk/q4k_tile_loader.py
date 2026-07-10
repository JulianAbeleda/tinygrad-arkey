#!/usr/bin/env python3
"""Bounded Q4_K tile-load reference helpers.

This module is a tinygrad-side spec for loading one 256-wide Q4_K tile. It is
CPU-only reference code and is not wired into production dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS


Q4K_TILE_LOAD_LAYOUT = "ggml_q4_k_256_block"
Q4K_GROUPS_PER_BLOCK = Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS
Q4K_VALUES_PER_GROUP = Q8_1_BLOCK_ELEMS
Q4K_D_OFFSET = 0
Q4K_DMIN_OFFSET = 2
Q4K_SCALE_MIN_OFFSET = 4
Q4K_SCALE_MIN_BYTES = 12
Q4K_QS_OFFSET = 16
Q4K_QS_BYTES = 128


@dataclass(frozen=True)
class Q4KTileLoadSpec:
  n: int
  k: int
  n0: int = 0
  n_tile: int = 1
  k0: int = 0
  quant_format: str = "Q4_K"
  layout: str = Q4K_TILE_LOAD_LAYOUT
  block_elems: int = Q4_K_BLOCK_ELEMS
  block_bytes: int = Q4_K_BLOCK_BYTES
  groups_per_block: int = Q4K_GROUPS_PER_BLOCK
  values_per_group: int = Q4K_VALUES_PER_GROUP
  q_dtype: str = "uint4_unpacked_to_uint8"
  scale_dtype: str = "float32"
  min_dtype: str = "float32"

  @property
  def tile_n(self) -> int:
    return min(self.n_tile, self.n - self.n0)

  @property
  def k_block(self) -> int:
    return self.k0 // self.block_elems

  @property
  def k1(self) -> int:
    return self.k0 + self.block_elems

  def validate(self) -> None:
    if self.quant_format != "Q4_K": raise ValueError(f"quant_format must be Q4_K, got {self.quant_format!r}")
    if self.layout != Q4K_TILE_LOAD_LAYOUT: raise ValueError(f"unsupported layout={self.layout!r}")
    if self.block_elems != Q4_K_BLOCK_ELEMS: raise ValueError(f"block_elems must be {Q4_K_BLOCK_ELEMS}, got {self.block_elems}")
    if self.block_bytes != Q4_K_BLOCK_BYTES: raise ValueError(f"block_bytes must be {Q4_K_BLOCK_BYTES}, got {self.block_bytes}")
    if self.groups_per_block != Q4K_GROUPS_PER_BLOCK:
      raise ValueError(f"groups_per_block must be {Q4K_GROUPS_PER_BLOCK}, got {self.groups_per_block}")
    if self.values_per_group != Q4K_VALUES_PER_GROUP:
      raise ValueError(f"values_per_group must be {Q4K_VALUES_PER_GROUP}, got {self.values_per_group}")
    if self.q_dtype != "uint4_unpacked_to_uint8": raise ValueError(f"q_dtype must be uint4_unpacked_to_uint8, got {self.q_dtype!r}")
    if self.scale_dtype != "float32": raise ValueError(f"scale_dtype must be float32, got {self.scale_dtype!r}")
    if self.min_dtype != "float32": raise ValueError(f"min_dtype must be float32, got {self.min_dtype!r}")
    if self.n <= 0 or self.k <= 0: raise ValueError(f"invalid Q4_K tensor shape n={self.n} k={self.k}")
    if self.k % self.block_elems: raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems {self.block_elems}")
    if self.k0 % self.block_elems: raise ValueError(f"k0={self.k0} must be Q4_K block aligned")
    if self.k1 > self.k: raise ValueError(f"Q4_K tile [{self.k0},{self.k1}) exceeds k={self.k}")
    if self.n_tile <= 0: raise ValueError(f"n_tile must be positive, got {self.n_tile}")
    if not (0 <= self.n0 < self.n): raise ValueError(f"Q4_K tile row origin outside shape: n0={self.n0}")

  def to_json(self) -> dict[str, Any]:
    return {
      "N": self.n, "K": self.k, "n0": self.n0, "n_tile": self.n_tile, "tile_n": self.tile_n,
      "k0": self.k0, "k1": self.k1, "k_block": self.k_block, "quant_format": self.quant_format,
      "layout": self.layout, "block_elems": self.block_elems, "block_bytes": self.block_bytes,
      "groups_per_block": self.groups_per_block, "values_per_group": self.values_per_group,
      "d_offset": Q4K_D_OFFSET, "dmin_offset": Q4K_DMIN_OFFSET,
      "scale_min_offset": Q4K_SCALE_MIN_OFFSET, "scale_min_bytes": Q4K_SCALE_MIN_BYTES,
      "qs_offset": Q4K_QS_OFFSET, "qs_bytes": Q4K_QS_BYTES,
      "q_dtype": self.q_dtype, "scale_dtype": self.scale_dtype, "min_dtype": self.min_dtype,
    }


@dataclass(frozen=True)
class Q4KLoadedTile:
  q: np.ndarray
  scales: np.ndarray
  mins: np.ndarray
  d: np.ndarray
  dmin: np.ndarray
  spec: Q4KTileLoadSpec

  def dequantized(self) -> np.ndarray:
    return (self.q.astype(np.float32) * self.scales[:, :, None] - self.mins[:, :, None]).reshape(self.spec.tile_n, self.spec.block_elems)

  def to_json(self) -> dict[str, Any]:
    return {
      "q4k_tile_load_spec": self.spec.to_json(),
      "q_shape": list(np.asarray(self.q).shape),
      "scales_shape": list(np.asarray(self.scales).shape),
      "mins_shape": list(np.asarray(self.mins).shape),
      "d_shape": list(np.asarray(self.d).shape),
      "dmin_shape": list(np.asarray(self.dmin).shape),
    }


def _as_q4k_blocks(q4k_bytes: np.ndarray, n: int, k: int) -> np.ndarray:
  raw = np.ascontiguousarray(np.asarray(q4k_bytes, dtype=np.uint8).reshape(-1))
  expected = n * (k // Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES
  if raw.size != expected:
    raise ValueError(f"expected {expected} Q4_K bytes for N={n} K={k}, got {raw.size}")
  return raw.reshape(n, k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)


def _unpack_scale_min(scale_min_bytes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  sb = np.asarray(scale_min_bytes, dtype=np.uint8)
  sc_lo = sb[:, 0:4] & np.uint8(63)
  mn_lo = sb[:, 4:8] & np.uint8(63)
  high = sb[:, 8:12]
  sc_hi = (high & np.uint8(0x0f)) | ((sb[:, 0:4] >> np.uint8(6)) << np.uint8(4))
  mn_hi = (high >> np.uint8(4)) | ((sb[:, 4:8] >> np.uint8(6)) << np.uint8(4))
  return np.concatenate([sc_lo, sc_hi], axis=1), np.concatenate([mn_lo, mn_hi], axis=1)


def _unpack_nibbles(qs: np.ndarray) -> np.ndarray:
  packed = np.asarray(qs, dtype=np.uint8).reshape(-1, 4, Q4K_VALUES_PER_GROUP)
  return np.stack([packed & np.uint8(0x0f), packed >> np.uint8(4)], axis=2).reshape(-1, Q4K_GROUPS_PER_BLOCK, Q4K_VALUES_PER_GROUP)


def load_q4k_256_tile(q4k_bytes: np.ndarray, spec: Q4KTileLoadSpec) -> Q4KLoadedTile:
  spec.validate()
  blocks = _as_q4k_blocks(q4k_bytes, spec.n, spec.k)[spec.n0:spec.n0 + spec.tile_n, spec.k_block]
  d = blocks[:, Q4K_D_OFFSET:Q4K_D_OFFSET + 2].reshape(-1, 2).view("<f2").astype(np.float32).reshape(spec.tile_n)
  dmin = blocks[:, Q4K_DMIN_OFFSET:Q4K_DMIN_OFFSET + 2].reshape(-1, 2).view("<f2").astype(np.float32).reshape(spec.tile_n)
  scale_codes, min_codes = _unpack_scale_min(blocks[:, Q4K_SCALE_MIN_OFFSET:Q4K_SCALE_MIN_OFFSET + Q4K_SCALE_MIN_BYTES])
  q = _unpack_nibbles(blocks[:, Q4K_QS_OFFSET:Q4K_QS_OFFSET + Q4K_QS_BYTES]).astype(np.uint8)
  scales = d[:, None] * scale_codes.astype(np.float32)
  mins = dmin[:, None] * min_codes.astype(np.float32)
  return Q4KLoadedTile(q=q, scales=scales, mins=mins, d=d, dmin=dmin, spec=spec)
