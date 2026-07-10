#!/usr/bin/env python3
"""GPU-free Q4_K x Q8_1 MMQ tile reference.

This module is deliberately a spec/reference skeleton, not a runtime route. It
spells out the tile ownership and computes the tile through the canonical Q4_K
dequant reference plus Q8_1 dequantization, so tests can pin MMQ algebra without
depending on AMD codegen or handwritten kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tinygrad import Tensor

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q4_k_reference


@dataclass(frozen=True)
class Q4KQ81MMQTileSpec:
  role: str
  m: int
  n: int
  k: int
  m0: int = 0
  n0: int = 0
  m_tile: int = 16
  n_tile: int = 16
  k0: int = 0
  k_groups: int | None = None
  quant_format: str = "Q4_K"
  activation_format: str = "Q8_1"
  packed_weight_layout: str = "ggml_q4_k_bytes_row_major_nk"
  activation_layout: str = "q8_1_row_major_mk_scales_per_32"
  output_layout: str = "row_major_mn_tile"
  split_policy: str = "single_k_tile"
  implementation: str = "numpy_reference_v0"

  @property
  def groups(self) -> int:
    return self.k // Q8_1_BLOCK_ELEMS

  @property
  def tile_m(self) -> int:
    return min(self.m_tile, self.m - self.m0)

  @property
  def tile_n(self) -> int:
    return min(self.n_tile, self.n - self.n0)

  @property
  def effective_k_groups(self) -> int:
    return self.groups - self.k0 // Q8_1_BLOCK_ELEMS if self.k_groups is None else self.k_groups

  @property
  def k1(self) -> int:
    return self.k0 + self.effective_k_groups * Q8_1_BLOCK_ELEMS

  def validate(self) -> None:
    if self.quant_format != "Q4_K": raise ValueError(f"quant_format must be Q4_K, got {self.quant_format!r}")
    if self.activation_format != "Q8_1": raise ValueError(f"activation_format must be Q8_1, got {self.activation_format!r}")
    if self.implementation != "numpy_reference_v0": raise ValueError(f"unsupported implementation={self.implementation!r}")
    if self.k % Q4_K_BLOCK_ELEMS: raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems {Q4_K_BLOCK_ELEMS}")
    if self.k0 % Q8_1_BLOCK_ELEMS: raise ValueError(f"k0={self.k0} must be Q8_1 block aligned")
    if self.m <= 0 or self.n <= 0 or self.k <= 0: raise ValueError(f"invalid shape m={self.m} n={self.n} k={self.k}")
    if self.m_tile <= 0 or self.n_tile <= 0: raise ValueError(f"tile sizes must be positive, got {self.m_tile}x{self.n_tile}")
    if not (0 <= self.m0 < self.m) or not (0 <= self.n0 < self.n): raise ValueError(f"tile origin outside shape: m0={self.m0} n0={self.n0}")
    if self.effective_k_groups <= 0: raise ValueError(f"k_groups must be positive, got {self.effective_k_groups}")
    if self.k1 > self.k: raise ValueError(f"k tile [{self.k0},{self.k1}) exceeds k={self.k}")
    if self.packed_weight_layout != "ggml_q4_k_bytes_row_major_nk":
      raise ValueError(f"unsupported packed_weight_layout={self.packed_weight_layout!r}")
    if self.activation_layout != "q8_1_row_major_mk_scales_per_32":
      raise ValueError(f"unsupported activation_layout={self.activation_layout!r}")
    if self.output_layout != "row_major_mn_tile": raise ValueError(f"unsupported output_layout={self.output_layout!r}")
    if self.split_policy != "single_k_tile": raise ValueError(f"unsupported split_policy={self.split_policy!r}")

  def to_json(self) -> dict[str, Any]:
    return {"role": self.role, "M": self.m, "N": self.n, "K": self.k, "m0": self.m0, "n0": self.n0,
            "m_tile": self.m_tile, "n_tile": self.n_tile, "k0": self.k0, "k_groups": self.effective_k_groups,
            "quant_format": self.quant_format, "activation_format": self.activation_format,
            "packed_weight_layout": self.packed_weight_layout, "activation_layout": self.activation_layout,
            "output_layout": self.output_layout, "split_policy": self.split_policy,
            "implementation": self.implementation}


def describe_q4k_q8_1_mmq_tile(*, role:str, m:int, n:int, k:int, m0:int=0, n0:int=0,
                               m_tile:int=16, n_tile:int=16, k0:int=0,
                               k_groups:int|None=None) -> Q4KQ81MMQTileSpec:
  spec = Q4KQ81MMQTileSpec(role=role, m=m, n=n, k=k, m0=m0, n0=n0, m_tile=m_tile, n_tile=n_tile,
                           k0=k0, k_groups=k_groups)
  spec.validate()
  return spec


def _as_u8_flat(x:np.ndarray) -> np.ndarray:
  arr = np.asarray(x, dtype=np.uint8)
  return np.ascontiguousarray(arr.reshape(-1))


def q4k_q8_1_mmq_tile_reference(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray,
                                spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  """Return fp32 `[tile_m, tile_n]` for one Q4_K x Q8_1 MMQ output tile.

  `q4k_bytes` is row-major `[N, K / 256, 144]` or a flat equivalent. `xq` is
  int8 row-major `[M, K]`; `xscales` is fp32-compatible `[M, K / 32]`.
  """
  spec.validate()
  expected_weight_bytes = spec.n * (spec.k // Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES
  raw = _as_u8_flat(q4k_bytes)
  if raw.size != expected_weight_bytes:
    raise ValueError(f"expected {expected_weight_bytes} Q4_K bytes for N={spec.n} K={spec.k}, got {raw.size}")
  xq_arr = np.asarray(xq, dtype=np.int8)
  xs_arr = np.asarray(xscales, dtype=np.float32)
  if xq_arr.shape != (spec.m, spec.k): raise ValueError(f"xq shape must be {(spec.m, spec.k)}, got {xq_arr.shape}")
  if xs_arr.shape != (spec.m, spec.groups): raise ValueError(f"xscales shape must be {(spec.m, spec.groups)}, got {xs_arr.shape}")

  weights = q4_k_reference(Tensor(raw.copy()), spec.n * spec.k).reshape(spec.n, spec.k).numpy().astype(np.float32)
  m_slice = slice(spec.m0, spec.m0 + spec.tile_m)
  n_slice = slice(spec.n0, spec.n0 + spec.tile_n)
  k_slice = slice(spec.k0, spec.k1)
  g_slice = slice(spec.k0 // Q8_1_BLOCK_ELEMS, spec.k1 // Q8_1_BLOCK_ELEMS)
  x_deq = (xq_arr[m_slice, k_slice].reshape(spec.tile_m, spec.effective_k_groups, Q8_1_BLOCK_ELEMS).astype(np.float32) *
           xs_arr[m_slice, g_slice].reshape(spec.tile_m, spec.effective_k_groups, 1)).reshape(spec.tile_m, -1)
  return (x_deq @ weights[n_slice, k_slice].T).astype(np.float32)
