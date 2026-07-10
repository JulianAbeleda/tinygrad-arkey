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
class Q81ActivationTileSpec:
  m: int
  k: int
  m0: int = 0
  m_tile: int = 16
  k0: int = 0
  k_groups: int | None = None
  activation_format: str = "Q8_1"
  activation_layout: str = "q8_1_row_major_mk_scales_per_32"
  group_elems: int = Q8_1_BLOCK_ELEMS
  value_dtype: str = "int8"
  scale_dtype: str = "float32"

  @property
  def groups(self) -> int:
    return self.k // self.group_elems

  @property
  def tile_m(self) -> int:
    return min(self.m_tile, self.m - self.m0)

  @property
  def effective_k_groups(self) -> int:
    return self.groups - self.k0 // self.group_elems if self.k_groups is None else self.k_groups

  @property
  def k1(self) -> int:
    return self.k0 + self.effective_k_groups * self.group_elems

  def validate(self) -> None:
    if self.activation_format != "Q8_1": raise ValueError(f"activation_format must be Q8_1, got {self.activation_format!r}")
    if self.activation_layout != "q8_1_row_major_mk_scales_per_32":
      raise ValueError(f"unsupported activation_layout={self.activation_layout!r}")
    if self.group_elems != Q8_1_BLOCK_ELEMS: raise ValueError(f"group_elems must be {Q8_1_BLOCK_ELEMS}, got {self.group_elems}")
    if self.value_dtype != "int8": raise ValueError(f"value_dtype must be int8, got {self.value_dtype!r}")
    if self.scale_dtype != "float32": raise ValueError(f"scale_dtype must be float32, got {self.scale_dtype!r}")
    if self.m <= 0 or self.k <= 0: raise ValueError(f"invalid activation shape m={self.m} k={self.k}")
    if self.k % self.group_elems: raise ValueError(f"k={self.k} must be Q8_1 block aligned")
    if self.k0 % self.group_elems: raise ValueError(f"k0={self.k0} must be Q8_1 block aligned")
    if self.m_tile <= 0: raise ValueError(f"m_tile must be positive, got {self.m_tile}")
    if not (0 <= self.m0 < self.m): raise ValueError(f"activation tile origin outside shape: m0={self.m0}")
    if self.effective_k_groups <= 0: raise ValueError(f"k_groups must be positive, got {self.effective_k_groups}")
    if self.k1 > self.k: raise ValueError(f"activation k tile [{self.k0},{self.k1}) exceeds k={self.k}")

  def to_json(self) -> dict[str, Any]:
    return {"M": self.m, "K": self.k, "m0": self.m0, "m_tile": self.m_tile, "tile_m": self.tile_m,
            "k0": self.k0, "k1": self.k1, "k_groups": self.effective_k_groups,
            "activation_format": self.activation_format, "activation_layout": self.activation_layout,
            "group_elems": self.group_elems, "value_dtype": self.value_dtype, "scale_dtype": self.scale_dtype}


@dataclass(frozen=True)
class MMQOutputTileSpec:
  m: int
  n: int
  m0: int = 0
  n0: int = 0
  m_tile: int = 16
  n_tile: int = 16
  output_layout: str = "row_major_mn_tile"
  accumulator_dtype: str = "float32"
  output_dtype: str = "float32"

  @property
  def tile_m(self) -> int:
    return min(self.m_tile, self.m - self.m0)

  @property
  def tile_n(self) -> int:
    return min(self.n_tile, self.n - self.n0)

  def validate(self) -> None:
    if self.m <= 0 or self.n <= 0: raise ValueError(f"invalid output shape m={self.m} n={self.n}")
    if self.m_tile <= 0 or self.n_tile <= 0: raise ValueError(f"tile sizes must be positive, got {self.m_tile}x{self.n_tile}")
    if not (0 <= self.m0 < self.m) or not (0 <= self.n0 < self.n):
      raise ValueError(f"output tile origin outside shape: m0={self.m0} n0={self.n0}")
    if self.output_layout != "row_major_mn_tile": raise ValueError(f"unsupported output_layout={self.output_layout!r}")
    if self.accumulator_dtype != "float32": raise ValueError(f"accumulator_dtype must be float32, got {self.accumulator_dtype!r}")
    if self.output_dtype != "float32": raise ValueError(f"output_dtype must be float32, got {self.output_dtype!r}")

  def to_json(self) -> dict[str, Any]:
    return {"M": self.m, "N": self.n, "m0": self.m0, "n0": self.n0, "m_tile": self.m_tile, "n_tile": self.n_tile,
            "tile_m": self.tile_m, "tile_n": self.tile_n, "output_layout": self.output_layout,
            "accumulator_dtype": self.accumulator_dtype, "output_dtype": self.output_dtype}


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
  accumulator_dtype: str = "float32"
  output_dtype: str = "float32"
  weight_block_elems: int = Q4_K_BLOCK_ELEMS
  activation_group_elems: int = Q8_1_BLOCK_ELEMS
  implementation: str = "numpy_reference_v0"

  @property
  def groups(self) -> int:
    return self.k // self.activation_group_elems

  @property
  def tile_m(self) -> int:
    return min(self.m_tile, self.m - self.m0)

  @property
  def tile_n(self) -> int:
    return min(self.n_tile, self.n - self.n0)

  @property
  def effective_k_groups(self) -> int:
    return self.groups - self.k0 // self.activation_group_elems if self.k_groups is None else self.k_groups

  @property
  def k1(self) -> int:
    return self.k0 + self.effective_k_groups * self.activation_group_elems

  @property
  def activation_spec(self) -> Q81ActivationTileSpec:
    return Q81ActivationTileSpec(m=self.m, k=self.k, m0=self.m0, m_tile=self.m_tile, k0=self.k0,
                                 k_groups=self.effective_k_groups, activation_format=self.activation_format,
                                 activation_layout=self.activation_layout, group_elems=self.activation_group_elems)

  @property
  def output_spec(self) -> MMQOutputTileSpec:
    return MMQOutputTileSpec(m=self.m, n=self.n, m0=self.m0, n0=self.n0, m_tile=self.m_tile, n_tile=self.n_tile,
                             output_layout=self.output_layout, accumulator_dtype=self.accumulator_dtype,
                             output_dtype=self.output_dtype)

  def validate(self) -> None:
    if self.quant_format != "Q4_K": raise ValueError(f"quant_format must be Q4_K, got {self.quant_format!r}")
    if self.activation_format != "Q8_1": raise ValueError(f"activation_format must be Q8_1, got {self.activation_format!r}")
    if self.implementation != "numpy_reference_v0": raise ValueError(f"unsupported implementation={self.implementation!r}")
    if self.weight_block_elems != Q4_K_BLOCK_ELEMS:
      raise ValueError(f"weight_block_elems must be {Q4_K_BLOCK_ELEMS}, got {self.weight_block_elems}")
    if self.activation_group_elems != Q8_1_BLOCK_ELEMS:
      raise ValueError(f"activation_group_elems must be {Q8_1_BLOCK_ELEMS}, got {self.activation_group_elems}")
    if self.k % self.weight_block_elems: raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems {self.weight_block_elems}")
    if self.k0 % self.activation_group_elems: raise ValueError(f"k0={self.k0} must be Q8_1 block aligned")
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
    if self.accumulator_dtype != "float32": raise ValueError(f"accumulator_dtype must be float32, got {self.accumulator_dtype!r}")
    if self.output_dtype != "float32": raise ValueError(f"output_dtype must be float32, got {self.output_dtype!r}")
    self.activation_spec.validate()
    self.output_spec.validate()

  def to_json(self) -> dict[str, Any]:
    return {"role": self.role, "M": self.m, "N": self.n, "K": self.k, "m0": self.m0, "n0": self.n0,
            "m_tile": self.m_tile, "n_tile": self.n_tile, "tile_m": self.tile_m, "tile_n": self.tile_n,
            "k0": self.k0, "k1": self.k1, "k_groups": self.effective_k_groups,
            "quant_format": self.quant_format, "activation_format": self.activation_format,
            "packed_weight_layout": self.packed_weight_layout, "activation_layout": self.activation_layout,
            "output_layout": self.output_layout, "split_policy": self.split_policy,
            "accumulator_dtype": self.accumulator_dtype, "output_dtype": self.output_dtype,
            "weight_block_elems": self.weight_block_elems, "activation_group_elems": self.activation_group_elems,
            "implementation": self.implementation}


def describe_q4k_q8_1_mmq_tile(*, role:str, m:int, n:int, k:int, m0:int=0, n0:int=0,
                               m_tile:int=16, n_tile:int=16, k0:int=0,
                               k_groups:int|None=None, quant_format:str="Q4_K", activation_format:str="Q8_1",
                               packed_weight_layout:str="ggml_q4_k_bytes_row_major_nk",
                               activation_layout:str="q8_1_row_major_mk_scales_per_32",
                               output_layout:str="row_major_mn_tile", split_policy:str="single_k_tile",
                               accumulator_dtype:str="float32", output_dtype:str="float32",
                               weight_block_elems:int=Q4_K_BLOCK_ELEMS,
                               activation_group_elems:int=Q8_1_BLOCK_ELEMS) -> Q4KQ81MMQTileSpec:
  spec = Q4KQ81MMQTileSpec(role=role, m=m, n=n, k=k, m0=m0, n0=n0, m_tile=m_tile, n_tile=n_tile,
                           k0=k0, k_groups=k_groups, quant_format=quant_format,
                           activation_format=activation_format, packed_weight_layout=packed_weight_layout,
                           activation_layout=activation_layout, output_layout=output_layout,
                           split_policy=split_policy, accumulator_dtype=accumulator_dtype,
                           output_dtype=output_dtype, weight_block_elems=weight_block_elems,
                           activation_group_elems=activation_group_elems)
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
