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


Q8_1_MMQ_DS4_LAYOUT = "q8_1_mmq_ds4_transposed_blocks"
Q8_1_ROW_MAJOR_LAYOUT = "q8_1_row_major_mk_scales_per_32"

Q8_1_MMQ_DS4_BLOCK_ELEMS = 4 * Q8_1_BLOCK_ELEMS
Q8_1_MMQ_DS4_GROUPS_PER_BLOCK = 4
Q8_1_MMQ_DS4_VALUES_PER_GROUP = Q8_1_BLOCK_ELEMS


@dataclass(frozen=True)
class Q81ActivationTileSpec:
  m: int
  k: int
  m0: int = 0
  m_tile: int = 16
  k0: int = 0
  k_groups: int | None = None
  activation_format: str = "Q8_1"
  activation_layout: str = Q8_1_ROW_MAJOR_LAYOUT
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
    if self.activation_layout != Q8_1_ROW_MAJOR_LAYOUT:
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
class Q81MMQDS4Activation:
  values: np.ndarray
  scales: np.ndarray
  sums: np.ndarray
  spec: Q81MMQDS4ActivationSpec

  def __iter__(self):
    yield self.values
    yield self.scales
    yield self.sums

  def to_json(self) -> dict[str, Any]:
    return {
      "activation_layout": self.spec.layout,
      "M": self.spec.m,
      "K": self.spec.k,
      "block_elems": self.spec.block_elems,
      "groups_per_block": self.spec.groups_per_block,
      "values_per_group": self.spec.values_per_group,
      "q8_values_shape": list(np.asarray(self.values).shape),
      "q8_scales_shape": list(np.asarray(self.scales).shape),
      "q8_sums_shape": list(np.asarray(self.sums).shape),
    }


@dataclass(frozen=True)
class Q81MMQDS4ActivationSpec:
  m: int
  k: int
  m0: int = 0
  m_tile: int = 16
  k0: int = 0
  k_groups: int | None = None
  activation_format: str = "Q8_1"
  layout: str = Q8_1_MMQ_DS4_LAYOUT
  block_elems: int = Q8_1_MMQ_DS4_BLOCK_ELEMS
  groups_per_block: int = Q8_1_MMQ_DS4_GROUPS_PER_BLOCK
  values_per_group: int = Q8_1_MMQ_DS4_VALUES_PER_GROUP
  value_dtype: str = "int8"
  scale_dtype: str = "float32"
  sum_dtype: str = "float32"

  @property
  def blocks(self) -> int:
    return self.k // self.block_elems

  @property
  def groups(self) -> int:
    return self.k // self.values_per_group

  @property
  def tile_m(self) -> int:
    return min(self.m_tile, self.m - self.m0)

  @property
  def effective_k_groups(self) -> int:
    return self.groups - self.k0 // self.values_per_group if self.k_groups is None else self.k_groups

  @property
  def k1(self) -> int:
    return self.k0 + self.effective_k_groups * self.values_per_group

  def validate(self) -> None:
    if self.activation_format != "Q8_1": raise ValueError(f"activation_format must be Q8_1, got {self.activation_format!r}")
    if self.layout != Q8_1_MMQ_DS4_LAYOUT: raise ValueError(f"unsupported layout={self.layout!r}")
    if self.block_elems != Q8_1_MMQ_DS4_BLOCK_ELEMS:
      raise ValueError(f"block_elems must be {Q8_1_MMQ_DS4_BLOCK_ELEMS}, got {self.block_elems}")
    if self.groups_per_block != Q8_1_MMQ_DS4_GROUPS_PER_BLOCK:
      raise ValueError(f"groups_per_block must be {Q8_1_MMQ_DS4_GROUPS_PER_BLOCK}, got {self.groups_per_block}")
    if self.values_per_group != Q8_1_MMQ_DS4_VALUES_PER_GROUP:
      raise ValueError(f"values_per_group must be {Q8_1_MMQ_DS4_VALUES_PER_GROUP}, got {self.values_per_group}")
    if self.value_dtype != "int8": raise ValueError(f"value_dtype must be int8, got {self.value_dtype!r}")
    if self.scale_dtype not in ("float16", "float32"): raise ValueError(f"scale_dtype must be float16 or float32, got {self.scale_dtype!r}")
    if self.sum_dtype not in ("float16", "float32"): raise ValueError(f"sum_dtype must be float16 or float32, got {self.sum_dtype!r}")
    if self.m <= 0 or self.k <= 0: raise ValueError(f"invalid activation shape m={self.m} k={self.k}")
    if self.k % self.block_elems: raise ValueError(f"k={self.k} must be 128-aligned for Q8_1 MMQ DS4")
    if self.k0 % self.values_per_group: raise ValueError(f"k0={self.k0} must be Q8_1 block aligned")
    if self.effective_k_groups % self.groups_per_block:
      raise ValueError(f"k_groups={self.effective_k_groups} must cover whole 128-value Q8_1 MMQ DS4 blocks")
    if self.m_tile <= 0: raise ValueError(f"m_tile must be positive, got {self.m_tile}")
    if not (0 <= self.m0 < self.m): raise ValueError(f"activation tile origin outside shape: m0={self.m0}")
    if self.effective_k_groups <= 0: raise ValueError(f"k_groups must be positive, got {self.effective_k_groups}")
    if self.k1 > self.k: raise ValueError(f"activation k tile [{self.k0},{self.k1}) exceeds k={self.k}")

  def to_json(self) -> dict[str, Any]:
    return {"M": self.m, "K": self.k, "m0": self.m0, "m_tile": self.m_tile, "tile_m": self.tile_m,
            "k0": self.k0, "k1": self.k1, "k_groups": self.effective_k_groups,
            "activation_format": self.activation_format, "layout": self.layout, "block_elems": self.block_elems,
            "groups_per_block": self.groups_per_block, "values_per_group": self.values_per_group,
            "value_dtype": self.value_dtype, "scale_dtype": self.scale_dtype, "sum_dtype": self.sum_dtype}


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
  activation_layout: str = Q8_1_ROW_MAJOR_LAYOUT
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
    if self.activation_layout == Q8_1_MMQ_DS4_LAYOUT:
      raise ValueError("activation_spec is row-major only; use ds4_activation_spec for DS4")
    return Q81ActivationTileSpec(m=self.m, k=self.k, m0=self.m0, m_tile=self.m_tile, k0=self.k0,
                                 k_groups=self.effective_k_groups, activation_format=self.activation_format,
                                 activation_layout=self.activation_layout, group_elems=self.activation_group_elems)

  @property
  def ds4_activation_spec(self) -> Q81MMQDS4ActivationSpec:
    return Q81MMQDS4ActivationSpec(m=self.m, k=self.k, m0=self.m0, m_tile=self.m_tile, k0=self.k0,
                                   k_groups=self.effective_k_groups, values_per_group=self.activation_group_elems,
                                   layout=self.activation_layout)

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
    if self.activation_layout not in (Q8_1_ROW_MAJOR_LAYOUT, Q8_1_MMQ_DS4_LAYOUT):
      raise ValueError(f"unsupported activation_layout={self.activation_layout!r}")
    if self.output_layout != "row_major_mn_tile": raise ValueError(f"unsupported output_layout={self.output_layout!r}")
    if self.split_policy != "single_k_tile": raise ValueError(f"unsupported split_policy={self.split_policy!r}")
    if self.accumulator_dtype != "float32": raise ValueError(f"accumulator_dtype must be float32, got {self.accumulator_dtype!r}")
    if self.output_dtype != "float32": raise ValueError(f"output_dtype must be float32, got {self.output_dtype!r}")
    if self.activation_layout == Q8_1_MMQ_DS4_LAYOUT:
      self.ds4_activation_spec.validate()
    else:
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
                               activation_layout:str=Q8_1_ROW_MAJOR_LAYOUT,
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


def _q4k_group_metadata(q4k_bytes:np.ndarray, spec:Q4KQ81MMQTileSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  raw = _as_u8_flat(q4k_bytes)
  expected_weight_bytes = spec.n * (spec.k // Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES
  if raw.size != expected_weight_bytes:
    raise ValueError(f"expected {expected_weight_bytes} Q4_K bytes for N={spec.n} K={spec.k}, got {raw.size}")
  blocks = raw.reshape(spec.n, spec.k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)
  d = blocks[:, :, 0:2].reshape(-1, 2).view(np.float16).astype(np.float32).reshape(spec.n, -1)
  dmin = blocks[:, :, 2:4].reshape(-1, 2).view(np.float16).astype(np.float32).reshape(spec.n, -1)
  sb = blocks[:, :, 4:16].astype(np.uint8)
  sc_lo = sb[:, :, 0:4] & np.uint8(63)
  mn_lo = sb[:, :, 4:8] & np.uint8(63)
  high = sb[:, :, 8:12]
  sc_hi = (high & np.uint8(0x0f)) | ((sb[:, :, 0:4] >> np.uint8(6)) << np.uint8(4))
  mn_hi = (high >> np.uint8(4)) | ((sb[:, :, 4:8] >> np.uint8(6)) << np.uint8(4))
  sc = np.concatenate([sc_lo, sc_hi], axis=2).reshape(spec.n, spec.groups).astype(np.float32)
  mn = np.concatenate([mn_lo, mn_hi], axis=2).reshape(spec.n, spec.groups).astype(np.float32)

  qs = blocks[:, :, 16:144].reshape(spec.n, -1, 4, Q8_1_BLOCK_ELEMS)
  codes = np.stack([qs & np.uint8(0x0f), qs >> np.uint8(4)], axis=3).reshape(spec.n, spec.groups, Q8_1_BLOCK_ELEMS)
  d_groups = np.repeat(d[:, :, None], Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS, axis=2).reshape(spec.n, spec.groups)
  dmin_groups = np.repeat(dmin[:, :, None], Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS, axis=2).reshape(spec.n, spec.groups)
  return codes.astype(np.int16), d_groups * sc, dmin_groups * mn


def _validate_ds4(q8_ds4:Q81MMQDS4Activation, spec:Q4KQ81MMQTileSpec) -> Q81MMQDS4Activation:
  if not isinstance(q8_ds4, Q81MMQDS4Activation):
    if isinstance(q8_ds4, tuple) and len(q8_ds4) == 4:
      q8_ds4 = Q81MMQDS4Activation(q8_ds4[0], q8_ds4[1], q8_ds4[2], q8_ds4[3])
    else:
      raise TypeError("q8_ds4 must be Q81MMQDS4Activation or (values, scales, sums, ds4_spec)")
  q8_ds4.spec.validate()
  if q8_ds4.spec.m != spec.m or q8_ds4.spec.k != spec.k:
    raise ValueError(f"DS4 activation shape must be {(spec.m, spec.k)}, got {(q8_ds4.spec.m, q8_ds4.spec.k)}")
  expected_values = (spec.k // 128, spec.m, 128)
  expected_meta = (spec.k // 128, spec.m, 4)
  if np.asarray(q8_ds4.values).shape != expected_values:
    raise ValueError(f"DS4 values shape must be {expected_values}, got {np.asarray(q8_ds4.values).shape}")
  if np.asarray(q8_ds4.scales).shape != expected_meta:
    raise ValueError(f"DS4 scales shape must be {expected_meta}, got {np.asarray(q8_ds4.scales).shape}")
  if np.asarray(q8_ds4.sums).shape != expected_meta:
    raise ValueError(f"DS4 sums shape must be {expected_meta}, got {np.asarray(q8_ds4.sums).shape}")
  return q8_ds4


def q8_1_mmq_ds4_from_row_major_reference(xq:np.ndarray, xscales:np.ndarray, *,
                                          scale_dtype:str="float32", sum_dtype:str="float32") -> Q81MMQDS4Activation:
  xq_arr = np.asarray(xq, dtype=np.int8)
  xs_arr = np.asarray(xscales, dtype=np.float32)
  if xq_arr.ndim != 2: raise ValueError(f"xq must be rank-2 [M,K], got {xq_arr.shape}")
  m, k = xq_arr.shape
  if k % 128: raise ValueError(f"k={k} must be 128-aligned for DS4 MMQ")
  if xs_arr.shape != (m, k // Q8_1_BLOCK_ELEMS): raise ValueError(f"xscales shape must be {(m, k // Q8_1_BLOCK_ELEMS)}, got {xs_arr.shape}")
  values = xq_arr.reshape(m, k // 128, 128).transpose(1, 0, 2).copy()
  scales = xs_arr.reshape(m, k // 128, 4).transpose(1, 0, 2).astype(scale_dtype, copy=True)
  sums = (xq_arr.reshape(m, k // 128, 4, Q8_1_BLOCK_ELEMS).astype(np.float32).sum(axis=3) *
          xs_arr.reshape(m, k // 128, 4)).transpose(1, 0, 2).astype(sum_dtype, copy=True)
  ds4_spec = Q81MMQDS4ActivationSpec(m=m, k=k, scale_dtype=np.dtype(scales.dtype).name, sum_dtype=np.dtype(sums.dtype).name)
  ds4_spec.validate()
  return Q81MMQDS4Activation(values=values, scales=scales, sums=sums, spec=ds4_spec)


def _require_q8_1_mmq_ds4_arrays(values:np.ndarray, scales:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  v = np.asarray(values, dtype=np.int8)
  s = np.asarray(scales, dtype=np.float32)
  if v.ndim != 3 or v.shape[2] != Q8_1_MMQ_DS4_BLOCK_ELEMS:
    raise ValueError(f"values shape must be (K/128, M, 128), got {v.shape}")
  if s.shape != (v.shape[0], v.shape[1], Q8_1_MMQ_DS4_GROUPS_PER_BLOCK):
    raise ValueError(f"scales shape must be {(v.shape[0], v.shape[1], Q8_1_MMQ_DS4_GROUPS_PER_BLOCK)}, got {s.shape}")
  return v, s


def q8_1_mmq_ds4_quantize_reference(x:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Quantize row-major fp32 `[M, K]` into llama-style Q8_1 MMQ DS4 blocks.

  The returned arrays use block-major layout: `values[K/128, M, 128]` and
  `scales/sums[K/128, M, 4]`. Each sum is computed from the original fp32
  values for that 32-value group, before quantization.
  """
  arr = np.asarray(x, dtype=np.float32)
  if arr.ndim != 2: raise ValueError(f"x must be rank-2 [M, K], got shape {arr.shape}")
  m, k = arr.shape
  spec = Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m)
  spec.validate()

  blocks = arr.reshape(m, spec.blocks, spec.groups_per_block, spec.values_per_group)
  amax = np.max(np.abs(blocks), axis=3, keepdims=True)
  scales = np.where(amax == 0.0, 1.0, amax / 127.0).astype(np.float32)
  values = np.rint(blocks / scales).clip(-128, 127).astype(np.int8)
  sums = blocks.sum(axis=3, dtype=np.float32)
  return (np.ascontiguousarray(values.transpose(1, 0, 2, 3).reshape(spec.blocks, m, spec.block_elems)),
          np.ascontiguousarray(scales.reshape(m, spec.blocks, spec.groups_per_block).transpose(1, 0, 2)),
          np.ascontiguousarray(sums.transpose(1, 0, 2)))


def q8_1_mmq_ds4_dequantize_reference(values:np.ndarray, scales:np.ndarray) -> np.ndarray:
  """Dequantize llama-style Q8_1 MMQ DS4 arrays back to row-major `[M, K]`."""
  v, s = _require_q8_1_mmq_ds4_arrays(values, scales)
  k_blocks, m, _ = v.shape
  grouped_values = v.reshape(k_blocks, m, Q8_1_MMQ_DS4_GROUPS_PER_BLOCK, Q8_1_MMQ_DS4_VALUES_PER_GROUP)
  deq = grouped_values.astype(np.float32) * s.reshape(k_blocks, m, Q8_1_MMQ_DS4_GROUPS_PER_BLOCK, 1)
  return np.ascontiguousarray(deq.transpose(1, 0, 2, 3).reshape(m, k_blocks * Q8_1_MMQ_DS4_BLOCK_ELEMS))


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


def q4k_q8_1_mmq_ds4_tile_reference(q4k_bytes:np.ndarray, q8_ds4:Q81MMQDS4Activation,
                                    spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  """Return fp32 `[tile_m, tile_n]` for one Q4_K x Q8_1 DS4 MMQ tile.

  The Q4_K algebra is decomposed as `q8_scale*q4_scale*dot(q4, q8) -
  q4_min*q8_sum`; the min correction consumes the DS4 precomputed sums.
  """
  spec.validate()
  if spec.activation_layout != Q8_1_MMQ_DS4_LAYOUT:
    raise ValueError(f"unsupported activation_layout={spec.activation_layout!r}")
  q8_ds4 = _validate_ds4(q8_ds4, spec)

  q4_codes, q4_scales, q4_mins = _q4k_group_metadata(q4k_bytes, spec)
  out = np.zeros((spec.tile_m, spec.tile_n), dtype=np.float32)
  values = np.asarray(q8_ds4.values, dtype=np.int8)
  scales = np.asarray(q8_ds4.scales, dtype=np.float32)
  sums = np.asarray(q8_ds4.sums, dtype=np.float32)

  g0, g1 = spec.k0 // Q8_1_BLOCK_ELEMS, spec.k1 // Q8_1_BLOCK_ELEMS
  for group_idx in range(g0, g1):
    ds4_block, ds4_group = divmod(group_idx, 4)
    q8_vals = values[ds4_block, spec.m0:spec.m0 + spec.tile_m, :].reshape(spec.tile_m, 4, Q8_1_BLOCK_ELEMS)[:, ds4_group, :].astype(np.int16)
    q8_scale = scales[ds4_block, spec.m0:spec.m0 + spec.tile_m, ds4_group].astype(np.float32)
    q8_sum = sums[ds4_block, spec.m0:spec.m0 + spec.tile_m, ds4_group].astype(np.float32)
    q4_vals = q4_codes[spec.n0:spec.n0 + spec.tile_n, group_idx, :].astype(np.int16)
    dot_term = q8_vals.astype(np.int32) @ q4_vals.astype(np.int32).T
    q4_scale = q4_scales[spec.n0:spec.n0 + spec.tile_n, group_idx].astype(np.float32)
    q4_min = q4_mins[spec.n0:spec.n0 + spec.tile_n, group_idx].astype(np.float32)
    out += q8_scale.reshape(spec.tile_m, 1) * dot_term.astype(np.float32) * q4_scale.reshape(1, spec.tile_n)
    out -= q8_sum.reshape(spec.tile_m, 1) * q4_min.reshape(1, spec.tile_n)
  return out.astype(np.float32)
