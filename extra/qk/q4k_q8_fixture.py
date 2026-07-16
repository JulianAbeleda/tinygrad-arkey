"""Deterministic Q4_K fixtures and Q8_1 operand construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q8_1_quantize
from extra.qk.mmq_q4k_q8_reference import (
  Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q8_1_MMQ_DS4_BLOCK_ELEMS,
  q8_1_mmq_ds4_quantize_reference,
)

ACTIVATION_LAYOUT_ROW_MAJOR = "row_major_q8_1"
ACTIVATION_LAYOUT_MMQ_DS4 = "mmq_ds4"

__all__ = (
  "ACTIVATION_LAYOUT_ROW_MAJOR", "ACTIVATION_LAYOUT_MMQ_DS4", "Q8ActivationInputs",
  "make_finite_q4k_bytes", "make_q8_activation_inputs", "q8_mmq_ds4_from_row_major",
)


@dataclass(frozen=True)
class Q8ActivationInputs:
  source_values: np.ndarray
  row_values: np.ndarray
  row_scales: np.ndarray
  q8_values: np.ndarray
  q8_scales: np.ndarray
  q8_sums: np.ndarray | None
  activation_layout_source: str
  ds4_activation: Q81MMQDS4Activation | None = None

  @property
  def q8_values_shape(self) -> list[int]:
    return list(self.q8_values.shape)

  @property
  def q8_scales_shape(self) -> list[int]:
    return list(self.q8_scales.shape)

  @property
  def q8_sums_shape(self) -> list[int] | None:
    return None if self.q8_sums is None else list(self.q8_sums.shape)


def q8_mmq_ds4_from_row_major(x:np.ndarray, xq:np.ndarray, xscales:np.ndarray) -> Q8ActivationInputs:
  m, k = xq.shape
  if k % Q8_1_MMQ_DS4_BLOCK_ELEMS:
    raise ValueError(f"k={k} must be MMQ DS4 block aligned")
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(x)
  ds4_spec = Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m)
  ds4_activation = Q81MMQDS4Activation(values=values, scales=scales, sums=sums, spec=ds4_spec)
  return Q8ActivationInputs(source_values=x, row_values=xq, row_scales=xscales, q8_values=values, q8_scales=scales,
                            q8_sums=sums, activation_layout_source="l0_l1_q8_1_mmq_ds4_reference_pack",
                            ds4_activation=ds4_activation)


def make_q8_activation_inputs(m:int, k:int, seed:int, activation_layout:str) -> Q8ActivationInputs:
  rng = np.random.default_rng(seed)
  x_np = rng.standard_normal((m, k)).astype(np.float32)
  x = Tensor(x_np).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  row_values = xq.numpy().reshape(m, k)
  row_scales = xscales.numpy().reshape(m, k // Q8_1_BLOCK_ELEMS)
  if activation_layout == ACTIVATION_LAYOUT_ROW_MAJOR:
    return Q8ActivationInputs(source_values=x_np, row_values=row_values, row_scales=row_scales, q8_values=row_values,
                              q8_scales=row_scales, q8_sums=None,
                              activation_layout_source="current_row_major_q8_1_reference_pack")
  if activation_layout == ACTIVATION_LAYOUT_MMQ_DS4:
    return q8_mmq_ds4_from_row_major(x_np, row_values, row_scales)
  raise ValueError(f"unknown activation_layout={activation_layout!r}")


def make_finite_q4k_bytes(n:int, k:int, seed:int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  if k % Q4_K_BLOCK_ELEMS: raise ValueError(f"k={k} must be Q4_K block aligned")
  nblocks = n * k // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=(nblocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
  raw[:, 0:2] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  return raw.reshape(n, k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)
