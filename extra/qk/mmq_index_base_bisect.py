"""Minimal, host-only bisection for the MMQ vector-pointer-base regression.

This deliberately stops at the Tensor/UOp boundary.  It is useful when a
backend reports ``INDEX(base=dtypes.float.vec(4))``: the four stages can be
enabled independently, so a change in the emitter cannot hide the owner.
"""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad import Tensor, dtypes

from extra.qk.prefill_int8_wmma_spec import _intdot_matmul, _q4k_group_codes_tensor, _q4k_group_params_tensor
from extra.qk.prefill_int8_wmma_spec import Q4KInt8WMMAPrefillSpec


@dataclass(frozen=True)
class MMQBisection:
  name: str
  helper: str
  value: Tensor


def bisect_small_mmq_graph() -> tuple[MMQBisection, ...]:
  """Build Q4, Q8, correction, dot, and join stages independently.

  Shapes are the smallest legal one-group tile (M=4, N=4, K=256).  In
  particular, ``scale/sum`` is not accidentally tested only as part of dot.
  """
  words = Tensor.zeros((4, 1, 36), dtype=dtypes.uint32)
  xq = Tensor.zeros((4, 256), dtype=dtypes.int8)
  scales = Tensor.ones((4, 8), dtype=dtypes.float32)
  ws = Q4KInt8WMMAPrefillSpec(n=4, k=256, m=4)
  q4 = _q4k_group_codes_tensor(words, 0, 0)
  q8 = xq[:, :32]
  raw = _intdot_matmul(q8, q4.transpose()).cast(dtypes.float32)
  qsum = q8.cast(dtypes.int32).sum(axis=1).cast(dtypes.float32)
  d, dmin, sc, mn = _q4k_group_params_tensor(words, 0, 0)
  correction = scales[:, 0].cast(dtypes.float32) * (d * sc.cast(dtypes.float32) - qsum * (dmin * mn.cast(dtypes.float32)))
  joined = raw + correction
  return (MMQBisection("q4_words_decode", "_q4k_group_codes_tensor", q4),
          MMQBisection("q8_activation_slice", "Tensor.__getitem__", q8),
          MMQBisection("scale_sum_indexing", "_group_params_flat + Tensor.__getitem__", correction),
          MMQBisection("intdot_matmul", "_intdot_matmul", raw),
          MMQBisection("concatenation_reshape", "Tensor reshape/cat boundary", joined.reshape(4, 4)))


def offending_helper() -> str:
  """The reusable fix owner: scalarize the scale view before late loads."""
  return "emit_q4k_int8_wmma_prefill_tensor: xscales.reshape -> xsc2[:, group_idx] -> cast -> reshape; materialize/contiguous the scalar scale view before indexing"
