"""Logical DS4 packed MMQ lowering.

This is a research lowering: it emits the proven bounded dot4x4 graph only
when the shared candidate explicitly declares the packed DS4 ABI and mapping.
The Q8 producer is kept in the same module so measurements can include its
cost instead of comparing a prepacked fast path with an unpacked route.
"""
from __future__ import annotations

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q8_1_quantize
from extra.qk.mmq_logical_vocabulary import DotOp, MMQCandidate
from extra.qk.mmq_q4k_q8_atom import _q4k_q8_1_bounded_ds4_dot4x4_kernel


def _axis_extent(candidate: MMQCandidate, name: str) -> int:
  for axis in candidate.descriptor.axes:
    if axis.name == name:
      if not isinstance(axis.extent, int): raise ValueError(f"candidate axis {name!r} must have an integer extent")
      return axis.extent
  raise ValueError(f"candidate is missing logical axis {name!r}")


def _validate_candidate(candidate: MMQCandidate) -> tuple[int, int, int]:
  if not isinstance(candidate, MMQCandidate):
    raise TypeError("DS4 MMQ lowering requires an MMQCandidate")
  if candidate.mapping.lifecycle != "packed_ds4":
    raise ValueError("DS4 MMQ lowering requires lifecycle='packed_ds4'")
  if candidate.descriptor.operation.name != DotOp.DOT_I8_I8_I32:
    raise ValueError("DS4 MMQ lowering requires the declared i8 dot operation")
  if candidate.descriptor.q8.sum_policy != "supplied" or not candidate.descriptor.q8.sum_operand:
    raise ValueError("DS4 MMQ lowering requires supplied Q8 group sums")
  if candidate.capability.backend != "amd" or candidate.mapping.wave_size not in candidate.capability.wave_sizes:
    raise ValueError("candidate capability does not cover the DS4 mapping")
  m, n, k = (_axis_extent(candidate, name) for name in ("m", "n", "k"))
  if k % Q4_K_BLOCK_ELEMS or k % (Q8_1_BLOCK_ELEMS * 4):
    raise ValueError("DS4 MMQ K must be aligned to Q4_K and Q8 DS4 blocks")
  if m % candidate.mapping.wmma_shape[0]:
    raise ValueError("DS4 MMQ M must cover whole declared output micro-tiles")
  return m, n, k


def pack_q8_1_mmq_ds4(x: Tensor, candidate: MMQCandidate) -> tuple[Tensor, Tensor, Tensor]:
  """Quantize and transpose row-major activations into the declared DS4 ABI."""
  m, _, k = _validate_candidate(candidate)
  if tuple(x.shape) != (m, k): raise ValueError(f"activation shape must be {(m, k)}, got {tuple(x.shape)}")
  x_f32 = x.cast(dtypes.float32)
  qvalues, qscales = q8_1_quantize(x_f32)
  values = qvalues.reshape(m, k // (Q8_1_BLOCK_ELEMS * 4), 4, Q8_1_BLOCK_ELEMS).permute(1, 0, 2, 3).reshape(-1).contiguous()
  scales = qscales.reshape(m, k // (Q8_1_BLOCK_ELEMS * 4), 4).permute(1, 0, 2).reshape(-1).contiguous()
  sums = x_f32.reshape(m, k // (Q8_1_BLOCK_ELEMS * 4), 4, Q8_1_BLOCK_ELEMS).sum(axis=3).permute(1, 0, 2).reshape(-1).contiguous()
  return values, scales, sums


def emit_q4k_q8_mmq_ds4(words: Tensor, q8_values: Tensor, q8_scales: Tensor,
                         q8_sums: Tensor, candidate: MMQCandidate) -> Tensor:
  """Emit one descriptor-shaped packed DS4 output tensor."""
  m, n, k = _validate_candidate(candidate)
  expected_words = n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK
  if tuple(words.shape) != (expected_words,): raise ValueError(f"words shape must be {(expected_words,)}, got {tuple(words.shape)}")
  if tuple(q8_values.shape) != (m * k,): raise ValueError(f"q8 values must be flat with {m*k} elements")
  expected_meta = (m * k) // Q8_1_BLOCK_ELEMS
  if tuple(q8_scales.shape) != (expected_meta,) or tuple(q8_sums.shape) != (expected_meta,):
    raise ValueError(f"q8 metadata must be flat with {expected_meta} elements")
  if not (words.dtype == dtypes.uint32 and q8_values.dtype == dtypes.int8 and
          q8_scales.dtype == dtypes.float32 and q8_sums.dtype == dtypes.float32):
    raise ValueError("DS4 MMQ operands have unsupported dtypes")
  if not (words.device == q8_values.device == q8_scales.device == q8_sums.device):
    raise ValueError("DS4 MMQ operands must be on the same device")
  role = str(candidate.descriptor.abi.get("role", "logical_ds4"))
  fxn = _q4k_q8_1_bounded_ds4_dot4x4_kernel(m, n, k, role, candidate.mapping)
  return Tensor.empty(m, n, dtype=dtypes.float32, device=words.device).custom_kernel(
    words.contiguous(), q8_values.contiguous(), q8_scales.contiguous(), q8_sums.contiguous(), fxn=fxn)[0]


__all__ = ["emit_q4k_q8_mmq_ds4", "pack_q8_1_mmq_ds4"]
