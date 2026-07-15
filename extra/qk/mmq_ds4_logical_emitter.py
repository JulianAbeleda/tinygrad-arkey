"""Logical DS4 packed MMQ lowering.

This is a research lowering: it emits the proven bounded dot4x4 graph only
when the shared candidate explicitly declares the packed DS4 ABI and mapping.
The Q8 producer is kept in the same module so measurements can include its
cost instead of comparing a prepacked fast path with an unpacked route.
"""
from __future__ import annotations

from dataclasses import replace

from tinygrad import Tensor, dtypes

from extra.qk.layout import q8_1_quantize
from extra.qk.mmq_logical_vocabulary import DotOp, MMQCandidate
from extra.qk.mmq_q4k_q8_atom import _q4k_q8_1_bounded_ds4_dot4x4_kernel, packed_ds4_geometry
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec


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
  if candidate.descriptor.abi.get("activation_storage") not in ("ds4", "row_major"):
    raise ValueError("DS4 MMQ candidate must declare activation_storage='ds4' or 'row_major'")
  q4_block_elements, _, _, _, _, q8_packed_elements, q8_group_elements, _, _ = packed_ds4_geometry(candidate.descriptor)
  m, n, k = (_axis_extent(candidate, name) for name in ("m", "n", "k"))
  if k % q4_block_elements or k % q8_packed_elements:
    raise ValueError("DS4 MMQ K must be aligned to Q4_K and Q8 DS4 blocks")
  if m % candidate.mapping.wmma_shape[0]:
    raise ValueError("DS4 MMQ M must cover whole declared output micro-tiles")
  return m, n, k


def packed_ds4_candidate(m: int, n: int, k: int, *, role: str, target: str = "amd_gfx1100") -> MMQCandidate:
  return Q4KQ8MMQPrefillSpec("prefill", "qwen3-14b", role, "Q4_K", "Q8_1", "q4k",
    "tokens_rows", m, n, k, target=target).packed_ds4_logical_candidate()


def packed_row_major_candidate(m: int, n: int, k: int, *, role: str, target: str = "amd_gfx1100") -> MMQCandidate:
  """Research candidate that preserves Q8 row-major storage through the dot atom."""
  candidate = packed_ds4_candidate(m, n, k, role=role, target=target)
  return replace(candidate, descriptor=replace(candidate.descriptor,
    abi={**candidate.descriptor.abi, "activation_storage": "row_major"}))




def pack_q8_1_mmq_ds4(x: Tensor, candidate: MMQCandidate) -> tuple[Tensor, Tensor, Tensor]:
  """Quantize and transpose row-major activations into the declared DS4 ABI."""
  m, _, k = _validate_candidate(candidate)
  if tuple(x.shape) != (m, k): raise ValueError(f"activation shape must be {(m, k)}, got {tuple(x.shape)}")
  q8 = candidate.descriptor.q8
  groups = q8.groups_per_packed_block
  group_elements = q8.block_elements
  packed_elements = q8.packed_block_elements
  x_f32 = x.cast(dtypes.float32)
  qvalues, qscales = q8_1_quantize(x_f32)
  if candidate.descriptor.abi["activation_storage"] == "row_major":
    values = qvalues.contiguous()
    scales = qscales.contiguous()
    weighted = qvalues.reshape(m, k // group_elements, group_elements).cast(dtypes.float32) * \
      qscales.reshape(m, k // group_elements, 1).expand(m, k // group_elements, group_elements)
    sums = weighted.sum(axis=2).reshape(-1).contiguous()
    return values, scales, sums
  values = qvalues.reshape(m, k // packed_elements, groups, group_elements).permute(1, 0, 2, 3).reshape(-1).contiguous()
  scales = qscales.reshape(m, k // packed_elements, groups).permute(1, 0, 2).reshape(-1).contiguous()
  weighted = qvalues.reshape(m, k // packed_elements, groups, group_elements).cast(dtypes.float32) * \
    qscales.reshape(m, k // packed_elements, groups, 1).expand(m, k // packed_elements, groups, group_elements)
  sums = weighted.sum(axis=3).permute(1, 0, 2).reshape(-1).contiguous()
  return values, scales, sums


def emit_q4k_q8_mmq_ds4(words: Tensor, q8_values: Tensor, q8_scales: Tensor,
                         q8_sums: Tensor, candidate: MMQCandidate) -> Tensor:
  """Emit one descriptor-shaped packed DS4 output tensor."""
  m, n, k = _validate_candidate(candidate)
  q4 = candidate.descriptor.q4k
  q8 = candidate.descriptor.q8
  expected_words = n * (k // q4.block_elements) * (q4.metadata_words + q4.packed_words)
  if tuple(words.shape) != (expected_words,): raise ValueError(f"words shape must be {(expected_words,)}, got {tuple(words.shape)}")
  if tuple(q8_values.shape) != (m * k,): raise ValueError(f"q8 values must be flat with {m*k} elements")
  expected_meta = (m * k) // q8.block_elements
  if tuple(q8_scales.shape) != (expected_meta,) or tuple(q8_sums.shape) != (expected_meta,):
    raise ValueError(f"q8 metadata must be flat with {expected_meta} elements")
  if not (words.dtype == dtypes.uint32 and q8_values.dtype == dtypes.int8 and
          q8_scales.dtype == dtypes.float32 and q8_sums.dtype == dtypes.float32):
    raise ValueError("DS4 MMQ operands have unsupported dtypes")
  if not (words.device == q8_values.device == q8_scales.device == q8_sums.device):
    raise ValueError("DS4 MMQ operands must be on the same device")
  role = str(candidate.descriptor.abi.get("role", "logical_ds4"))
  fxn = _q4k_q8_1_bounded_ds4_dot4x4_kernel(m, n, k, role, candidate.mapping, candidate.descriptor)
  return Tensor.empty(m, n, dtype=dtypes.float32, device=words.device).custom_kernel(
    words.contiguous(), q8_values.contiguous(), q8_scales.contiguous(), q8_sums.contiguous(), fxn=fxn)[0]


__all__ = ["emit_q4k_q8_mmq_ds4", "pack_q8_1_mmq_ds4", "packed_ds4_candidate", "packed_row_major_candidate"]
