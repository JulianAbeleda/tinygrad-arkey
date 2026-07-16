"""CPU-only deterministic inputs for the Q4_K/Q8_1 five-buffer MMQ ABI."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from extra.qk.layout import (Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_MMQ_BLOCK_ELEMS,
                             Q8_1_MMQ_GROUPS_PER_BLOCK)
from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference
from extra.qk.q4k_q8_fixture import make_finite_q4k_bytes, q4k_dequantize_selected_positions

BUFFER_NAMES = ("q4_packed_words", "q8_ds4_values", "q8_scales", "q8_weighted_sums", "reference")
SCHEMA = "tinygrad.q4k_q8_five_buffer_input.v1"


def _identity(value:np.ndarray) -> str:
  arr = np.ascontiguousarray(value)
  header = f"{arr.dtype.str}:{','.join(map(str, arr.shape))}:".encode()
  return hashlib.sha256(header + arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class Q4KQ8FiveBufferArtifact:
  q4_packed_words: np.ndarray
  q8_ds4_values: np.ndarray
  q8_scales: np.ndarray
  q8_weighted_sums: np.ndarray
  reference: np.ndarray
  metadata: dict[str, Any]

  def arrays(self) -> dict[str, np.ndarray]:
    return {name: getattr(self, name) for name in BUFFER_NAMES}


def _expected(m:int, n:int, k:int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
  return {
    "q4_packed_words": (np.dtype(np.uint32), (n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,)),
    "q8_ds4_values": (np.dtype(np.int8), ((k // Q8_1_MMQ_BLOCK_ELEMS) * m * Q8_1_MMQ_BLOCK_ELEMS,)),
    "q8_scales": (np.dtype(np.float32), ((k // Q8_1_MMQ_BLOCK_ELEMS) * m * Q8_1_MMQ_GROUPS_PER_BLOCK,)),
    "q8_weighted_sums": (np.dtype(np.float32), ((k // Q8_1_MMQ_BLOCK_ELEMS) * m * Q8_1_MMQ_GROUPS_PER_BLOCK,)),
    "reference": (np.dtype(np.float32), (m, n)),
  }


def validate_q4k_q8_five_buffer_artifact(artifact:Q4KQ8FiveBufferArtifact) -> Q4KQ8FiveBufferArtifact:
  if not isinstance(artifact, Q4KQ8FiveBufferArtifact): raise TypeError("artifact must be Q4KQ8FiveBufferArtifact")
  shape = artifact.metadata.get("shape", {})
  try: m, n, k = (shape[x] for x in ("M", "N", "K"))
  except (KeyError, TypeError): raise ValueError("metadata shape must contain M, N, and K") from None
  if any(type(x) is not int or x <= 0 for x in (m, n, k)) or m % 16 or n % 16 or k % 256:
    raise ValueError("M/N/K must be positive and aligned to 16/16/256")
  if artifact.metadata.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  expected = _expected(m, n, k)
  for name, (dtype, arr_shape) in expected.items():
    value = getattr(artifact, name)
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != arr_shape or not value.flags.c_contiguous:
      raise ValueError(f"{name} must be contiguous {dtype.name}{arr_shape}")
    row = artifact.metadata.get("buffers", {}).get(name, {})
    if row != {"dtype": dtype.name, "shape": list(arr_shape), "nbytes": value.nbytes,
               "sha256": _identity(value)}:
      raise ValueError(f"{name} metadata or content hash mismatch")
  return artifact


def build_q4k_q8_five_buffer_artifact(m:int, n:int, k:int, *, seed:int=0) -> Q4KQ8FiveBufferArtifact:
  if any(type(x) is not int or x <= 0 for x in (m, n, k)) or m % 16 or n % 16 or k % 256:
    raise ValueError("M/N/K must be positive and aligned to 16/16/256")
  q4_bytes = np.ascontiguousarray(make_finite_q4k_bytes(n, k, seed))
  # Power-of-two block multipliers keep both canonical fp32 association and
  # selected-position dequantization exact (the remaining metadata stays seeded).
  q4_bytes[:, :, 0:2] = np.array([1 / 32], dtype=np.float16).view(np.uint8)
  q4_bytes[:, :, 2:4] = np.array([1 / 64], dtype=np.float16).view(np.uint8)
  # One exact, nonzero fp32 value owns each row's selected Q8_1 group.
  positions = ((np.arange(m, dtype=np.int64) * 131 + seed) % k).astype(np.int64)
  coefficients = np.where(np.arange(m) & 1, np.float32(-127.0), np.float32(127.0)).astype(np.float32)
  source = np.zeros((m, k), dtype=np.float32)
  source[np.arange(m), positions] = coefficients
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  selected_q4 = q4k_dequantize_selected_positions(q4_bytes, positions)
  reference = (selected_q4.T * coefficients[:, None]).astype(np.float32)
  arrays = {
    "q4_packed_words": q4_bytes.reshape(-1).copy().view(np.uint32),
    "q8_ds4_values": values.reshape(-1), "q8_scales": scales.reshape(-1),
    "q8_weighted_sums": sums.reshape(-1), "reference": reference,
  }
  arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
  metadata = {"schema": SCHEMA, "shape": {"M": m, "N": n, "K": k},
              "selected_positions": positions.tolist(), "coefficients_fp32": coefficients.tolist(),
              "buffers": {name: {"dtype": value.dtype.name, "shape": list(value.shape), "nbytes": value.nbytes,
                                     "sha256": _identity(value)} for name, value in arrays.items()}}
  return validate_q4k_q8_five_buffer_artifact(Q4KQ8FiveBufferArtifact(**arrays, metadata=metadata))


def save_q4k_q8_five_buffer_artifact(path:str|Path, artifact:Q4KQ8FiveBufferArtifact) -> dict[str, Any]:
  artifact = validate_q4k_q8_five_buffer_artifact(artifact)
  np.savez(Path(path), **artifact.arrays())
  return artifact.metadata


__all__ = ["BUFFER_NAMES", "SCHEMA", "Q4KQ8FiveBufferArtifact", "build_q4k_q8_five_buffer_artifact",
           "save_q4k_q8_five_buffer_artifact", "validate_q4k_q8_five_buffer_artifact"]
