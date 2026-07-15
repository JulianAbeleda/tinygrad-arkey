"""Logical Q6_K MMQ grammar and bounded reference.

This module intentionally stops at the semantic boundary.  It does not emit
UOps, register a route, or build a dequantized weight tensor.  The reference
accumulates one output tile directly from packed Q6 blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from extra.qk.mmq_logical_vocabulary import (
  Axis, BackendCapability, EdgePredicate, Ownership, PhysicalMapping,
  Stage, Staging, Synchronization,
)

Q6K_BLOCK_ELEMENTS = 256
Q6K_SCALE_COUNT = 16
Q6K_DATA_BYTES = 192
Q6K_SCALE_OFFSET = 192
Q6K_D_OFFSET = 208
Q6K_BLOCK_BYTES = 210


@dataclass(frozen=True)
class Q6KDecode:
  block_elements: int = Q6K_BLOCK_ELEMENTS
  scale_count: int = Q6K_SCALE_COUNT
  low_bits: int = 4
  high_bits: int = 2
  zero_point: int = 32
  scale_offset: int = Q6K_SCALE_OFFSET
  d_offset: int = Q6K_D_OFFSET
  byte_layout: str = "ql_128_qh_64_scales_16_d_f16"

  def validate(self) -> None:
    if (self.block_elements, self.scale_count, self.low_bits, self.high_bits,
        self.zero_point) != (256, 16, 4, 2, 32):
      raise ValueError("unsupported Q6_K decode grammar")
    if self.scale_offset != 192 or self.d_offset != 208:
      raise ValueError("unsupported Q6_K field offsets")


@dataclass(frozen=True)
class Q6KMMQDescriptor:
  axes: tuple[Axis, ...]
  decode: Q6KDecode = Q6KDecode()
  staging: Staging = Staging()
  synchronization: Synchronization = Synchronization()
  ownership: Ownership = Ownership()
  edge_predicates: tuple[EdgePredicate, ...] = ()
  abi: dict[str, str] = None  # type: ignore[assignment]
  quant: str = "Q6_K"

  def __post_init__(self) -> None:
    self.decode.validate()
    if self.quant != "Q6_K":
      raise ValueError("Q6 descriptor cannot be used for another quantization")
    if {a.name for a in self.axes} != {"m", "n", "k", "group", "activation_block"}:
      raise ValueError("Q6 descriptor requires the common MMQ axes")
    if not self.edge_predicates:
      raise ValueError("Q6 edge predicates must be explicit")
    if self.abi is None:
      object.__setattr__(self, "abi", {"output_layout": "tokens_rows"})

  def canonical(self) -> dict:
    return {"quant": self.quant, "decode": self.decode.__dict__,
            "axes": [a.__dict__ for a in self.axes],
            "staging": self.staging.__dict__, "abi": self.abi}


def q6k_weight(block: bytes | bytearray | memoryview, group: int, pos: int) -> float:
  """Decode one Q6 value without allocating a dequantized block."""
  if len(block) != Q6K_BLOCK_BYTES:
    raise ValueError(f"Q6_K block must be {Q6K_BLOCK_BYTES} bytes")
  if not 0 <= group < 16 or not 0 <= pos < 16:
    raise IndexError("Q6_K group/position out of range")
  half, pgrp = divmod(group, 8)
  ql_index = half * 64 + (pgrp % 4) * 16 + pos
  qh_index = 128 + half * 32 + (pgrp % 2) * 16 + pos
  ql = (block[ql_index] >> (4 if pgrp >= 4 else 0)) & 0xf
  qh = ((block[qh_index] >> ((pgrp // 2) * 2)) & 0x3) << 4
  q = ql | qh
  scale = int.from_bytes(bytes([block[Q6K_SCALE_OFFSET + group]]), "little", signed=True)
  d = int.from_bytes(block[Q6K_D_OFFSET:Q6K_D_OFFSET + 2], "little")
  import struct
  return struct.unpack("<e", d.to_bytes(2, "little"))[0] * scale * (q - 32)


def q6k_block_dot(block: bytes, activation: Sequence[float]) -> float:
  if len(activation) != Q6K_BLOCK_ELEMENTS:
    raise ValueError("Q6 activation must contain exactly 256 values")
  return sum(q6k_weight(block, g, p) * activation[g * 16 + p]
             for g in range(16) for p in range(16))


__all__ = ["Q6KDecode", "Q6KMMQDescriptor", "q6k_weight", "q6k_block_dot"]
