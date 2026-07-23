"""Verifier-safe block-major StateHandle construction for the rotating-PV probe.

No backend lowering is selected here.  This only gives the experimental
scheduler a typed, exact LDS ownership map for eight float8 accumulator windows.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import dtypes
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.uop.ops import Ops, UOp


@dataclass(frozen=True)
class RotatingPVStateHandle:
  storage: UOp
  lane: UOp
  generation: int = 0
  total_blocks: int = 8
  lanes_per_block: int = 8

  def block_offset(self, index: int, element: int = 0) -> UOp:
    if not isinstance(index, int) or not 0 <= index < self.total_blocks:
      raise ValueError(f"rotating PV block must be in [0,{self.total_blocks}), got {index!r}")
    if not isinstance(element, int) or not 0 <= element < self.lanes_per_block:
      raise ValueError(f"rotating PV element must be in [0,{self.lanes_per_block}), got {element!r}")
    # Block-major LDS: each of eight blocks owns 32 lanes x 8 floats = 256 floats.
    return UOp.const(self.lane.dtype, index * 256 + element) + self.lane * UOp.const(self.lane.dtype, self.lanes_per_block)

  def validate(self) -> "RotatingPVStateHandle":
    if (self.total_blocks, self.lanes_per_block) != (8, 8):
      raise ValueError("rotating PV state requires eight float8 windows per wave lane")
    if self.storage.op is not Ops.DEFINE_LOCAL or not isinstance(self.storage.dtype, PtrDType) or +       self.storage.dtype.base != dtypes.float or self.storage.dtype.addrspace is not AddrSpace.LOCAL or self.storage.dtype.size != 2048:
      raise ValueError("rotating PV state requires DEFINE_LOCAL fp32[2048]")
    if self.lane.op is not Ops.SPECIAL or self.lane.arg != "lidx0" or len(self.lane.src) != 1 or self.lane.src[0].arg != 32:
      raise ValueError("rotating PV state requires the wave32 lidx0 owner")
    if not isinstance(self.generation, int) or self.generation < 0:
      raise ValueError("rotating PV state generation must be non-negative")
    for block in range(self.total_blocks):
      for element in range(self.lanes_per_block): self.block_offset(block, element)
    return self
