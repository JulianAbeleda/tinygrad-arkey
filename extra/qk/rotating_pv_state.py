"""Verifier-safe block-major StateHandle construction for the rotating-PV probe.

No backend lowering is selected here.  This only gives the experimental
scheduler a typed, exact LDS ownership map for eight float8 accumulator windows.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import dtypes
from tinygrad.uop.ops import PhaseBoundarySpec, StateHandle, StateRegionSpec, UOp


@dataclass(frozen=True)
class RotatingPVStateHandle:
  storage: UOp
  lane: UOp
  generation: int = 0
  total_blocks: int = 8
  lanes_per_block: int = 8

  @property
  def lane_stride(self) -> int: return self.total_blocks * self.lanes_per_block

  def block(self, index: int) -> StateHandle:
    if not isinstance(index, int) or not 0 <= index < self.total_blocks:
      raise ValueError(f"rotating PV block must be in [0,{self.total_blocks}), got {index!r}")
    handle = StateHandle(StateRegionSpec(f"rotating_pv_acc_{index}", dtypes.float, self.lanes_per_block),
      PhaseBoundarySpec("pv_update", "pv_reload", index), self.generation, self.storage, self.lane,
      self.lane_stride, index * self.lanes_per_block)
    return handle.validate()

  def validate(self) -> "RotatingPVStateHandle":
    if (self.total_blocks, self.lanes_per_block, self.lane_stride) != (8, 8, 64):
      raise ValueError("rotating PV state requires eight float8 windows per wave lane")
    for block in range(self.total_blocks): self.block(block)
    return self
