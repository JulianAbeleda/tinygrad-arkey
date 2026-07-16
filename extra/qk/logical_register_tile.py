"""Research vocabulary for logical register-resident operand tiles."""
from __future__ import annotations
from dataclasses import dataclass
from math import prod
from tinygrad.dtype import DType

@dataclass(frozen=True)
class LogicalRegisterTile:
  role:str; dtype:DType; tile_shape:tuple[int, ...]; fragments:int; lane_width:int; carrier_width:int
  slot_count:int; slot_addressing:str; layout:str; alignment_bytes:int|None=None
  ownership:tuple[str, ...] = ("producer", "consumer")
  lifetime:tuple[str, ...] = ("produce", "consume", "release")

  def __post_init__(self) -> None:
    if not isinstance(self.role, str) or not self.role.strip(): raise ValueError("logical register tile role must be non-empty")
    if not isinstance(self.dtype, DType) or self.dtype.count != 1 or self.dtype.name == "void":
      raise ValueError("logical register tile dtype must be a scalar non-void dtype")
    if not isinstance(self.tile_shape, tuple) or not self.tile_shape or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.tile_shape):
      raise ValueError("logical register tile shape must contain positive integers")
    for name in ("fragments", "lane_width", "carrier_width", "slot_count"):
      if not isinstance((value := getattr(self, name)), int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"logical register tile {name} must be a positive int")
    if self.slot_addressing not in ("static", "sequential", "proven"):
      raise ValueError("logical register tile slot addressing must be static, sequential, or proven")
    if not isinstance(self.layout, str) or not self.layout.strip(): raise ValueError("logical register tile layout identity must be non-empty")
    alignment = self.carrier_width * self.dtype.itemsize if self.alignment_bytes is None else self.alignment_bytes
    if self.alignment_bytes is None: object.__setattr__(self, "alignment_bytes", alignment)
    if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0 or alignment % self.dtype.itemsize:
      raise ValueError("logical register tile alignment must be a positive scalar-size multiple")
    if not self.ownership or any(not isinstance(x, str) or not x.strip() for x in self.ownership):
      raise ValueError("logical register tile ownership labels must be non-empty")
    if len(self.lifetime) < 2 or any(not isinstance(x, str) or not x.strip() for x in self.lifetime):
      raise ValueError("logical register tile lifetime requires producer and consumer labels")

  @property
  def scalar_bytes(self) -> int: return self.dtype.itemsize
  @property
  def tile_elements(self) -> int: return prod(self.tile_shape)
  @property
  def fragment_elements(self) -> int: return self.fragments * self.carrier_width
  @property
  def logical_bytes(self) -> int: return self.tile_elements * self.scalar_bytes * self.slot_count

  def snapshot(self) -> dict[str, object]:
    return {"role": self.role, "dtype": self.dtype.name, "tile_shape": self.tile_shape,
      "fragments": self.fragments, "lane_width": self.lane_width, "carrier_width": self.carrier_width,
      "slot_count": self.slot_count, "slot_addressing": self.slot_addressing, "layout": self.layout,
      "alignment_bytes": self.alignment_bytes, "ownership": self.ownership, "lifetime": self.lifetime,
      "tile_elements": self.tile_elements, "logical_bytes": self.logical_bytes}

__all__ = ["LogicalRegisterTile"]
