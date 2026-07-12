"""Backend-neutral logical register lease contracts.

These contracts describe ownership and reservation metadata only.  They do not
allocate physical registers or change renderer output.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

from tinygrad.dtype import DType


class RegisterRole(str, Enum):
  KERNARG = "kernarg"
  WORKGROUP_ID = "workgroup_id"
  WORKITEM_ID = "workitem_id"
  POINTER = "pointer"
  SCALAR_COUNTER = "scalar_counter"
  SCALAR_TEMP = "scalar_temp"
  ACCUMULATOR = "accumulator"
  FRAGMENT = "fragment"
  LDS_PACK = "lds_pack"
  VIRTUAL = "virtual"


@dataclass(frozen=True)
class LogicalRegisterTile:
  """Backend-neutral description of one register-resident GEMM operand tile.

  This is intentionally a logical contract.  It describes ownership and
  packing requirements in elements; physical register banks, spans, and lane
  remaps remain backend/consumer responsibilities.  ``slot_addressing`` is
  deliberately fail-closed: a dynamic slot is not a promise that a backend
  can indirectly address registers.
  """
  role: str
  dtype: DType
  tile_shape: tuple[int, ...]
  fragments: int
  lane_width: int
  carrier_width: int
  slot_count: int
  slot_addressing: str
  layout: str
  alignment_bytes: int | None = None
  ownership: tuple[str, ...] = ("producer", "consumer")
  lifetime: tuple[str, ...] = ("produce", "consume", "release")

  def __post_init__(self) -> None:
    if not isinstance(self.role, str) or not self.role.strip():
      raise ValueError("logical register tile role must be non-empty")
    if not isinstance(self.dtype, DType) or self.dtype.count != 1 or self.dtype.name == "void":
      raise ValueError("logical register tile dtype must be a scalar non-void dtype")
    if not isinstance(self.tile_shape, tuple) or not self.tile_shape or any(
        not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.tile_shape):
      raise ValueError("logical register tile shape must contain positive integers")
    for name, value in (("fragments", self.fragments), ("lane_width", self.lane_width),
                        ("carrier_width", self.carrier_width), ("slot_count", self.slot_count)):
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"logical register tile {name} must be a positive int")
    if self.slot_addressing not in ("static", "sequential", "proven"):
      raise ValueError("logical register tile slot addressing must be static, sequential, or proven")
    if not isinstance(self.layout, str) or not self.layout.strip():
      raise ValueError("logical register tile layout identity must be non-empty")
    alignment = self.alignment_bytes
    if alignment is None:
      alignment = self.carrier_width * self.dtype.itemsize
      object.__setattr__(self, "alignment_bytes", alignment)
    if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0 or alignment % self.dtype.itemsize:
      raise ValueError("logical register tile alignment must be a positive scalar-size multiple")
    if not isinstance(self.ownership, tuple) or not self.ownership or any(not isinstance(x, str) or not x.strip() for x in self.ownership):
      raise ValueError("logical register tile ownership labels must be non-empty")
    if not isinstance(self.lifetime, tuple) or len(self.lifetime) < 2 or any(not isinstance(x, str) or not x.strip() for x in self.lifetime):
      raise ValueError("logical register tile lifetime requires producer and consumer labels")

  @property
  def scalar_bytes(self) -> int:
    return self.dtype.itemsize

  @property
  def tile_elements(self) -> int:
    return prod(self.tile_shape)

  @property
  def fragment_elements(self) -> int:
    return self.fragments * self.carrier_width

  @property
  def logical_bytes(self) -> int:
    return self.tile_elements * self.scalar_bytes * self.slot_count

  def snapshot(self) -> dict[str, object]:
    """Stable identity payload for evidence and compatibility tests."""
    return {
      "role": self.role, "dtype": self.dtype.name, "tile_shape": self.tile_shape,
      "fragments": self.fragments, "lane_width": self.lane_width,
      "carrier_width": self.carrier_width, "slot_count": self.slot_count,
      "slot_addressing": self.slot_addressing, "layout": self.layout,
      "alignment_bytes": self.alignment_bytes, "ownership": self.ownership,
      "lifetime": self.lifetime, "tile_elements": self.tile_elements,
      "logical_bytes": self.logical_bytes,
    }


class RegisterBank(str, Enum):
  SGPR = "sgpr"
  VGPR = "vgpr"


@dataclass(frozen=True)
class Lease:
  """A logical contiguous register lease within one bank and mode."""
  role: RegisterRole
  bank: RegisterBank
  start: int
  width: int
  mode: str = "default"
  alignment: int = 1
  reserved: bool = True

  def __post_init__(self) -> None:
    if not isinstance(self.role, RegisterRole): raise TypeError("lease role must be RegisterRole")
    if not isinstance(self.bank, RegisterBank): raise TypeError("lease bank must be RegisterBank")
    for name, value in (("start", self.start), ("width", self.width), ("alignment", self.alignment)):
      if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"lease {name} must be a non-negative int")
    if self.width <= 0: raise ValueError("lease width must be positive")
    if self.alignment <= 0 or self.start % self.alignment: raise ValueError("lease start violates alignment")
    if not isinstance(self.mode, str) or not self.mode: raise ValueError("lease mode must be non-empty")
    if not isinstance(self.reserved, bool): raise TypeError("lease reserved must be bool")

  @property
  def end(self) -> int:
    return self.start + self.width

  def snapshot(self) -> dict[str, object]:
    return {"role": self.role.value, "bank": self.bank.value, "start": self.start, "end": self.end,
            "width": self.width, "mode": self.mode, "alignment": self.alignment, "reserved": self.reserved}


# Descriptive alias for callers that prefer an explicit type name.
RegisterLease = Lease


@dataclass(frozen=True)
class RegisterDescriptor:
  """Register-bank capacity and mode-scoped logical reservations."""
  target: str
  wave_size: int
  vgpr_count: int
  sgpr_count: int
  leases: tuple[Lease, ...]

  def __post_init__(self) -> None:
    if not isinstance(self.target, str) or not self.target: raise ValueError("register target must be non-empty")
    if self.wave_size not in (32, 64): raise ValueError("wave_size must be 32 or 64")
    for name, value in (("vgpr_count", self.vgpr_count), ("sgpr_count", self.sgpr_count)):
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} must be positive")
    object.__setattr__(self, "leases", tuple(self.leases))
    for lease in self.leases:
      limit = self.vgpr_count if lease.bank is RegisterBank.VGPR else self.sgpr_count
      if lease.end > limit: raise ValueError(f"{lease.bank.value} lease exceeds {limit}-register bank")
    self.validate_overlaps()

  def validate_overlaps(self) -> None:
    """Reject same-mode overlap; different modes are explicit alternatives."""
    for index, left in enumerate(self.leases):
      for right in self.leases[index + 1:]:
        if left.bank is not right.bank or left.mode != right.mode: continue
        if left.start < right.end and right.start < left.end:
          raise ValueError(f"overlapping {left.bank.value} leases in mode {left.mode!r}")

  def lease(self, role: RegisterRole, *, mode: str = "default") -> Lease:
    matches = tuple(x for x in self.leases if x.role is role and x.mode == mode)
    if len(matches) != 1: raise KeyError(f"no unique lease for {role.value} in mode {mode!r}")
    return matches[0]

  def snapshot(self) -> dict[str, object]:
    return {"target": self.target, "wave_size": self.wave_size, "vgpr_count": self.vgpr_count,
            "sgpr_count": self.sgpr_count, "leases": [x.snapshot() for x in self.leases]}
