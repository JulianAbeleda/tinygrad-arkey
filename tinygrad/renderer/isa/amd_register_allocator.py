"""AMD virtual register lease allocator backed by the gfx1100 contract."""
from __future__ import annotations

from tinygrad.codegen.opt.register_contracts import Lease, RegisterBank, RegisterRole
from tinygrad.renderer.isa.amd_register_contracts import GFX1100_REGISTER_DESCRIPTOR


_ROLE_BY_NAME = {
  "abi": RegisterRole.KERNARG, "buffer_a": RegisterRole.POINTER, "buffer_b": RegisterRole.POINTER,
  "output": RegisterRole.POINTER, "workgroup_coords": RegisterRole.WORKGROUP_ID,
  "loop_counter": RegisterRole.SCALAR_COUNTER, "fixed_lane_and_address": RegisterRole.WORKITEM_ID,
  "wmma_fragment_a": RegisterRole.FRAGMENT, "wmma_fragment_b": RegisterRole.FRAGMENT,
  "wmma_accumulator": RegisterRole.ACCUMULATOR, "lds_pack_a": RegisterRole.LDS_PACK,
  "lds_pack_b": RegisterRole.LDS_PACK, "address_scratch": RegisterRole.SCALAR_TEMP,
}


class AMDStageBufferSpec:
  """Logical contract for one alternating register-resident stage buffer.

  The compiler describes half-element ownership; the ISA backend decides how
  those elements are packed into physical VGPRs. Keeping this contract in the
  allocator module prevents route code from inventing a combined A+B width.
  """
  __slots__ = ("role", "slots", "fragments", "lane_width")

  def __init__(self, role: str, slots: int, fragments: int, lane_width: int = 16):
    if role not in ("A", "B"): raise ValueError("stage-buffer role must be A or B")
    for name, value in (("slots", slots), ("fragments", fragments), ("lane_width", lane_width)):
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"stage-buffer {name} must be a positive int")
    if lane_width != 16: raise ValueError("gfx1100 WMMA stage buffers require half.vec(16) lanes")
    self.role, self.slots, self.fragments, self.lane_width = role, slots, fragments, lane_width

  @property
  def role_width(self) -> int: return self.fragments * self.lane_width
  @property
  def half_elements(self) -> int: return self.slots * self.role_width
  @property
  def fragment_count(self) -> int: return self.slots * self.fragments
  @property
  def packed_vgpr_width(self) -> int:
    """Physical VGPR span when each half.vec(16) uses eight packed VGPRs."""
    return self.fragment_count * (self.lane_width // 2)
  @property
  def half_bytes(self) -> int: return self.half_elements * 2
  def snapshot(self) -> dict[str, int | str]:
    return {"role": self.role, "slots": self.slots, "fragments": self.fragments,
            "lane_width": self.lane_width, "role_width": self.role_width,
            "half_elements": self.half_elements, "half_bytes": self.half_bytes,
            "packed_vgpr_width": self.packed_vgpr_width}


class AMDRegisterLeaseAllocator:
  """Reserve non-overlapping virtual SGPR/VGPR windows.

  Capacities are inherited from the authoritative gfx1100 descriptor rather
  than duplicated in a route module.  Lease ``mode`` is kept as one allocator
  namespace so alternatives cannot accidentally overlap in an emitted layout.
  """
  def __init__(self, *, vgpr_capacity: int | None = None, sgpr_capacity: int | None = None):
    self.vgpr_capacity = GFX1100_REGISTER_DESCRIPTOR.vgpr_count if vgpr_capacity is None else vgpr_capacity
    self.sgpr_capacity = GFX1100_REGISTER_DESCRIPTOR.sgpr_count if sgpr_capacity is None else sgpr_capacity
    self._leases: list[Lease] = []

  @property
  def leases(self) -> tuple[Lease, ...]: return tuple(self._leases)
  @property
  def virtual_vgpr_pool(self) -> int: return max((x.end for x in self._leases if x.bank is RegisterBank.VGPR), default=0)
  @property
  def virtual_sgpr_pool(self) -> int: return max((x.end for x in self._leases if x.bank is RegisterBank.SGPR), default=0)

  def reserve(self, name: str, start: int, count: int, *, bank: str | RegisterBank, align: int = 1) -> Lease:
    bank = RegisterBank(bank)
    role = _ROLE_BY_NAME.get(name, RegisterRole.VIRTUAL)
    if start < 0 or count <= 0: raise ValueError("invalid AMD register lease")
    capacity = self.vgpr_capacity if bank is RegisterBank.VGPR else self.sgpr_capacity
    if start + count > capacity: raise ValueError(f"{bank.value} lease exceeds virtual pool")
    if any(x.bank is bank and start < x.end and x.start < start + count for x in self._leases):
      raise ValueError(f"{bank.value} lease overlaps an existing reservation")
    lease = Lease(role, bank, start, count, mode="allocator", alignment=align)
    self._leases.append(lease)
    return lease

  def allocate(self, name: str, count: int, *, bank: str | RegisterBank, align: int = 1) -> Lease:
    bank = RegisterBank(bank)
    if not isinstance(align, int) or align <= 0: raise ValueError("lease alignment must be positive")
    capacity = self.vgpr_capacity if bank is RegisterBank.VGPR else self.sgpr_capacity
    cursor = 0
    for lease in sorted((x for x in self._leases if x.bank is bank), key=lambda x: x.start):
      cursor = (cursor + align - 1) // align * align
      if cursor + count <= lease.start: return self.reserve(name, cursor, count, bank=bank, align=align)
      cursor = max(cursor, lease.end)
    cursor = (cursor + align - 1) // align * align
    if cursor + count > capacity: raise ValueError(f"{bank.value} virtual pool exhausted")
    return self.reserve(name, cursor, count, bank=bank, align=align)

  def allocate_in_window(self, name: str, count: int, *, bank: str | RegisterBank,
                         start: int, end: int, align: int = 1) -> Lease:
    """First-fit a lease inside ``[start, end)`` and fail closed on pressure."""
    bank = RegisterBank(bank)
    if not (0 <= start < end <= (self.vgpr_capacity if bank is RegisterBank.VGPR else self.sgpr_capacity)):
      raise ValueError(f"invalid {bank.value} allocation window")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
      raise ValueError("lease width must be positive")
    if not isinstance(align, int) or isinstance(align, bool) or align <= 0:
      raise ValueError("lease alignment must be positive")
    cursor = (start + align - 1) // align * align
    for lease in sorted((x for x in self._leases if x.bank is bank and x.end > start and x.start < end), key=lambda x: x.start):
      if cursor + count <= min(lease.start, end): return self.reserve(name, cursor, count, bank=bank, align=align)
      cursor = (max(cursor, lease.end) + align - 1) // align * align
    if cursor + count > end: raise ValueError(f"{bank.value} allocation window exhausted for {name} ({count} regs)")
    return self.reserve(name, cursor, count, bank=bank, align=align)

  @classmethod
  def with_fixed_abi(cls) -> "AMDRegisterLeaseAllocator":
    out = cls()
    for name, start, count in (("abi", 0, 4), ("buffer_a", 4, 2), ("buffer_b", 6, 2),
                               ("output", 8, 2), ("workgroup_coords", 10, 2), ("loop_counter", 16, 1)):
      out.reserve(name, start, count, bank=RegisterBank.SGPR, align=2 if count == 2 else 1)
    out.reserve("fixed_lane_and_address", 0, 10, bank=RegisterBank.VGPR)
    return out


def allocate_amd_stage_buffer_leases(specs: tuple[AMDStageBufferSpec, ...], *, window: tuple[int, int],
                                     reserved: tuple[tuple[str, int, int], ...] = (), vgpr_capacity: int | None = None
                                     ) -> dict[str, Lease]:
  """Allocate physical stage carriers once, in stable A/B order.

  ``reserved`` contains physical ``(name, start, end)`` intervals owned by the
  ABI, accumulators, fragments, scratch, or another renderer facility.  The
  returned leases are the physical truth consumed by instruction selection.
  """
  by_role: dict[str, AMDStageBufferSpec] = {}
  for spec in specs:
    prior = by_role.get(spec.role)
    if prior is not None and prior.snapshot() != spec.snapshot():
      raise ValueError(f"conflicting register stage-buffer contracts for {spec.role}")
    by_role[spec.role] = spec
  alloc = AMDRegisterLeaseAllocator(vgpr_capacity=vgpr_capacity)
  for name, start, end in sorted(reserved, key=lambda x: (x[1], x[2], x[0])):
    if end > start: alloc.reserve(name, start, end-start, bank=RegisterBank.VGPR)
  out: dict[str, Lease] = {}
  for role in ("A", "B"):
    if (spec := by_role.get(role)) is not None:
      out[role] = alloc.allocate_in_window(f"stage_{role}", spec.packed_vgpr_width, bank=RegisterBank.VGPR,
                                           start=window[0], end=window[1])
  return out
