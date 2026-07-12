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

  @classmethod
  def with_fixed_abi(cls) -> "AMDRegisterLeaseAllocator":
    out = cls()
    for name, start, count in (("abi", 0, 4), ("buffer_a", 4, 2), ("buffer_b", 6, 2),
                               ("output", 8, 2), ("workgroup_coords", 10, 2), ("loop_counter", 16, 1)):
      out.reserve(name, start, count, bank=RegisterBank.SGPR, align=2 if count == 2 else 1)
    out.reserve("fixed_lane_and_address", 0, 10, bank=RegisterBank.VGPR)
    return out

