"""AMD gfx1100 register reservation and physical layout descriptor."""
from __future__ import annotations

from tinygrad.codegen.opt.register_contracts import Lease, RegisterBank, RegisterDescriptor, RegisterRole
from tinygrad.renderer.isa import Register


# Physical gfx1100 layout shared by lowering, allocation, and resource reporting.
KARG = Register("s0", 0); SPTR_POOL = tuple(Register(f"s{i}", i) for i in range(6, 40, 2))
SCNT_POOL = tuple(Register(f"s{i}", i) for i in range(40, 64)); VBASE = tuple(Register(f"v{i}", i) for i in range(256))
TID = Register("v0", 0); WGID_S0 = 2; FRAG_BASE, FRAG_TOP = 200, 238
LDS_PACK_BASE, LDS_PACK_TOP = 232, 236; WMMA_ACC_BASE = 8


def gfx1100_register_descriptor() -> RegisterDescriptor:
  """Return the immutable logical reservation snapshot for the current renderer."""
  sgpr = RegisterBank.SGPR
  vgpr = RegisterBank.VGPR
  leases = (
    Lease(RegisterRole.KERNARG, sgpr, 0, 2, mode="abi", alignment=2),
    Lease(RegisterRole.WORKGROUP_ID, sgpr, 2, 4, mode="abi"),
    Lease(RegisterRole.POINTER, sgpr, 6, 34, mode="abi", alignment=2),
    Lease(RegisterRole.SCALAR_COUNTER, sgpr, 40, 24, mode="default"),
    Lease(RegisterRole.SCALAR_TEMP, sgpr, 64, 40, mode="default"),
    Lease(RegisterRole.WORKITEM_ID, vgpr, 0, 1, mode="abi"),
    Lease(RegisterRole.VIRTUAL, vgpr, 1, 255, mode="no_wmma"),
    Lease(RegisterRole.VIRTUAL, vgpr, 1, 199, mode="wmma_single_tile"),
    Lease(RegisterRole.ACCUMULATOR, vgpr, 1, 16, mode="legacy_accum"),
    Lease(RegisterRole.ACCUMULATOR, vgpr, 8, 8, mode="wmma_multi_tile", alignment=8),
    Lease(RegisterRole.FRAGMENT, vgpr, 200, 38, mode="wmma_single_tile", alignment=2),
    Lease(RegisterRole.LDS_PACK, vgpr, 232, 4, mode="lds_pack", alignment=2),
  )
  return RegisterDescriptor("AMD:gfx1100", 32, 256, 104, leases)


GFX1100_REGISTER_DESCRIPTOR = gfx1100_register_descriptor()
