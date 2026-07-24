"""AMD gfx1100 register reservation and physical layout descriptor."""
from __future__ import annotations

from typing import NamedTuple

from tinygrad.codegen.opt.register_contracts import Lease, RegisterBank, RegisterDescriptor, RegisterRole
from tinygrad.renderer.isa import Register


# Physical gfx1100 layout shared by lowering, allocation, and resource reporting.
KARG = Register("s0", 0); SPTR_POOL = tuple(Register(f"s{i}", i) for i in range(6, 40, 2))
SCNT_POOL = tuple(Register(f"s{i}", i) for i in range(40, 64)); VBASE = tuple(Register(f"v{i}", i) for i in range(256))
TID = Register("v0", 0); WGID_S0 = 2; FRAG_BASE, FRAG_TOP = 200, 238
LDS_PACK_BASE, LDS_PACK_TOP = 232, 236; WMMA_ACC_BASE = 8


class AMDAttentionLoopStateMap(NamedTuple):
  """The one physical VGPR map for the native Hd128 fused-attention loop state.

  This is a privileged operation: it hands out *physical* register numbers that
  bypass the register allocator entirely. It is contained here, with its
  contract stated, because it used to exist as a bare `{"m":72,"l":80,"acc":8}`
  dict repeated in four places with no stated invariant, no owner, and no
  defending test.

  WHAT INVARIANT MAKES IT VALID
    Only a kernel for which `_has_hd128_attention(ctx)` is true may use this
    map. For such a kernel `_n_c_runs()` returns `runs()` (12), so `_acc_top()`
    reserves the whole window `[acc, top())` = v8..v103 and `_vpool()` excludes
    it: no virtual register can ever be placed there. Every span is 8 registers
    wide, 8-aligned, mutually disjoint, and strictly below FRAG_BASE. If any of
    those stop holding, `validate()` raises rather than letting two roles alias.

  WHO IS ALLOWED TO CALL IT
    Only the AMD:ISA native attention lowering path:
      * amd_attention_abi.lower_amd_attention_loop_state (typed Ops ->
        fixed-register alias),
      * amd.py isel_attention_loop_state and isel_customi's
        amd_gfx1100_{attention_loop_state,row_state}_write markers,
      * amd.py lower_inst's ATTENTION_OUTPUT_DRAIN / ATTENTION_STATS_DRAIN
        encoders, which read the m/l/acc spans back out,
      * amd_wmma_residency, which sizes the reservation from `runs()`.
    Generic instruction selection must not call it, and nothing outside
    DEV=AMD:ISA may call it at all.

  WHAT STATE IT MAY MUTATE
    None. It is a pure descriptor: it returns integers. The register writes
    those integers name are made by the caller, which is why the caller list
    above is the real boundary and this object is only its documentation.

  WHAT DEFENDS IT
    `validate()` (called at module import on the shipped instance) plus
    test/unit/test_state_phase_abi.py, which asserts the spans are disjoint,
    aligned, inside the reserved window, and that every consumer derives its
    register numbers from this map rather than restating them.

  RECORDED COST OF HAVING BEEN UNDOCUMENTED
    docs/shared-attention-phase-lds-negative-result-20260724.md. The phase-ABI
    LDS experiment tried to move this state to a typed StateHandle region and
    measured a negative result (+2048 B LDS, 197 VGPRs, zero registers freed):
    publication created a *mirror* of these registers instead of transferring
    ownership of them, because this hard alias had no owner to transfer from.
  """
  acc: int = WMMA_ACC_BASE   # 8 persistent PV C fragments, v8..v71 (block i at acc + i*8)
  m: int = 72                # native row max,   v72..v79
  l: int = 80                # native row sum,   v80..v87
  qk_c: int = 88             # transient QK C,   v88..v95
  alpha: int = 96            # online rescale,   v96..v103
  lanes: int = 8             # wave32 C-fragment width; also the span width of every role
  blocks: int = 8            # Hd128 head-dim blocks -> acc spans blocks*lanes registers

  def base(self, role:str, block:int=0) -> int:
    """Physical VGPR base for one loop-state role. `block` applies to "acc" only."""
    if role == "acc":
      if not 0 <= block < self.blocks: raise ValueError(f"AMD attention loop-state acc block {block} out of range")
      return self.acc + block*self.lanes
    if block: raise ValueError(f"AMD attention loop-state role {role!r} has no block dimension")
    if (start := getattr(self, role, None)) is None or role not in ("m", "l", "qk_c", "alpha"):
      raise ValueError(f"unknown AMD attention loop-state role {role!r}")
    return start

  def spans(self) -> tuple[tuple[str, int, int], ...]:
    """(role, first, last_exclusive) for every span this map owns."""
    return tuple([(f"acc{i}", self.acc + i*self.lanes, self.acc + (i+1)*self.lanes) for i in range(self.blocks)] +
                 [(r, getattr(self, r), getattr(self, r)+self.lanes) for r in ("m", "l", "qk_c", "alpha")])

  def top(self) -> int:
    """First VGPR above the reserved window."""
    return max(end for _, _, end in self.spans())

  def runs(self) -> int:
    """Number of 8-VGPR runs the reservation must cover (what _n_c_runs reports)."""
    return (self.top() - self.acc) // self.lanes

  def validate(self):
    if self.lanes != 8 or self.acc != WMMA_ACC_BASE: raise ValueError("AMD attention loop state requires the wave32 8-lane C fragment at WMMA_ACC_BASE")
    seen: dict[int, str] = {}
    for role, start, end in self.spans():
      if start % self.lanes or end > FRAG_BASE: raise ValueError(f"AMD attention loop-state span {role} is misaligned or outside the low window")
      for i in range(start, end):
        if i in seen: raise ValueError(f"AMD attention loop-state spans {seen[i]} and {role} alias at v{i}")
        seen[i] = role
    if self.top() - self.acc != len(seen): raise ValueError("AMD attention loop-state reservation is not contiguous")
    return self


# The shipped map. Validated at import so a bad edit fails loudly instead of aliasing two roles.
AMD_ATTENTION_LOOP_STATE = AMDAttentionLoopStateMap().validate()


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
