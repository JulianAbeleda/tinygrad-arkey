"""Physical VGPR aliasing primitives for the native AMD/rdna3 ISA backend.

This is the ONE place that turns a physical VGPR *number* into something the
shared register allocator will honour, and the one place that reads such a
number back off a UOp.  Everything above it (fragment residency policy, the
attention ABI, tensor-core emit) decides *which* register; this module only
knows how a chosen register is spelled.

Keeping it separate matters because these two directions must stay symmetric:
``_pin``/``_fixed_alias`` write a physical index into a tag, and
``_fixed_vgpr_index``/``_constrained_vgpr_index`` are the only readers of that
encoding.  A change to either side that is not mirrored in the other silently
mis-reads register identity, which is not something codegen can detect.
"""
from __future__ import annotations
from tinygrad.uop.ops import UOp, Ops
from tinygrad.renderer.isa import FixedRegisterUse, Register

def _pin(base:int, i:int): return (Register(f"v{base+i}", base+i),)   # physical VGPR -> constrained vreg via alloc_vregs
def _fixed_alias(base:int, i:int, dtype, *deps:UOp) -> UOp:
  fixed = UOp(Ops.NOOP, dtype, tag=(FixedRegisterUse(f"v{base+i}", base+i),))
  # NOOP.reg resolves through src[0], so keep the fixed-register sentinel first and ordering dependencies after it.
  return fixed if not deps else UOp(Ops.NOOP, dtype, src=(fixed,) + deps)

def _fixed_vgpr_index(u:UOp) -> int|None:
  return u.tag[0].index if isinstance(u.tag, tuple) and len(u.tag) == 1 and isinstance(u.tag[0], Register) else None

def _constrained_vgpr_index(u:UOp) -> int|None:
  if not (isinstance(u.tag, tuple) and len(u.tag) == 1 and isinstance(u.tag[0], Register)): return None
  cons = u.tag[0].cons
  return cons[0].index if len(cons) == 1 else None

def _fixed_contiguous_vgpr4(us:tuple[UOp, ...]) -> bool:
  return len(us) == 4 and (b := _fixed_vgpr_index(us[0])) is not None and [_fixed_vgpr_index(u) for u in us] == list(range(b, b + 4))
