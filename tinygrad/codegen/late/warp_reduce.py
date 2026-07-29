#!/usr/bin/env python3
"""Shape-safe warp/lane primitives for AMD (gfx1100, wave32) custom kernels.

The retired flash-attention reference used warp reductions with `UOp(Ops.CUSTOM, ds_bpermute)`, but Ops.CUSTOM is
SHAPELESS in current tinygrad (ops.py:229), so feeding it into Ops.MAX/reshape trips shape inference. Two fixes
(both kernel-authoring, no codegen surgery):
  1. use **Ops.CUSTOMI** (the inline variant) -- it carries `src[0]._shape` (ops.py:306), so put the shaped
     value first; the result is shaped and composes with Ops.MAX/+/reshape.
  2. tie the lane to a real thread dim (`UOp.special(32, "lidx0")`), NOT a bare AxisType.WARP range -- a bare
     WARP axis renders as a serial for-loop in a 1-thread workgroup, so ds_bpermute has no wave (garbage).

These are the shape-safe building blocks for reviving the flash-attention kernel (WR ladder). gfx1100 = wave32.
"""
from __future__ import annotations

from tinygrad.uop.ops import UOp, Ops, AxisType, PatternMatcher, UPat
from tinygrad.dtype import AddrSpace, dtypes

WARP = 32  # gfx1100 wave width
_STAGE_SLOT = 90  # REG slots for staging cross-lane reads (kept clear of kernel slots)

def warp_shfl_xor(val:UOp, offset:int, lane:UOp) -> UOp:
  """Read `val` from lane (lane ^ offset) via ds_bpermute. `lane` must be a real thread dim (lidx). Shape-safe:
  CUSTOMI carries src[0] (=val) shape. Float values only (bit-cast through int for the permute)."""
  idx = ((lane ^ offset) * 4).cast(dtypes.int)   # ds_bpermute addr = source_lane*4 (byte address)
  return UOp(Ops.CUSTOMI, val.dtype, (val, idx),
             arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({1}, __builtin_bit_cast(int, {0})))")

def _staged_shfl(val:UOp, offset:int, lane:UOp, slot:int) -> UOp:
  # Materialize the cross-lane read into a REG before consuming it. CUSTOMI is INLINE, so feeding the shuffle
  # straight into a max() ternary puts ds_bpermute (a wave-level op) inside a data-dependent conditional ->
  # lane divergence -> garbage. Staging forces ONE unconditional bpermute; the max then selects two registers.
  reg = UOp.placeholder((1,), val.dtype, slot, addrspace=AddrSpace.REG)
  return reg.after(reg[0].store(warp_shfl_xor(val, offset, lane)))[0]

def warp_reduce_max(val:UOp, lane:UOp, width:int = WARP, slot_base:int = _STAGE_SLOT) -> UOp:
  off, slot = width >> 1, slot_base
  while off >= 1:
    val = val.maximum(_staged_shfl(val, off, lane, slot)); off >>= 1; slot += 1
  return val   # every lane holds the width-wide max

def warp_reduce_max_native_vgpr(val:UOp, lane:UOp, width:int = WARP) -> UOp:
  """Native AMD form: ds_bpermute already materializes an unconditional VGPR result."""
  off = width >> 1
  while off >= 1:
    val = val.maximum(warp_shfl_xor(val, off, lane)); off >>= 1
  return val

def warp_reduce_sum(val:UOp, lane:UOp, width:int = WARP) -> UOp:
  off = width >> 1
  while off >= 1:
    val = val + warp_shfl_xor(val, off, lane)
    off >>= 1
  return val   # every lane holds the width-wide sum


# Auto-lowering for optimizer-produced lane reductions. This must stage every shuffle into a REG because an inline
# ds_bpermute can be pulled into a divergent single-lane writeback gate. The hand-built primitives above remain
# available to kernel authors through this core owner.
def _warp_reduce_sum_staged(val:UOp, lane:UOp, width:int = WARP, slot_base:int = _STAGE_SLOT) -> UOp:
  off = width >> 1
  while off >= 1:
    val = val + _staged_shfl(val, off, lane, slot_base); off >>= 1
  return val


_LADDER = {Ops.ADD: _warp_reduce_sum_staged, Ops.MAX: warp_reduce_max}
_LANE_AXES = (AxisType.WARP, AxisType.GROUP_REDUCE)
_POW2_WIDTHS = (2, 4, 8, 16, 32)


def _lane_width(r:UOp) -> int|None:
  if r.op is Ops.RANGE and r.arg[-1] in _LANE_AXES and r.src[0].op is Ops.CONST and r.src[0].arg in _POW2_WIDTHS:
    return r.src[0].arg
  return None


def lower_warp_reduce(red:UOp) -> UOp|None:
  """Lower one scalar WARP/GROUP_REDUCE REDUCE to the staged ds_bpermute ladder."""
  alu, _axes = red.arg
  if alu not in _LADDER: return None
  ranges = red.src[1:]
  if not all(r.op is Ops.RANGE for r in ranges): return None
  group = [r for r in ranges if r.arg[-1] in _LANE_AXES]
  serial = [r for r in ranges if r.arg[-1] not in _LANE_AXES]
  if len(group) != 1: return None
  if (w := _lane_width(group[0])) is None: return None
  if red.dtype.scalar() not in (dtypes.float32, dtypes.float): return None
  if any(u.op is Ops.RANGE and u.arg[-1] in (AxisType.UPCAST, AxisType.UNROLL) for u in red.src[0].toposort()): return None
  inner = red.src[0].reduce(*serial, arg=red.arg) if serial else red.src[0]
  return _LADDER[alu](inner, group[0], w)


pm_warp_reduce = PatternMatcher([
  (UPat(Ops.REDUCE, name="red"), lower_warp_reduce),
])
