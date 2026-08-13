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

# Renderer-lowered operation tag (same shape as CStyleLanguage.barrier/float4/smem_prefix, but this one needs a
# builder rather than a fixed string, so it is dispatched through `CStyleLanguage.warp_shfl_xor` -- see
# `pm_lower_warp_shfl_xor` below and the providers on HIPRenderer/MetalRenderer/CUDARenderer in renderer/cstyle.py
# and renderer/cuda.py). This tag makes the op renderer-agnostic where it is built (kernel-authoring time, before
# any target is chosen) and resolved only once a concrete renderer is known.
WARP_SHFL_XOR_TAG = "warp_shfl_xor"

def warp_shfl_xor(val:UOp, offset:int, lane:UOp) -> UOp:
  """Cross-lane XOR shuffle: read `val` from lane (lane ^ offset). Shape-safe: CUSTOMI carries src[0] (=val)
  shape. `lane` must be a real thread dim (lidx) for providers that need per-lane addressing (e.g. AMD's
  ds_bpermute, which computes a byte address from it); providers that take a lane mask directly (e.g. Metal's
  simd_shuffle_xor) ignore `lane` entirely. Which text (if any) this lowers to is a renderer decision, resolved
  by `pm_lower_warp_shfl_xor`, never baked in here."""
  return UOp(Ops.CUSTOMI, val.dtype, (val, lane), arg=(WARP_SHFL_XOR_TAG, offset))

# Byte-address variant of the same mechanism, used by the fused-attention row-softmax lowering. The caller
# computes the register byte address itself (the repack's XOR butterfly `(lane^mask)*4` and the row-state
# broadcast's `(lane&16)*4` differ, so the address cannot be derived from a mask alone), and the operation is
# "read the fp32 slot at byte address `addr`". AMD renders this as ds_bpermute with the unsigned-int value cast
# the pinned attention rendering uses; CUDA renders `__shfl_sync(..., addr >> 2)`; a renderer with no provider
# fails loudly at lowering. This tag is deliberately NOT `warp_shfl_xor`: decode's shuffle providers are
# byte-pinned with their own spelling, and this op carries `(addr, value)` sources.
WARP_BPERMUTE_TAG = "warp_bpermute"

def warp_bpermute(addr:UOp, value:UOp) -> UOp:
  """Cross-lane register read by byte address, renderer-neutral. `addr` must be the byte offset of the source
  lane's register slot (AMD ds_bpermute convention); `value` is the fp32 this lane holds. Resolution is a
  renderer decision (`pm_lower_warp_bpermute`), never baked in here."""
  return UOp(Ops.CUSTOMI, value.dtype, (addr, value), arg=(WARP_BPERMUTE_TAG,))

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


def warp_reduce_sum_across_groups(val:UOp, lane:UOp, group_size:int, slot_base:int = _STAGE_SLOT) -> UOp:
  """Sum `val` across `group_size`-aligned lane groups (offsets group_size, 2*group_size, ... < WARP).

  This is the upper half of a full warp reduce: offsets below `group_size` are assumed already reduced
  inside each group. Used by the llama-vec flash substrate where an 8-lane QK dot is followed by a
  cross-group (4-groups-per-warp) PV/denominator combine. XOR offsets >= group_size preserve the
  lane-within-group bits, so each lane ends up holding the whole-warp sum for its group position.
  """
  off = group_size
  while off < WARP:
    val = val + _staged_shfl(val, off, lane, slot_base); off <<= 1
  return val


def warp_reduce_max_across_groups(val:UOp, lane:UOp, group_size:int, slot_base:int = _STAGE_SLOT) -> UOp:
  """Max-variant of `warp_reduce_sum_across_groups` (see that docstring)."""
  off = group_size
  while off < WARP:
    val = val.maximum(_staged_shfl(val, off, lane, slot_base)); off <<= 1
  return val


# Auto-lowering for optimizer-produced lane reductions. This must stage every shuffle into a REG because an inline
# ds_bpermute can be pulled into a divergent single-lane writeback gate. The hand-built primitives above remain
# available to kernel authors and to the existing extra/llm_research emitters through this core owner.
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


def _lower_warp_shfl_xor(ctx, x:UOp) -> UOp|None:
  """Resolve one tagged warp_shfl_xor CUSTOMI against the target renderer (`ctx`). Never falls back silently:
  a renderer that does not declare `warp_shfl_xor` raises, naming both the operation and the target, per the
  evidence contract (a target that cannot express a program must fail loudly at lowering, not run generic text
  on hardware it wasn't written for)."""
  if not (isinstance(x.arg, tuple) and x.arg[:1] == (WARP_SHFL_XOR_TAG,)): return None
  _, offset = x.arg
  val, lane = x.src
  if (provider := getattr(ctx, "warp_shfl_xor", None)) is None:
    raise NotImplementedError(f"{WARP_SHFL_XOR_TAG} is not available on {type(ctx).__name__} "
                               f"(target={ctx.target.device}:{ctx.target.arch})")
  return provider(val, offset, lane)


def _lower_warp_bpermute(ctx, x:UOp) -> UOp|None:
  """Resolve one tagged warp_bpermute CUSTOMI against the target renderer (`ctx`), same fail-loud contract as
  `_lower_warp_shfl_xor` above: no provider means this target cannot express a byte-address cross-lane read,
  and that is an error named at lowering, never a silent fallback."""
  if not (isinstance(x.arg, tuple) and x.arg[:1] == (WARP_BPERMUTE_TAG,)): return None
  if (provider := getattr(ctx, "warp_bpermute", None)) is None:
    raise NotImplementedError(f"{WARP_BPERMUTE_TAG} is not available on {type(ctx).__name__} "
                               f"(target={ctx.target.device}:{ctx.target.arch})")
  return provider(*x.src)


# Dispatches purely on the renderer instance passed as `ctx` (never on Device.DEFAULT or a device string) --
# see codegen/__init__.py for where this runs: once eagerly on the incoming AST (so hand-authored kernels see
# the exact same lowering position as the AMD string used to occupy) and once more as a final-rewrite safety
# net (so any WARP_REDUCE_LOWERING-produced tag, built later during the expander pass, still gets resolved).
pm_lower_warp_shfl_xor = PatternMatcher([
  (UPat(Ops.CUSTOMI, name="x"), _lower_warp_shfl_xor),
])

pm_lower_warp_bpermute = PatternMatcher([
  (UPat(Ops.CUSTOMI, name="x"), _lower_warp_bpermute),
])
