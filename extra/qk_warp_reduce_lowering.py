#!/usr/bin/env python3
"""Auto-lowering of a warp-axis REDUCE to the AMD cross-lane ds_bpermute ladder (Milestone 5 of the
generic-low-level-search goal: docs/generic-low-level-search-goal-scope.md).

WHAT THIS ADDS over extra/amd_warp_reduce.py: that module exposes the ladder as a function you CALL by hand
(`warp_reduce_sum(val, lane)`) -- a kernel-authoring escape hatch. This module is a PatternMatcher RULE that
AUTO-DETECTS a generic `Ops.REDUCE` over a full-warp lane axis and rewrites it to that ladder. That is the piece a
scheduler / search needs: a reduction the optimizer maps onto the warp (AxisType.WARP range) is lowered to
`ds_bpermute` cross-lane shuffles instead of the default LDS-tree (DEFINE_LOCAL + store + s_barrier), with no
hand-written kernel.

INSERTION POINT (see docs): inject `pm_warp_reduce` into the `expander` graph_rewrite in
tinygrad/codegen/__init__.py:80, BEFORE pm_group_for_reduce, so it claims the warp-axis reduce before the
GROUP_REDUCE->LDS machinery does. The lane (the WARP RANGE) is left intact in the ladder so pm_add_gpudims
(tinygrad/codegen/gpudims.py) maps it to a real `lidx` SPECIAL -- the ladder must run multi-thread or ds_bpermute
has no wave (extra/amd_warp_reduce.py docstring, gotcha #2).

STATUS: first pass. The rule + its rewrite logic are unit-tested (test/external/test_warp_reduce_lowering.py).
Pipeline-integration behind WARP_REDUCE_LOWERING is opt-in; full model wiring (an opt that maps a decode-GEMV
reduce axis onto the warp, then comparing a search-generated GEMV against the owned warp GEMV under W==D) is the
Milestone-6 follow-on. AMD wave32 (gfx1100); ADD/MAX, float only.
"""
from __future__ import annotations

from tinygrad.uop.ops import UOp, Ops, AxisType, PatternMatcher, UPat
from tinygrad.dtype import dtypes
from extra.amd_warp_reduce import warp_reduce_max, _staged_shfl, _STAGE_SLOT, WARP

# Auto-lowering must STAGE every shuffle into a REG (like warp_reduce_max does). When a reduce result is stored
# from a single lane (the gpudims gate `if(lidx0==0)`), an INLINE ds_bpermute (plain warp_reduce_sum) gets pulled
# INSIDE that divergent gate -> the cross-lane read targets a masked-off lane -> garbage (verified: K=16 matvec,
# max_err 6.7). Staging forces all ds_bpermute to run unconditionally on every lane before the gated store.
def _warp_reduce_sum_staged(val:UOp, lane:UOp, width:int = WARP, slot_base:int = _STAGE_SLOT) -> UOp:
  # Phase M (occupancy): the butterfly stages are SEQUENTIAL (each stage's store->load->add completes before the
  # next), so they reuse ONE staging slot instead of one-per-stage. Saves (log2(width)-1) per-thread LDS slots
  # (e.g. 4*4B*128 = 2048 B for width=32) -> lifts the native decode tile's group segment and occupancy.
  off = width >> 1
  while off >= 1:
    val = val + _staged_shfl(val, off, lane, slot_base); off >>= 1
  return val

_LADDER = {Ops.ADD: _warp_reduce_sum_staged, Ops.MAX: warp_reduce_max}
# A reduce axis that the optimizer maps onto the wave: WARP (TC path) or GROUP_REDUCE (a GROUP/GROUPTOP opt --
# this is what the matrix-vector heuristic applies, Opt(OptOps.GROUP,0,MV_THREADS_PER_ROW), heuristic.py:77).
_LANE_AXES = (AxisType.WARP, AxisType.GROUP_REDUCE)

_POW2_WIDTHS = (2, 4, 8, 16, 32)   # within a wave32; the ladder is log2(width) ds_bpermute steps

def _lane_width(r:UOp) -> int|None:
  """The wave width of a lane reduce range, or None if not a lowerable lane axis. A sub-wave group (e.g. the
  heuristic's GROUPTOP=16) maps to lanes 0..w-1 and the xor-ladder offsets stay in range, so it is correct."""
  if r.op is Ops.RANGE and r.arg[-1] in _LANE_AXES and r.src[0].op is Ops.CONST and r.src[0].arg in _POW2_WIDTHS:
    return r.src[0].arg
  return None

def lower_warp_reduce(red:UOp) -> UOp|None:
  """REDUCE(val, [serial...,] lane_range, arg=(alu, axes)) -> [serial reduce then] the ds_bpermute ladder.
  Handles a single power-of-2 (<=32) WARP/GROUP_REDUCE lane axis, with OR without serial reduce ranges
  (a real GEMV's K=4096 becomes group + serial-K/group). alu in {ADD,MAX}; float.
  Mirror of fix_group_for_reduce (expander.py): do the non-lane (serial) reduce first as a per-lane partial,
  then the cross-lane ladder over the lane group. The lane RANGE is reused (gpudims binds it to a lidx); the
  serial ranges stay AxisType.REDUCE and are lowered to a REG-accumulator loop by pm_reduce. REDUCE.arg is
  already (op, ()) at this stage (indexing.py converts the positional-axes form away) -- never recompute it.
  Strict superset of the single-axis case: serial==[] reproduces the original ladder-over-src[0]."""
  alu, _axes = red.arg
  if alu not in _LADDER: return None
  ranges = red.src[1:]
  if not all(r.op is Ops.RANGE for r in ranges): return None
  group = [r for r in ranges if r.arg[-1] in _LANE_AXES]
  serial = [r for r in ranges if r.arg[-1] not in _LANE_AXES]
  if len(group) != 1: return None
  if (w := _lane_width(group[0])) is None: return None
  if red.dtype.scalar() not in (dtypes.float32, dtypes.float): return None
  # Decline a VECTORIZED reduce value (UPCAST/UNROLL axis in src[0], e.g. the matvec heuristic's
  # MV_ROWS_PER_THREAD>1): the scalar ds_bpermute ladder bit-casts one float per lane and cannot shuffle a vector.
  # Falls back to the LDS tree (correct). Per-component shuffle is a follow-on; use MV_ROWS_PER_THREAD=1 to fire.
  if any(u.op is Ops.RANGE and u.arg[-1] in (AxisType.UPCAST, AxisType.UNROLL) for u in red.src[0].toposort()): return None
  inner = red.src[0].reduce(*serial, arg=red.arg) if serial else red.src[0]   # per-lane partial; lane stays live
  return _LADDER[alu](inner, group[0], w)

pm_warp_reduce = PatternMatcher([
  (UPat(Ops.REDUCE, name="red"), lower_warp_reduce),
])
