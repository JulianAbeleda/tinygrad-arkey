"""Row-local lowering: one workgroup reduces a row, then the same lanes write the row's elementwise outputs.

This is the shape of vLLM's fused_add_rms_norm, derived from the kernel instead of written per op. It fits a kernel with
one plain REDUCE over a single reduce range R and an output range d that the reduce does not depend on. Under
SIBLING_FUSE>=2, rangeify keeps a row reduce inside its consumer (schedule/indexing.py), which produces this kernel.
The generic heuristic would make d a global axis, so every thread (or CPU core) recomputes the whole reduce. Here:
  - the output ranges the reduce depends on (the rows) are the grid
  - R and d are split by the same lane count L, and both splits use ONE lane range. R's lane part is a GROUP_REDUCE:
    per-lane partials go through shared memory and every lane reads the combined value. d's lane part is that same
    lane index, so the lanes that reduced the row then write it (d = d_out*L + lane, d_out a loop)
  - without local memory (CPU), the rows stay the only parallel axis and d stays a loop after the reduce
Legality is structural. The reduce must not depend on d (if it did, sharing a lane between them would change the
reduce's value), and L must divide both extents. The rule declines otherwise, and the heuristic runs as before.
Reduction order: each lane sums R/L strided elements, then the L partials are summed in lane order. That is a
different tree from the separate reduce kernel, so values change at rounding level (a reorder, not a precision loss).
"""
from __future__ import annotations
from tinygrad.uop.ops import Ops, AxisType, UOp

def _is_contraction(red:UOp) -> bool:
  rr = set(red.src[1:])
  return any(u.op is Ops.MUL and u.src[0] is not u.src[1] and all(rr & set(x.ranges) for x in u.src) for u in red.src[0].toposort())

def lane_count(r:int, d:int, max_lanes:int) -> int|None:
  """The largest common divisor of r and d up to max_lanes, preferring whole warps (multiples of 32)."""
  common = [l for l in range(max_lanes, 1, -1) if r % l == 0 and d % l == 0]
  warps = [l for l in common if l % 32 == 0]
  return (warps or common or [None])[0]

def row_local_plan(k) -> tuple[UOp, UOp, int]|None:
  """(reduce range R, independent output range d, lanes L) when the kernel has the row-local shape, else None."""
  reds = k.reduceops
  if len(reds) != 1: return None
  red = reds[0]
  if not (isinstance(red.arg, tuple) and red.arg[0] in {Ops.ADD, Ops.MAX}) or _is_contraction(red): return None
  rr = [r for r in red.src[1:] if r.op is Ops.RANGE and r.arg[-1] is AxisType.REDUCE]
  if len(rr) != 1 or len(red.src) != 2: return None
  R = rr[0]
  deps = set(red.src[0].ranges)
  outs = [r for r in k._output_rngs() if r.arg[-1] in (AxisType.GLOBAL, AxisType.LOOP)]
  indep = [r for r in outs if r not in deps]
  if len(indep) != 1 or not isinstance(R.vmax, int) or not isinstance(indep[0].vmax, int): return None
  d = indep[0]
  if not k.ren.has_local: return R, d, 1
  max_lanes = min(k.ren.local_max[0] if k.ren.local_max else 256, 512)
  if (L := lane_count(R.vmax+1, d.vmax+1, max_lanes)) is None: return None
  return R, d, L

def apply_row_local(k) -> bool:
  if (plan := row_local_plan(k)) is None: return False
  R, d, L = plan
  if L == 1:
    # CPU: rows are the parallel axis, d stays a loop after the reduce (a global d recomputes the reduce per core)
    if d.arg[-1] is not AxisType.LOOP: k.ast = k.ast.substitute({d: d.replace(arg=d.arg[:-1]+(AxisType.LOOP,))})
    return True
  d_loop = d.replace(arg=d.arg[:-1]+(AxisType.LOOP,))
  k.ast = k.ast.substitute({d: d_loop})
  _, lane = k.shift_to(R, L, AxisType.GROUP_REDUCE)
  k.shift_to(d_loop, L, AxisType.GROUP_REDUCE, input_new_rng=lane)
  return True
