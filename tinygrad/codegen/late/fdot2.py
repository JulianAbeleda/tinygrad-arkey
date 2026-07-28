#!/usr/bin/env python3
"""Opt-in AMD fdot2 lowering for the exact post-devectorize fp16 dot2 idiom.

This is intentionally narrow. It recognizes the C-style AMD lowering shape:

  (float)(a.x*b.x) + (float)(a.y*b.y)

where a.x/a.y and b.x/b.y are the two lanes of the same two half2 values, and
rewrites it to:

  __builtin_amdgcn_fdot2(a, b, acc, false)

The rule is a primitive-exposure hook, not a speed claim. It is gated by
V_DOT2_LOWERING in tinygrad/codegen/__init__.py.
"""
from __future__ import annotations

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat

_FDOT2 = "__builtin_amdgcn_fdot2({1}, {2}, {0}, false)"

def _const_idx(u: UOp) -> int | None:
  return u.arg if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.int, dtypes.weakint) and u.arg in (0, 1) else None

def _indexed_half_lane(u: UOp) -> tuple[UOp, int] | None:
  if u.op is not Ops.INDEX or len(u.src) != 2 or u.dtype.scalar() is not dtypes.half: return None
  idx = _const_idx(u.src[1])
  if idx is None: return None
  return u.src[0], idx

def _cast_half_mul_term(u: UOp) -> tuple[UOp, UOp, int] | None:
  if u.op is not Ops.CAST or u.dtype.scalar() is not dtypes.float or len(u.src) != 1: return None
  mul = u.src[0]
  if mul.op is not Ops.MUL or mul.dtype.scalar() is not dtypes.half or len(mul.src) != 2: return None
  a = _indexed_half_lane(mul.src[0])
  b = _indexed_half_lane(mul.src[1])
  if a is None or b is None or a[1] != b[1]: return None
  return a[0], b[0], a[1]

def _dot2_pair(a: UOp, b: UOp) -> tuple[UOp, UOp] | None:
  ta = _cast_half_mul_term(a)
  tb = _cast_half_mul_term(b)
  if ta is None or tb is None: return None
  # Same two half2 sources, opposite lanes. Decline anything except x/y from both operands.
  if ta[0] is tb[0] and ta[1] is tb[1] and {ta[2], tb[2]} == {0, 1}: return ta[0], ta[1]
  return None

def _fdot2(a: UOp, b: UOp, acc: UOp | None = None) -> UOp:
  # CUSTOMI carries src[0]'s shape. Put the scalar f32 accumulator first so the fdot2 result is scalar-shaped even
  # when a/b are half2 values.
  return UOp(Ops.CUSTOMI, dtypes.float, (acc if acc is not None else UOp.const(dtypes.float, 0.0), a, b), arg=_FDOT2)

def line_lower_fdot2(lst: list[UOp]) -> list[UOp]:
  """Apply fdot2 lowering to a linearized UOp list while preserving replacement dependencies."""
  out: list[UOp] = []
  emitted: set[UOp] = set()
  replaced: dict[UOp, UOp] = {}
  for u in lst:
    nu = u.replace(src=tuple(replaced.get(x, x) for x in u.src))
    lowered = lower_fdot2_add(nu)
    if lowered is not None:
      for s in lowered.src:
        if s not in emitted:
          out.append(s); emitted.add(s)
    replaced[u] = lowered if lowered is not None else nu
    out.append(replaced[u])
    emitted.add(replaced[u])
  return out

def lower_fdot2_add(add: UOp) -> UOp | None:
  if add.op is not Ops.ADD or add.dtype.scalar() is not dtypes.float or len(add.src) != 2: return None
  if (pair := _dot2_pair(add.src[0], add.src[1])) is not None: return _fdot2(*pair)
  # Optional f32 accumulator: acc + dot2_pair or dot2_pair + acc. The pair may be nested under one ADD side.
  for pair_side, acc_side in ((add.src[0], add.src[1]), (add.src[1], add.src[0])):
    if acc_side.dtype.scalar() is not dtypes.float: continue
    if pair_side.op is Ops.ADD and (pair := _dot2_pair(pair_side.src[0], pair_side.src[1])) is not None:
      return _fdot2(pair[0], pair[1], acc_side)
  return None

pm_fdot2 = PatternMatcher([
  (UPat(Ops.ADD, name="add"), lower_fdot2_add),
])
