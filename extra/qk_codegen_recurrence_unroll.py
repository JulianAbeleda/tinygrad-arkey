#!/usr/bin/env python3
"""Recurrence-aware loop-unroll primitive (AFTER-chain reconstruction) — the foundational codegen
scheduling primitive. Scalar-unrolls a REDUCE loop by U, re-threading the Ops.AFTER(acc, range) loop-carry
across the U copies, so the copies' independent prologues coexist in one basic block where the layer-1 list
scheduler (SCHED_LIST) interleaves them. Default-off behind SCHED_UNROLL=<U>. Pure transform, no materialize.

Recurrence representation (recon, docs/decode-codegen-recurrence-unroll-primitive-scope.md):
  carry read in-loop = AFTER(X, r)   ; per-iteration result = END(store_chain, r).src[0] ; post = AFTER(X, END)
Unroll: r2 = range(N/U); for u in 0..U-1 substitute r -> r2*U+u in indices, and rewire AFTER(X,r) to the
previous copy's state (copy0 -> AFTER(X, r2)); END now closes r2 over copy U-1's state.
"""
from __future__ import annotations
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp, AxisType


def _root_buf(u: UOp) -> UOp:
  if u.op is Ops.INDEX: return _root_buf(u.src[0])
  if u.op is Ops.AFTER: return _root_buf(u.src[0])
  return u


def _after_with_replaced_range(a: UOp, r: UOp, repl: UOp) -> UOp:
  # Preserve multi-range AFTERs: AFTER(X, a, r) -> AFTER(X, a, repl), not AFTER(X, repl).
  return UOp(Ops.AFTER, a.dtype, (a.src[0],) + tuple(repl if s is r else s for s in a.src[1:]), a.arg)


def _true_carry_afters(tl: list[UOp], afters: list[UOp]) -> list[UOp]:
  """A true carry is an AFTER whose loaded value contributes to a STORE back to the same underlying buffer.

  This intentionally excludes per-iteration re-inits like dotp.after(b, tt)[0].store(0.0): that AFTER is the
  target/order base of the STORE, not part of the stored value. It also excludes inner accumulators for other
  ranges because they are not in `afters` for the selected range.
  """
  ret = []
  for a in afters:
    # Per-iteration re-init: AFTER(X, ..., r)[idx].store(CONST). This may feed an inner accumulator over a
    # different range later (dotp/rp), but it is not the outer recurrence carry and must not be re-threaded.
    is_reinit = any(st.op is Ops.STORE and len(st.src) >= 2 and st.src[1].op is Ops.CONST and
                    a in st.src[0].toposort() for st in tl)
    if is_reinit: continue
    root = _root_buf(a)
    for st in (u for u in tl if u.op is Ops.STORE):
      if len(st.src) < 2 or _root_buf(st.src[0]) is not root: continue
      if a in st.src[1].toposort():
        ret.append(a); break
  return ret


def _last_store_to_root(u: UOp, root: UOp) -> UOp:
  stores = [x for x in u.toposort() if x.op is Ops.STORE and len(x.src) >= 1 and _root_buf(x.src[0]) is root]
  if not stores: return u
  st = stores[-1]
  # If the store is closed by an inner LOOP/REDUCE END (for example acc[dd].store(...).end(dd)), the carry
  # for the next unrolled token must depend on that closure, not the raw STORE inside the sibling range.
  for x in u.toposort():
    if x.op is Ops.END and st in x.toposort(): return x
  return st


def _unroll_one_range(sink: UOp, U: int) -> UOp | None:
  tl = list(sink.toposort())
  # candidate REDUCE ranges that have a recurrence carry AFTER(_, r) and an END(_, r), and U | N
  carry_afters: dict[UOp, list[UOp]] = defaultdict(list)
  for u in tl:
    if u.op is Ops.AFTER and len(u.src) >= 2:
      for r in u.src[1:]:
        if r.op is Ops.RANGE: carry_afters[r].append(u)
  end_of: dict[UOp, UOp] = {}
  for u in tl:
    if u.op is Ops.END:
      for r in u.src[1:]:
        if r.op is Ops.RANGE: end_of.setdefault(r, u)
  cands = [r for r in carry_afters if r.arg[-1] is AxisType.REDUCE and r in end_of and (int(r.vmax) + 1) % U == 0
           and int(r.vmax) + 1 > U]
  if not cands: return None
  r = max(cands, key=lambda x: x.arg[0])   # innermost (highest axis id)
  N = int(r.vmax) + 1
  end_r = end_of[r]
  final_state = end_r.src[0]                # the per-iteration store-chain result
  all_afters = carry_afters[r]
  afters = _true_carry_afters(tl, all_afters)
  if not afters: return None
  noncarry_afters = [a for a in all_afters if a not in afters]

  base_id = max((rr.arg[0] for rr in tl if rr.op is Ops.RANGE), default=0)
  # fresh outer range r2 of size N//U, same REDUCE axis-type, unique axis id
  r2 = UOp(Ops.RANGE, r.dtype, (r.src[0].const_like(N // U),), (base_id + 1000, AxisType.REDUCE))
  # inner ranges nested inside r's body must be DUPLICATED per copy (fresh ids) so the U copies are
  # independent loops -- otherwise the control-flow pass sees one inner range used in U places (CFG cycle).
  inner_ranges = [u for u in final_state.toposort() if u.op is Ops.RANGE and u is not r and r in u.ranges]

  # thread the carry across U copies
  carry = {a: _after_with_replaced_range(a, r, r2) for a in afters}   # u=0: AFTER(X, ..., r2)
  state = None
  for u in range(U):
    dvars: dict[UOp, UOp] = {r: r2 * U + u}
    for a in afters: dvars[a] = carry[a]
    for a in noncarry_afters: dvars[a] = _after_with_replaced_range(a, r, r2)
    for k, ir in enumerate(inner_ranges):   # fresh inner range per copy
      dvars[ir] = UOp(Ops.RANGE, ir.dtype, ir.src, (base_id + 2000 + u * 100 + k, ir.arg[-1]))
    state = final_state.substitute(dvars)
    carry = {a: _after_with_replaced_range(a, r, _last_store_to_root(state, _root_buf(a)))
             for a in afters}  # next copy reads after its matching prior-copy store

  new_end = UOp(Ops.END, end_r.dtype, (state,) + end_r.src[1:] if len(end_r.src) > 2 else (state, r2), end_r.arg)
  # if the END closed only r, replace with one closing r2; if it closed multiple ranges, swap r->r2 in them
  if len(end_r.src) > 2:
    new_end = UOp(Ops.END, end_r.dtype, (state,) + tuple(r2 if s is r else s for s in end_r.src[1:]), end_r.arg)
  return sink.substitute({end_r: new_end})


def unroll_recurrence(sink: UOp, U: int) -> UOp:
  if U <= 1: return sink
  prev = None
  out = sink
  # unroll one recurrence range per pass until none remain (handles a single innermost loop per call site)
  for _ in range(8):
    nxt = _unroll_one_range(out, U)
    if nxt is None or nxt is out: break
    out = nxt
    break   # v1: unroll only the innermost recurrence loop (the hot one); generalize later
  return out
