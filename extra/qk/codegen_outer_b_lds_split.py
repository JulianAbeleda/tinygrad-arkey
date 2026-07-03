#!/usr/bin/env python3
"""Outer-b independent split-combine codegen lowering for decode attention.

The slope-bending primitive the diagnose->solve loop named (SEARCH_BLOCKED_BY_CODEGEN): the generated block tile's
ctx-slope is the SERIAL outer-`b` block loop. Recurrence-unroll cannot reach `b` (no AFTER(_,b) carry edge) and a
serial b-unroll adds VGPRs onto an occupancy-pinned tile (refuted SCHED_UNROLL_SPLIT). This pass instead splits the
`b` REDUCE loop into K INDEPENDENT partitions over disjoint block sub-ranges, each with PRIVATE online-softmax state
+ private LDS tile (so the K long-latency chains overlap without a serial carry), then combines once with the flash
log-sum-exp merge. Default-off behind DECODE_OUTER_B_SPLIT=<K>; flag unset => byte-identical.

Scope: docs/decode-attention-outer-b-lds-split-combine-scope-20260627.md. Declines (returns the sink unchanged) on
any structure it does not positively recognize -- like axis_stride and the recurrence-unroll safety gates.
"""
from __future__ import annotations
from collections import defaultdict
from tinygrad import dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops, UOp, AxisType, AddrSpace

_F32 = dtypes.float32
_LOG2E = 1.4426950408889634

def _fexp(x: UOp) -> UOp:
  # match the kernel's online-softmax exp (DECODE_FAST_EXP2 aware) so the combine is numerically identical.
  arg = x * _LOG2E
  if getenv("DECODE_FAST_EXP2", 0): return UOp(Ops.CUSTOMI, arg.dtype, (arg,), arg="__builtin_amdgcn_exp2f({0})")
  return arg.exp2()

def _root(u: UOp) -> UOp:
  while u.op in (Ops.INDEX, Ops.AFTER): u = u.src[0]
  return u

def _dbg(*a):
  if getenv("DECODE_OUTER_B_SPLIT_DEBUG"): print("[OUTER_B_SPLIT]", *a)

def outer_b_split(sink: UOp, K: int) -> UOp:
  if K <= 1: return sink
  out = _split(sink, K)
  return out if out is not None else sink

def _split(sink: UOp, K: int) -> UOp | None:
  tl = list(sink.toposort())

  # 1. Find the b-loop END: the END over a const-bound REDUCE range that has EXACTLY 3 DEFINE_REG post-loop reads
  #    (acc/den/mx of the online softmax). Reads look like reg.after(END(state, b)).
  by_end: dict[UOp, list[UOp]] = defaultdict(list)
  for u in tl:
    if u.op is Ops.AFTER and len(u.src) >= 2 and _root(u.src[0]).op is Ops.DEFINE_REG:
      o = u.src[1]
      if (o.op is Ops.END and len(o.src) == 2 and o.src[1].op is Ops.RANGE
          and o.src[1].arg[-1] is AxisType.REDUCE and o.src[1].src[0].op is Ops.CONST):
        by_end[o].append(u)
  cand = [(o, rs) for o, rs in by_end.items() if len(rs) == 3]
  if len(cand) != 1:
    _dbg(f"decline: found {len(cand)} END-with-3-reg-reads candidates (want 1)"); return None
  mxu, finals = cand[0]
  b = mxu.src[1]
  NB = int(b.vmax) + 1
  if NB % K != 0 or NB <= K:
    _dbg(f"decline: NB={NB} not splittable by K={K}"); return None
  state_b = mxu.src[0]
  state_nodes = state_b.toposort()

  # 2. Classify the three carries. mx = the reg whose in-loop store value contains a MAX. acc = the post-loop read
  #    indexed by a RANGE (the PV array). den = the remaining scalar.
  def read_by_range(after_node: UOp):
    for s in tl:
      if s.op is Ops.INDEX and s.src[0] is after_node and s.src[1].op is Ops.RANGE:
        return s.src[1]
    return None
  # acc = the post-loop read indexed by a RANGE (the PV array). mx = among the rest, the one whose in-loop store
  # value is DIRECTLY a MAX (new_m); corr/p propagate that max into acc/den, so only the TOP op distinguishes it.
  def reg_store_top_max(reg_root: UOp) -> bool:
    return any(s.op is Ops.STORE and _root(s.src[0]) is reg_root and len(s.src) >= 2 and s.src[1].op is Ops.MAX
               for s in state_nodes)
  af = next((u for u in finals if read_by_range(u) is not None), None)
  rest = [u for u in finals if u is not af]
  mf = next((u for u in rest if reg_store_top_max(_root(u.src[0]))), None)
  lf = next((u for u in rest if u is not mf), None)
  if mf is None or af is None or lf is None:
    _dbg(f"decline: classify failed af={af is not None} lf={lf is not None} mf={mf is not None}"); return None
  acc_root, den_root, mx_root = _root(af.src[0]), _root(lf.src[0]), _root(mf.src[0])
  dd2 = read_by_range(af)
  R = int(dd2.vmax) + 1
  _dbg(f"b axis={b.arg[0]} NB={NB} K={K} acc={acc_root.arg} den={den_root.arg} mx={mx_root.arg} R={R}")

  # 3. Privatize set: every DEFINE_REG/DEFINE_LOCAL and every non-GLOBAL range inside the b body gets a fresh copy
  #    per partition, so the K partitions share NO mutable state (true independence -> latency overlap).
  priv_syms = [u for u in state_nodes if u.op in (Ops.DEFINE_REG, Ops.DEFINE_LOCAL)]
  inner_ranges = [u for u in state_nodes if u.op is Ops.RANGE and u is not b and u.arg[-1] is not AxisType.GLOBAL]
  max_reg = max([u.arg for u in tl if u.op in (Ops.DEFINE_REG, Ops.DEFINE_LOCAL) and isinstance(u.arg, int)], default=0)
  max_ax = max([u.arg[0] for u in tl if u.op is Ops.RANGE], default=0)

  def make_partition(k: int):
    dvars: dict[UOp, UOp] = {}
    rid = max_reg + 1 + k * 1000
    for j, s in enumerate(priv_syms): dvars[s] = UOp(s.op, s.dtype, s.src, rid + j)
    aid = max_ax + 1 + k * 1000
    for j, ir in enumerate(inner_ranges): dvars[ir] = UOp(Ops.RANGE, ir.dtype, ir.src, (aid + j, ir.arg[-1]))
    b_k = UOp(Ops.RANGE, b.dtype, (b.src[0].const_like(NB // K),), (aid + 900, AxisType.REDUCE))
    boff = b_k + (k * (NB // K))
    dvars[b] = boff
    state_k = state_b.substitute(dvars)
    # b appears in INDEX context (correctly offset) and as an ordering src of the dotp re-init AFTER. Substitute put
    # the offset expr into that ordering slot too -- restore the bare range there (ordering needs a RANGE).
    fix = {u: UOp(Ops.AFTER, u.dtype, (u.src[0],) + tuple(b_k if s is boff else s for s in u.src[1:]), u.arg)
           for u in state_k.toposort() if u.op is Ops.AFTER and any(s is boff for s in u.src[1:])}
    if fix: state_k = state_k.substitute(fix)
    end_k = UOp(Ops.END, mxu.dtype, (state_k, b_k), mxu.arg)
    return end_k, dvars[acc_root], dvars[den_root], dvars[mx_root]

  parts = [make_partition(k) for k in range(K)]
  end_ks = [p[0] for p in parts]
  mxk = [parts[k][3].after(end_ks[k])[0] for k in range(K)]
  denk = [parts[k][2].after(end_ks[k])[0] for k in range(K)]

  # 4. Combine (flash log-sum-exp merge), staged through fresh combine regs ordered after all K partitions.
  cid = max_reg + 1 + K * 1000
  M_expr = mxk[0]
  for k in range(1, K): M_expr = M_expr.maximum(mxk[k])
  Mreg = UOp.placeholder((1,), _F32, cid, addrspace=AddrSpace.REG)
  Ms = Mreg.after(*end_ks)[0].store(M_expr)
  M = Mreg.after(Ms)[0]
  wk = [_fexp(mxk[k] - M) for k in range(K)]
  den_expr = denk[0] * wk[0]
  for k in range(1, K): den_expr = den_expr + denk[k] * wk[k]
  denreg = UOp.placeholder((1,), _F32, cid + 1, addrspace=AddrSpace.REG)
  dens = denreg.after(Ms)[0].store(den_expr)
  ddc = UOp.range(R, max_ax + 1 + K * 1000 + 950)
  acc_expr = parts[0][1].after(end_ks[0])[ddc] * wk[0]
  for k in range(1, K): acc_expr = acc_expr + parts[k][1].after(end_ks[k])[ddc] * wk[k]
  accreg = UOp.placeholder((R,), _F32, cid + 2, addrspace=AddrSpace.REG)
  accs = accreg.after(dens)[ddc].store(acc_expr).end(ddc)

  out = sink.substitute({af: accreg.after(accs), lf: denreg.after(accs), mf: Mreg.after(accs)})
  _dbg(f"built K={K} split + combine (combine regs {cid}/{cid+1}/{cid+2})")
  return out
