"""Canonical generated score-broadcast kernel builders for decode attention.

These builders are used by both diagnostic probes and the default-off model route.
The current PV builder is chunked score recompute: it broadcasts one computed score
across a 32-column PV chunk, but each chunk recomputes q.k independently.
"""
from __future__ import annotations

from tinygrad import dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged

_F32 = dtypes.float32
_LOG2E = 1.4426950408889634

def _fc(v: float) -> UOp: return UOp.const(_F32, v)
def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * _LOG2E).exp2()

def score_once_state_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, L: int, S: int, Tc: int):
  if Hd % 64 != 0: raise ValueError(f"requires Hd divisible by 64, got {Hd}")
  G = Hq // Hkv; R = Hd // 32; RP = Hd // 64
  def kernel(state: UOp, q: UOp, cache: UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    j = UOp.range(L, 2, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    r = UOp.range(R, 3, axis_type=AxisType.REDUCE)
    e = lane * R + r
    klds = UOp.placeholder((Hd,), dtypes.half, 203, addrspace=AddrSpace.LOCAL)
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(dtypes.half), in_r).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 4, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    g = UOp.range(G, 5)
    h = kvh * G + g
    dot = UOp.placeholder((G,), _F32, 204, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 6)
    dot = dot.after(dot.after(kvh, s, j)[zi].store(0.0).end(zi))
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + e2].cast(dtypes.half), q[h * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (dot.after(rp)[g], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    dot_upd = dot[g].store(dot2).end(g).end(rp)
    dot_f = dot.after(dot_upd)
    l = UOp.placeholder((G,), _F32, 205, addrspace=AddrSpace.REG)
    m = UOp.placeholder((G,), _F32, 206, addrspace=AddrSpace.REG)
    zl = UOp.range(G, 7)
    init = l.after(kvh, s)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 8)
    init = m.after(init)[zm].store(-float("inf")).end(zm)
    l, m = l.after(init), m.after(init)
    gs = UOp.range(G, 9)
    old_m = m.after(j)[gs]
    sc = _warp_reduce_sum_staged(in_r.where(dot_f[gs] * (1.0 / (Hd ** 0.5)), _fc(-float("inf"))), lane, 32)
    mn = old_m.maximum(sc)
    upd = l[gs].store(l.after(j)[gs] * in_r.where(_fexp(old_m - mn), _fc(1.0)) + in_r.where(_fexp(sc - mn), _fc(0.0)), lane.eq(0))
    upd = m.after(upd)[gs].store(mn, lane.eq(0)).end(gs).end(j)
    g2 = UOp.range(G, 10)
    col = UOp.range(2, 11, AxisType.GLOBAL)
    val = col.eq(0).where(l.after(upd)[g2], m.after(upd)[g2])
    return state[((kvh * G + g2) * S + s) * 2 + col].store(val, lane.eq(0)).end(col, g2).end(kvh, s).sink(
      arg=_fki(f"flash_pall_score_once_state_{Hq}_{Hd}"))
  return kernel

def score_broadcast_pv_cols_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, L: int, S: int, Tc: int, Wp: int, C0: int=0):
  if Hd % 64 != 0: raise ValueError(f"requires Hd divisible by 64, got {Hd}")
  G = Hq // Hkv; R = Hd // 32; RP = Hd // 64
  def kernel(pv: UOp, state: UOp, q: UOp, cache: UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    j = UOp.range(L, 2, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    r = UOp.range(R, 3, axis_type=AxisType.REDUCE)
    e = lane * R + r
    klds = UOp.placeholder((Hd,), dtypes.half, 207, addrspace=AddrSpace.LOCAL)
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(dtypes.half), in_r).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 4, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    g = UOp.range(G, 5)
    h = kvh * G + g
    dot = UOp.placeholder((G,), _F32, 208, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 6)
    dot = dot.after(dot.after(kvh, s, j)[zi].store(0.0).end(zi))
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + e2].cast(dtypes.half), q[h * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (dot.after(rp)[g], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    dot_upd = dot[g].store(dot2).end(g).end(rp)
    dot_f = dot.after(dot_upd)
    l = UOp.placeholder((G,), _F32, 209, addrspace=AddrSpace.REG)
    m = UOp.placeholder((G,), _F32, 210, addrspace=AddrSpace.REG)
    acc = UOp.placeholder((G * Wp,), _F32, 211, addrspace=AddrSpace.REG)
    zl = UOp.range(G, 7)
    init = l.after(kvh, s)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 8)
    init = m.after(init)[zm].store(-float("inf")).end(zm)
    zc = UOp.range(Wp, 9)
    zg = UOp.range(G, 10)
    init = acc.after(init)[zg * Wp + zc].store(0.0).end(zg, zc)
    l, m, acc = l.after(init), m.after(init), acc.after(init)
    gs = UOp.range(G, 11)
    old_m = m.after(j)[gs]
    sc = _warp_reduce_sum_staged(in_r.where(dot_f[gs] * (1.0 / (Hd ** 0.5)), _fc(-float("inf"))), lane, 32)
    mn = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - mn), _fc(1.0))
    p = in_r.where(_fexp(sc - mn), _fc(0.0))
    c = UOp.range(Wp, 12, AxisType.LOCAL)
    gd = gs * Wp + c
    vd = cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + C0 + c].cast(_F32)
    upd = acc[gd].store(acc.after(j)[gd] * corr + p * vd, lane.eq(0)).end(c)
    upd = l.after(upd)[gs].store(l.after(j)[gs] * corr + p, lane.eq(0))
    upd = m.after(upd)[gs].store(mn, lane.eq(0)).end(gs).end(j)
    g2 = UOp.range(G, 13)
    c2 = UOp.range(Wp, 14, AxisType.GLOBAL)
    return pv[((kvh * G + g2) * S + s) * Wp + c2].store(acc.after(upd)[g2 * Wp + c2], lane.eq(0)).end(c2, g2).end(kvh, s).sink(
      arg=_fki(f"flash_pall_score_broadcast_pv_cols_{C0}_{Wp}_{Hq}_{Hd}"))
  return kernel
