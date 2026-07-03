from __future__ import annotations
from extra.qk.flash_common import _F32, _fexp, _fc, _fki, _ceildiv, Tensor, dtypes, getenv, AddrSpace, AxisType, KernelInfo, Ops, UOp  # noqa: F401
from extra.qk.kv_load import make_kv_element_loader  # noqa: F401
"""Generated UOp flash-decode kernel builders (the block-tile live-split kernel + score/pv/combine/partial variants). No handwritten kernels here -- pure UOp construction. Re-exported by flash_decode.py for external importers."""

def flash_score_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, Tc, use_vdot2:bool=False):
  G = Hq // Hkv
  def kernel(score:UOp, q:UOp, cache:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    e = UOp.range(Hd, 2, axis_type=AxisType.REDUCE)
    kv = h // G
    acc = UOp.placeholder((1,), _F32, 120, addrspace=AddrSpace.REG)
    acc = acc.after(h, t)[0].set(0.0)
    qv = q[h * Hd + e].cast(_F32)
    kvv = cache[((0 * Hkv + kv) * MAXC + t) * Hd + e].cast(_F32)
    acc = acc[0].set(acc.after(e)[0] + qv * kvv, end=e)
    return score[h * MAXC + t].store(acc[0] * (1.0 / (Hd ** 0.5))).end(h, t).sink(
      arg=_fki(f"flash_score_whole_cache{'_vdot2' if use_vdot2 else ''}_{Hq}_{Hd}"))
  return kernel

def flash_score_whole_cache_xlane_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, Tc):
  if Hd % 32 != 0: raise ValueError(f"x-lane score requires Hd divisible by 32, got {Hd}")
  G = Hq // Hkv; elems_per_lane = Hd // 32
  def kernel(score:UOp, q:UOp, cache:UOp) -> UOp:
    from extra.qk.lane_partition_reduce import LanePartition, lane_partition_reduce_sum
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    r = UOp.range(elems_per_lane, 2, axis_type=AxisType.REDUCE)
    kv = h // G
    e = lane * elems_per_lane + r
    acc = UOp.placeholder((1,), _F32, 122, addrspace=AddrSpace.REG)
    acc = acc.after(h, t)[0].set(0.0)
    qv = q[h * Hd + e].cast(_F32)
    kvv = cache[((0 * Hkv + kv) * MAXC + t) * Hd + e].cast(_F32)
    acc = acc[0].set(acc.after(r)[0] + qv * kvv, end=r)
    total = lane_partition_reduce_sum(acc[0], LanePartition(lane))
    return score[h * MAXC + t].store(total * (1.0 / (Hd ** 0.5)), lane.eq(0)).end(h, t).sink(
      arg=_fki(f"flash_score_whole_cache_xlane_{Hq}_{Hd}"))
  return kernel


def flash_p1_crosslane_score_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, Tc):
  """Generated route-level P1 score kernel: lanes shard Hd and ds_bpermute-reduce q.k.

  This integrates the LaneMap/CrossLane primitive into the actual generated whole-cache decode route.
  It intentionally does not claim LDS or v_dot2; the primitive gap gate must verify emitted ISA before promotion.
  """
  if Hd % 32 != 0: raise ValueError(f"P1 crosslane score requires Hd divisible by 32, got {Hd}")
  G = Hq // Hkv; R = Hd // 32
  def kernel(score:UOp, q:UOp, cache:UOp) -> UOp:
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    from extra.qk.amd_warp_reduce import warp_reduce_sum
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    e = lane * R + r
    kvh = h // G
    acc = UOp.placeholder((1,), _F32, 171, addrspace=AddrSpace.REG)
    acc_init = acc.after(h, t)[0].store(0.0)
    acc = acc.after(acc_init)
    qv = q[h * Hd + e].cast(_F32)
    kv = cache[((0 * Hkv + kvh) * MAXC + t) * Hd + e].cast(_F32)
    acc_upd = acc[0].store(acc.after(r)[0] + qv * kv).end(r)
    total = _warp_reduce_sum_staged(acc.after(acc_upd)[0], lane, 32)
    return score[h * MAXC + t].store(total * (1.0 / (Hd ** 0.5)), lane.eq(0)).end(h, t).sink(
      arg=_fki(f"flash_p1_crosslane_score_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_pall_lds_crosslane_fdot2_score_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, Tc):
  """Generated route-level PALL score kernel: LDS-staged K + lane-sharded/cross-lane q.k + fdot2.

  This is the composed physical score hot path. It does not claim full decode lifecycle fusion; online state and PV
  still run through the existing generated split pipeline until a full physical-tile lifecycle route is built.
  """
  if Hd % 64 != 0: raise ValueError(f"PALL score requires Hd divisible by 64, got {Hd}")
  G = Hq // Hkv; R = Hd // 32; RP = Hd // 64
  def kernel(score:UOp, q:UOp, cache:UOp) -> UOp:
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    e = lane * R + r
    kvh = h // G
    klds = UOp.placeholder((Hd,), dtypes.half, 172, addrspace=AddrSpace.LOCAL)
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t) * Hd + e].cast(dtypes.half)).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 3, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    acc = UOp.placeholder((1,), _F32, 173, addrspace=AddrSpace.REG)
    init = acc.after(h, t)[0].store(0.0)
    acc = acc.after(init)
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + e2].cast(dtypes.half), q[h * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (acc.after(rp)[0], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    upd = acc[0].store(dot2).end(rp)
    total = _warp_reduce_sum_staged(acc.after(upd)[0], lane, 32)
    return score[h * MAXC + t].store(total * (1.0 / (Hd ** 0.5)), lane.eq(0)).end(h, t).sink(
      arg=_fki(f"flash_pall_lds_crosslane_score_{Hq}_{Hd}"))
  return kernel

def flash_pall_score_state_pv_lifecycle_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Generated PALL lifecycle route: physical q.k score + online state + PV in one tile.

  This keeps the route default-off. It proves generated lifecycle fusion with LDS, cross-lane reduction, and fdot2, but
  still recomputes q.k per output column because score reuse across the PV column axis is not expressible yet.
  """
  if Hd % 64 != 0: raise ValueError(f"PALL lifecycle requires Hd divisible by 64, got {Hd}")
  G = Hq // Hkv; W = Hd + 2; R = Hd // 32; RP = Hd // 64
  def kernel(pout:UOp, q:UOp, cache:UOp) -> UOp:
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    is_v = d < Hd
    is_l = d.eq(Hd)
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    r = UOp.range(R, 4, axis_type=AxisType.REDUCE)
    e = lane * R + r
    klds = UOp.placeholder((Hd,), dtypes.half, 174, addrspace=AddrSpace.LOCAL)
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(dtypes.half), in_r).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 5, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    g_dot = UOp.range(G, 6)
    h_dot = kvh * G + g_dot
    dot = UOp.placeholder((G,), _F32, 175, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 7)
    dot_init = dot.after(kvh, s, d, j)[zi].store(0.0).end(zi)
    dot = dot.after(dot_init)
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h_dot * Hd + e2].cast(dtypes.half), q[h_dot * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (dot.after(rp)[g_dot], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    dot_upd = dot[g_dot].store(dot2).end(g_dot).end(rp)
    dot_f = dot.after(dot_upd)
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((G,), _F32, 176, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), _F32, 177, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), _F32, 178, addrspace=AddrSpace.REG)
    za = UOp.range(G, 8)
    init = acc.after(kvh, s, d)[za].store(0.0).end(za)
    zl = UOp.range(G, 9)
    init = den.after(init)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 10)
    init = mx.after(init)[zm].store(-float("inf")).end(zm)
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    g_state = UOp.range(G, 11)
    old_m = mx.after(j)[g_state]
    sc_lane = in_r.where(dot_f[g_state] * (1.0 / (Hd ** 0.5)), _fc(-float("inf")))
    sc = _warp_reduce_sum_staged(sc_lane, lane, 32)
    new_m = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
    p = in_r.where(_fexp(sc - new_m), _fc(0.0))
    upd = acc[g_state].store(acc.after(j)[g_state] * corr + p * vd, lane.eq(0))
    upd = den.after(upd)[g_state].store(den.after(j)[g_state] * corr + p, lane.eq(0))
    upd = mx.after(upd)[g_state].store(new_m, lane.eq(0)).end(g_state).end(j)
    g2 = UOp.range(G, 12)
    af, lf, mf = acc.after(upd), den.after(upd), mx.after(upd)
    val = is_v.where(af[g2], is_l.where(lf[g2], mf[g2]))
    return pout[((kvh * G + g2) * S + s) * W + d].store(val, lane.eq(0)).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_pall_score_state_pv_lifecycle_{Hq}_{Hd}"))
  return kernel

def flash_tile_placeholder_kernel(Hd:int, Hq:int, MAXC:int, Tc):
  def kernel(out:UOp, score:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    return out[h * MAXC + t].store(score[h * MAXC + t]).end(h, t).sink(
      arg=_fki(f"flash_tile_placeholder_{Hq}_{Hd}"))
  return kernel

def flash_tile_score_max_kernel(Hd:int, Hq:int, MAXC:int, L:int, S:int, Tc):
  def kernel(pm:UOp, score:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    j = UOp.range(L, 2, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    m = UOp.placeholder((1,), _F32, 125, addrspace=AddrSpace.REG)
    m = m.after(h, s)[0].set(-float("inf"))
    sv = score[h * MAXC + t_safe]
    v = in_r.where(sv, sv.const_like(-float("inf")))
    m = m[0].set(m.after(j)[0].maximum(v), end=j)
    return pm[h * S + s].store(m[0]).end(h, s).sink(arg=_fki(f"flash_tile_score_max_{Hq}_{Hd}"))
  return kernel

def flash_tile_prob_kernel(Hd:int, Hq:int, MAXC:int, L:int, S:int, Tc):
  def kernel(prob:UOp, pm:UOp, score:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    j = UOp.range(L, 2, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    p = in_r.where(_fexp(score[h * MAXC + t_safe] - pm[h * S + s]), UOp.const(_F32, 0.0))
    return prob[h * MAXC + t_safe].store(p, in_r).end(h, s, j).sink(arg=_fki(f"flash_tile_prob_{Hq}_{Hd}"))
  return kernel

def flash_partial_coop_vec_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, prob:UOp, cache:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 121, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    p = in_r.where(prob[(kvh * G + g) * MAXC + t], _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_partial_coop_vec_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_tile_partial_pv_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, prob:UOp, cache:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 127, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    p = in_r.where(prob[(kvh * G + g) * MAXC + t], _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_tile_partial_pv_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_tile_prob_partial_pv_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, pm:UOp, score:UOp, cache:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 128, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    h = kvh * G + g
    p = in_r.where(_fexp(score[h * MAXC + t_safe] - pm[h * S + s]), _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_tile_prob_partial_pv_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_online_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """P2 structural generated online-softmax+PV tile skeleton.

  This intentionally shares A3.10's external per-split max input for correctness while giving the primitive-complete
  path a distinct route/program identity. P3/P4 are responsible for replacing this structural shell with true lane-owned
  register-resident online state and packed-dot/reduction lowerings.
  """
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, pm:UOp, score:UOp, cache:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 129, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    h = kvh * G + g
    p = in_r.where(_fexp(score[h * MAXC + t_safe] - pm[h * S + s]), _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_online_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_online_state_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """P5 structural online-state+PV tile.

  Output width W=Hd+2: [0:Hd)=unnormalized PV, Hd=per-split l, Hd+1=per-split m. This moves per-split m/l state
  into the tile lifecycle so later cross-lane/dot lowerings have a real in-tile dataflow site to bind.
  """
  G = Hq // Hkv; W = Hd + 2
  def kernel(pout:UOp, score:UOp, cache:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    is_l = d.eq(Hd)
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 130, addrspace=AddrSpace.REG)
    l = UOp.placeholder((G,), _F32, 131, addrspace=AddrSpace.REG)
    m = UOp.placeholder((G,), _F32, 132, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4)
    init = c[zi].store(0.0).end(zi)
    zi2 = UOp.range(G, 5)
    init = l.after(init)[zi2].store(0.0).end(zi2)
    zi3 = UOp.range(G, 6)
    init = m.after(init)[zi3].store(-float("inf")).end(zi3)
    c, l, m = c.after(init), l.after(init), m.after(init)
    g = UOp.range(G, 7)
    h = kvh * G + g
    old_m = m.after(j)[g]
    sc = in_r.where(score[h * MAXC + t_safe], old_m)
    mn = in_r.where(old_m.maximum(sc), old_m)
    corr = in_r.where(_fexp(old_m - mn), _fc(1.0))
    p = in_r.where(_fexp(sc - mn), _fc(0.0))
    upd = c[g].store(c.after(j)[g] * corr + p * vd)
    upd = l.after(upd)[g].store(l.after(j)[g] * corr + p)
    upd = m.after(upd)[g].store(mn).end(g).end(j)
    g2 = UOp.range(G, 8)
    cf, lf, mf = c.after(upd), l.after(upd), m.after(upd)
    val = is_v.where(cf[g2], is_l.where(lf[g2], mf[g2]))
    return pout[((kvh * G + g2) * S + s) * W + d].store(val).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_online_state_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_state_gmax_kernel(Hd:int, Hq:int, S, stride=None):
  # N3F: stride (partials layout, = compiled Smax) is decoupled from S (reduce count, = dynamic valid splits).
  W = Hd + 2; M_COL = Hd + 1; stride = S if stride is None else stride
  def kernel(gm:UOp, pout:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 133, addrspace=AddrSpace.REG)
    g = g.after(h)[0].set(-1e30)
    g = g[0].set(g.after(s)[0].maximum(pout[(h * stride + s) * W + M_COL]), end=s)
    return gm[h].store(g[0]).end(h).sink(arg=_fki(f"flash_state_gmax_{Hq}_{Hd}"))
  return kernel

def flash_state_combine_kernel(Hd:int, Hq:int, S, stride=None):
  # N3F: stride (partials layout, = compiled Smax) is decoupled from S (reduce count, = dynamic valid splits).
  W = Hd + 2; L_COL = Hd; M_COL = Hd + 1; stride = S if stride is None else stride
  def kernel(out:UOp, pout:UOp, gm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    gm_h = gm[h]
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    w = _fexp(pout[(h * stride + s) * W + M_COL] - gm_h)
    num = UOp.placeholder((1,), _F32, 134, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 135, addrspace=AddrSpace.REG)
    num = num.after(h, d)[0].set(0.0)
    den = den.after(h, d)[0].set(0.0)
    upd = num[0].store(num.after(s)[0] + w * pout[(h * stride + s) * W + d])
    upd = den.after(upd)[0].store(den.after(s)[0] + w * pout[(h * stride + s) * W + L_COL]).end(s)
    nf, df = num.after(upd)[0], den.after(upd)[0]
    return out[h * Hd + d].store(nf / df).end(h, d).sink(arg=_fki(f"flash_state_combine_{Hq}_{Hd}"))
  return kernel

def flash_fused_state_combine_kernel(Hd:int, Hq:int, S, stride=None):
  """P5 missing-primitive: fuse the global-max into the combine -> ONE dispatch instead of gmax+combine, and no gm
  buffer round-trip. Each (h,d) thread computes gm[h]=max_s M[s] inline (pass 1) then the log-sum-exp rescale
  (pass 2). M is read twice but in-kernel (no global gm buffer). Default-off (DECODE_ATTN_FUSED_COMBINE=1).
  docs/decode-attention-owned-lifecycle-missing-primitives-scope-20260627.md."""
  W = Hd + 2; L_COL = Hd; M_COL = Hd + 1; stride = S if stride is None else stride
  def kernel(out:UOp, pout:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    s1 = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 136, addrspace=AddrSpace.REG)
    g = g.after(h, d)[0].set(-1e30)
    g = g[0].set(g.after(s1)[0].maximum(pout[(h * stride + s1) * W + M_COL]), end=s1)   # pass 1: inline gmax
    gm_h = g[0]
    s2 = UOp.range(S, 3, axis_type=AxisType.REDUCE)
    w = _fexp(pout[(h * stride + s2) * W + M_COL] - gm_h)
    num = UOp.placeholder((1,), _F32, 137, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 138, addrspace=AddrSpace.REG)
    num = num.after(g)[0].set(0.0); den = den.after(g)[0].set(0.0)
    upd = num[0].store(num.after(s2)[0] + w * pout[(h * stride + s2) * W + d])
    upd = den.after(upd)[0].store(den.after(s2)[0] + w * pout[(h * stride + s2) * W + L_COL]).end(s2)   # pass 2: rescale+norm
    nf, df = num.after(upd)[0], den.after(upd)[0]
    return out[h * Hd + d].store(nf / df).end(h, d).sink(arg=_fki(f"flash_fused_state_combine_{Hq}_{Hd}"))
  return kernel

def flash_pall_score_broadcast_combine4_kernel(Hd:int, Hq:int, S):
  CH = 32
  def kernel(out:UOp, state:UOp, pv0:UOp, pv1:UOp, pv2:UOp, pv3:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    gm = UOp.placeholder((1,), _F32, 179, addrspace=AddrSpace.REG)
    gm = gm.after(h)[0].set(-float("inf"))
    gm = gm[0].set(gm.after(s)[0].maximum(state[(h * S + s) * 2 + 1]), end=s)
    gm_f = gm[0]
    num = UOp.placeholder((1,), _F32, 180, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 181, addrspace=AddrSpace.REG)
    num = num.after(h, d)[0].set(0.0)
    den = den.after(h, d)[0].set(0.0)
    s2 = UOp.range(S, 3, axis_type=AxisType.REDUCE)
    c = d % CH
    pv = (d < CH).where(pv0[(h * S + s2) * CH + c],
      (d < CH * 2).where(pv1[(h * S + s2) * CH + c],
      (d < CH * 3).where(pv2[(h * S + s2) * CH + c], pv3[(h * S + s2) * CH + c])))
    w = _fexp(state[(h * S + s2) * 2 + 1] - gm_f)
    upd = num[0].store(num.after(s2)[0] + w * pv)
    upd = den.after(upd)[0].store(den.after(s2)[0] + w * state[(h * S + s2) * 2]).end(s2)
    return out[h * Hd + d].store(num.after(upd)[0] / den.after(upd)[0]).end(h, d).sink(
      arg=_fki(f"flash_pall_score_broadcast_combine4_{Hq}_{Hd}"))
  return kernel

def flash_online_state_pv_tile_xlane_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """P7 token-sharded online-state+PV tile.

  Grid owns (kvh, split, output column). Local wave lanes shard the token loop and cross-lane merge online-softmax
  state. Output width W=Hd+2: PV, per-split l, per-split m.
  """
  G = Hq // Hkv; W = Hd + 2; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(pout:UOp, score:UOp, cache:UOp) -> UOp:
    from extra.qk.amd_warp_reduce import warp_reduce_max
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    is_v = d < Hd
    is_l = d.eq(Hd)
    r = UOp.range(R, 3, axis_type=AxisType.REDUCE)
    j = r * LANES + lane
    t = s * L + j; in_r = (j < L) & (t < Tc)
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    # d is a GLOBAL output-column axis in this x-lane route. Keep recurrence
    # registers indexed by both query-group and output column; otherwise
    # independent PV/state columns alias the same REG slots and corrupt the
    # online update before the lane merge.
    c = UOp.placeholder((G * W,), _F32, 136, addrspace=AddrSpace.REG)
    l = UOp.placeholder((G * W,), _F32, 137, addrspace=AddrSpace.REG)
    m = UOp.placeholder((G * W,), _F32, 138, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4)
    init = c[zi * W + d].store(0.0).end(zi)
    zi2 = UOp.range(G, 5)
    init = l.after(init)[zi2 * W + d].store(0.0).end(zi2)
    zi3 = UOp.range(G, 6)
    init = m.after(init)[zi3 * W + d].store(-float("inf")).end(zi3)
    c, l, m = c.after(init), l.after(init), m.after(init)
    g = UOp.range(G, 7)
    h = kvh * G + g
    gd = g * W + d
    old_m = m.after(r)[gd]
    sc = in_r.where(score[h * MAXC + t_safe], old_m)
    mn = in_r.where(old_m.maximum(sc), old_m)
    corr = in_r.where(_fexp(old_m - mn), _fc(1.0))
    p = in_r.where(_fexp(sc - mn), _fc(0.0))
    upd = c[gd].store(c.after(r)[gd] * corr + p * vd)
    upd = l.after(upd)[gd].store(l.after(r)[gd] * corr + p)
    upd = m.after(upd)[gd].store(mn).end(g).end(r)
    g2 = UOp.range(G, 8)
    gd2 = g2 * W + d
    cf, lf, mf = c.after(upd), l.after(upd), m.after(upd)
    gm = warp_reduce_max(mf[gd2], lane, LANES, 90)
    w = _fexp(mf[gd2] - gm)
    acc_all = _warp_reduce_sum_staged(cf[gd2] * w, lane, LANES, 96)
    l_all = _warp_reduce_sum_staged(lf[gd2] * w, lane, LANES, 102)
    val = is_v.where(acc_all, is_l.where(l_all, gm))
    return pout[((kvh * G + g2) * S + s) * W + d].store(val, lane.eq(0)).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_online_state_pv_tile_xlane_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_xlane_split_m_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(pm:UOp, score:UOp) -> UOp:
    from extra.qk.amd_warp_reduce import warp_reduce_max
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    j = r * LANES + lane
    t = s * L + j
    in_r = (j < L) & (t < Tc)
    t_safe = in_r.where(t, t.const_like(0))
    g = UOp.range(G, 3)
    h = kvh * G + g
    sc = in_r.where(score[h * MAXC + t_safe], _fc(-float("inf")))
    m = UOp.placeholder((G,), _F32, 145, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4)
    init = m[zi].store(-float("inf")).end(zi)
    m = m.after(init)
    upd = m[g].store(m.after(r)[g].maximum(sc)).end(g).end(r)
    g2 = UOp.range(G, 5)
    gm = warp_reduce_max(m.after(upd)[g2], lane, LANES, 90)
    h2 = kvh * G + g2
    return pm[h2 * S + s].store(gm, lane.eq(0)).end(g2).end(kvh, s).sink(arg=_fki(f"flash_xlane_split_m_{Hq}_{Hd}"))
  return kernel

def flash_xlane_pv_from_m_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; W = Hd + 1; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(pv:UOp, pm:UOp, score:UOp, cache:UOp) -> UOp:
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    is_v = d < Hd
    r = UOp.range(R, 3, axis_type=AxisType.REDUCE)
    j = r * LANES + lane
    t = s * L + j
    in_r = (j < L) & (t < Tc)
    t_safe = in_r.where(t, t.const_like(0))
    g = UOp.range(G, 4)
    h = kvh * G + g
    p = in_r.where(_fexp(score[h * MAXC + t_safe] - pm[h * S + s]), _fc(0.0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((G * W,), _F32, 147, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 5)
    init = acc[zi * W + d].store(0.0).end(zi)
    acc = acc.after(init)
    gd = g * W + d
    upd = acc[gd].store(acc.after(r)[gd] + p * vd).end(g).end(r)
    g2 = UOp.range(G, 6)
    part = _warp_reduce_sum_staged(acc.after(upd)[g2 * W + d], lane, LANES, 90)
    h2 = kvh * G + g2
    return pv[(h2 * S + s) * W + d].store(part, lane.eq(0)).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_xlane_pv_from_m_{Hq}_{Hd}"))
  return kernel

def flash_fused_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Generated fused PV tile candidate.

  This is the first standalone rung for the pure-generated fused-PV project. It
  is deliberately not wired into decode routing yet. Compared with the refuted
  split x-lane PV route, the important structural change is that output column
  d is LOCAL/cooperative, not a GLOBAL output-column grid axis. The kernel owns
  a whole (kv-head, split) tile, reuses V for the G query heads in registers,
  and emits compact augmented PV partials: columns [0:Hd) are PV, column Hd is
  the denominator contribution.

  Inputs:
    pm    : [Hq, S] per-split max
    score : [Hq, MAXC] precomputed q.k scores
    cache : whole cache [2, 1, Hkv, MAXC, Hd] flattened

  Output:
    pout  : [Hq, S, Hd+1]
  """
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, pm:UOp, score:UOp, cache:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 151, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4)
    c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    h = kvh * G + g
    p = in_r.where(_fexp(score[h * MAXC + t_safe] - pm[h * S + s]), _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6)
    fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_fused_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_fused_score_state_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Generated fused score + online-state + PV tile candidate.

  Standalone rung only. This intentionally has a distinct program identity from
  the route-clean fused-PV-only candidate so the gates can prove whether score
  and split max/state were actually pulled into the tile lifecycle.

  Output width W=Hd+2:
    [0:Hd) -> unnormalized PV accumulator
    Hd     -> split denominator l
    Hd+1   -> split max m
  """
  G = Hq // Hkv; W = Hd + 2
  def kernel(pout:UOp, q:UOp, cache:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    is_l = d.eq(Hd)
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    e = UOp.range(Hd, 4, axis_type=AxisType.REDUCE)
    g_dot = UOp.range(G, 5)
    h_dot = kvh * G + g_dot
    dot = UOp.placeholder((G,), _F32, 152, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 6)
    dot_init = dot.after(kvh, s, d, j)[zi].store(0.0).end(zi)
    dot = dot.after(dot_init)
    qv = q[h_dot * Hd + e].cast(_F32)
    kvv = cache[0, 0, kvh, t_safe, e].cast(_F32)
    dot_upd = dot[g_dot].store(dot.after(e)[g_dot] + qv * kvv).end(g_dot).end(e)
    dot_f = dot.after(dot_upd)
    vd = is_v.where(cache[1, 0, kvh, t_safe, is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((G,), _F32, 153, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), _F32, 154, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), _F32, 155, addrspace=AddrSpace.REG)
    za = UOp.range(G, 7)
    init = acc.after(kvh, s, d)[za].store(0.0).end(za)
    zl = UOp.range(G, 8)
    init = den.after(init)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 9)
    init = mx.after(init)[zm].store(-float("inf")).end(zm)
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    g_state = UOp.range(G, 10)
    old_m = mx.after(j)[g_state]
    sc = in_r.where(dot_f[g_state] * (1.0 / (Hd ** 0.5)), _fc(-float("inf")))
    new_m = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
    p = in_r.where(_fexp(sc - new_m), _fc(0.0))
    upd = acc[g_state].store(acc.after(j)[g_state] * corr + p * vd)
    upd = den.after(upd)[g_state].store(den.after(j)[g_state] * corr + p)
    upd = mx.after(upd)[g_state].store(new_m).end(g_state).end(j)
    g2 = UOp.range(G, 11)
    af, lf, mf = acc.after(upd), den.after(upd), mx.after(upd)
    val = is_v.where(af[g2], is_l.where(lf[g2], mf[g2]))
    return pout[((kvh * G + g2) * S + s) * W + d].store(val).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_fused_score_state_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_fused_xlane_score_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Fused score(once) + online-state + d-sharded PV tile (the physically-fast fused tile).

  One 32-lane warp owns one (kvh, split). Per token: lanes e-shard the q.k dot (LDS-staged K + fdot2 +
  cross-lane reduce -> ONE score per token, reused across all PV columns), online-softmax update, then the
  same lanes d-shard the PV (each owns Hd/32 output columns). Buffer-identity clean: indexes the raw
  [2,1,Hkv,MAXC,Hd] cache, no full-cache copy. Output W=Hd+2: [0:Hd) PV partial, Hd l, Hd+1 m -- consumed
  by flash_state_gmax + flash_state_combine. Occupancy is set by the split count S (see
  docs/decode-fused-tile-occupancy-roofline-baseline.md); layout validated by
  extra/qk/decode_attention_fused_xlane_score_pv_microgate.py.
  """
  if Hd % 64 != 0: raise ValueError(f"fused xlane score+PV requires Hd%%64==0, got {Hd}")
  G = Hq // Hkv; W = Hd + 2; LANES = 32; R = Hd // LANES; RP = Hd // 64
  scale = 1.0 / (Hd ** 0.5)
  def kernel(pout:UOp, q:UOp, cache:UOp) -> UOp:
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    from extra.qk.amd_warp_reduce import warp_reduce_sum
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    acc = UOp.placeholder((G * R,), _F32, 220, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), _F32, 221, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), _F32, 222, addrspace=AddrSpace.REG)
    za = UOp.range(G * R, 2)
    init = acc.after(kvh, s)[za].store(0.0).end(za)
    zl = UOp.range(G, 3)
    init = den.after(init)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 4)
    init = mx.after(init)[zm].store(-float("inf")).end(zm)
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    j = UOp.range(L, 5, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    klds = UOp.placeholder((Hd,), dtypes.half, 225, addrspace=AddrSpace.LOCAL)
    k_upcast = bool(getenv("REG_STORE_DEVEC"))
    rk = UOp.range(R, 6, axis_type=AxisType.UPCAST if k_upcast else AxisType.REDUCE)
    ek = lane * R + rk
    kstage_val = cache[0, 0, kvh, t_safe, ek].cast(dtypes.half)
    kstage = (klds[ek].store(kstage_val).end(rk) if k_upcast else klds[ek].store(kstage_val, in_r).end(rk))
    bar = UOp.barrier(UOp.group(kstage))
    g = UOp.range(G, 7)
    h = kvh * G + g
    dotp = UOp.placeholder((1,), _F32, 226, addrspace=AddrSpace.REG)
    dinit = dotp.after(kvh, s, j, g)[0].store(0.0)
    dotp = dotp.after(dinit)
    rp = UOp.range(RP, 8, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + e2].cast(dtypes.half), q[h * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (dotp.after(rp)[0], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    dupd = dotp[0].store(dot2).end(rp)
    partial = dotp.after(dupd)[0]
    sc_full = (warp_reduce_sum(partial, lane, LANES) if getenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE", 0)
               else _warp_reduce_sum_staged(partial, lane, LANES)) * scale
    sc = in_r.where(sc_full, _fc(-float("inf")))
    old_m = mx.after(j)[g]
    new_m = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
    p = in_r.where(_fexp(sc - new_m), _fc(0.0))
    dd = UOp.range(R, 9, axis_type=AxisType.UPCAST if getenv("REG_STORE_DEVEC") else AxisType.REDUCE)
    d = lane * R + dd
    vd = cache[1, 0, kvh, t_safe, d].cast(_F32)
    accu = acc[g * R + dd].store(acc.after(j)[g * R + dd] * corr + p * vd).end(dd)
    denu = den.after(accu)[g].store(den.after(j)[g] * corr + p)
    mxu = mx.after(denu)[g].store(new_m).end(g).end(j)
    af, lf, mf = acc.after(mxu), den.after(mxu), mx.after(mxu)
    g2 = UOp.range(G, 10)
    base = ((kvh * G + g2) * S + s) * W
    dd2 = UOp.range(R, 11)
    d2 = lane * R + dd2
    pv = pout[base + d2].store(af[g2 * R + dd2]).end(dd2)
    ls = pout.after(pv)[base + Hd].store(lf[g2], lane.eq(0))
    ms = pout.after(ls)[base + (Hd + 1)].store(mf[g2], lane.eq(0)).end(g2)
    return ms.end(kvh, s).sink(arg=_fki(f"flash_fused_xlane_score_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc, staging:str="KV_BOTH", quant:bool=False, rope:bool=False):
  """Block-tiled generated decode candidate.

  Mirrors the owned tile's topology at the UOp level: one workgroup per (kvh, split), G warps per
  workgroup, one warp per GQA query head, TK=16 K rows staged in LDS, then online softmax + d-sharded PV.
  staging="KV_BOTH" (default): both K and V staged in LDS (original behavior, 8KB LDS).
  staging="K_ONLY": K staged in LDS (4KB), V read directly from global cache (L2-warmed by E_49152).
  This is default-off and guarded by extra/qk/decode_attention_block_tile_microgate.py before route use.

  quant=False (default): `cache` is fp16 K/V, read directly (byte-identical to the shipped route).
  quant=True: `cache` is INT8 K/V and an extra `scale` buffer (fp16, shape [2,1,Hkv,MAXC]) is bound after `cache`;
    each element is dequantized IN-REGISTER as int8*scale[kv,head,token] at the load site -- no materialized fp16 KV
    (the buffer stays int8-sized), model-agnostic (keys off the KV shape). Scale is per-(K|V, kv_head, token),
    symmetric absmax over head_dim. This is the fused-dequant path for the KV-quant long-context tier.
  """
  if Hd % 64 != 0: raise ValueError(f"block tile requires Hd%%64==0, got {Hd}")
  G = Hq // Hkv; W = Hd + 2; LANES = 32; WARPS = G; THREADS = LANES * WARPS; TK = 16
  R = Hd // LANES; RP = Hd // 64; STAGES = _ceildiv(TK * Hd, THREADS); NB = _ceildiv(L, TK)
  scale = 1.0 / (Hd ** 0.5)
  def kernel(pout:UOp, q:UOp, cache:UOp, *extra) -> UOp:
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    from extra.qk.amd_warp_reduce import warp_reduce_sum
    # Optional extra input buffers, bound in this fixed order by the wrapper: [kvscale] if quant, [freqs] if rope.
    _ex = list(extra)
    kvscale = _ex.pop(0) if quant else None
    freqs = _ex.pop(0) if rope else None
    if quant and kvscale is None: raise ValueError("quant=True requires a scale buffer bound after cache")
    if rope and freqs is None: raise ValueError("rope=True requires a freqs (cos|sin) buffer bound after cache/scale")
    # centralized KV-element load: owns the int8 dequant + in-register rope-at-read transforms (see extra/qk/kv_load.py)
    kv_load = make_kv_element_loader(cache, Hd, kvscale=kvscale, freqs=freqs)
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    # Use LOCAL ranges (not UOp.special) so add_gpudims can run and emit gidx for kvh/s.
    # UOp.special blocks add_gpudims via the any(Ops.SPECIAL) guard in gpudims.py:61.
    # AxisType.LOCAL → lidx0/lidx1 (real thread dims), correct for ds_bpermute lane addressing.
    lane = UOp.range(LANES, 10, AxisType.LOCAL)
    warp = UOp.range(WARPS, 11, AxisType.LOCAL)
    h = kvh * G + warp
    tid = warp * LANES + lane
    ksh = UOp.placeholder((TK * Hd,), dtypes.half, 230, addrspace=AddrSpace.LOCAL)
    vsh = UOp.placeholder((TK * Hd,), dtypes.half, 231, addrspace=AddrSpace.LOCAL) if staging == "KV_BOTH" else None
    acc = UOp.placeholder((R,), _F32, 232, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 233, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((1,), _F32, 234, addrspace=AddrSpace.REG)
    za = UOp.range(R, 2)
    init = acc.after(kvh, s)[za].store(0.0).end(za)
    init = den.after(init)[0].store(0.0)
    init = mx.after(init)[0].store(-float("inf"))
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    b = UOp.range(NB, 3, axis_type=AxisType.REDUCE)
    # opt-in (DECODE_STAGE_COALESCE=<W>): cooperative-staging LaneMap -- each thread owns a contiguous W-chunk
    # of the TK*Hd LDS tile, so the global cache load presents a unit-stride W loop axis that
    # COALESCED_LOAD_LOWERING folds to a vectorized load (global_load_dwordx4). Default-off: original
    # one-element-per-thread staging is byte-identical. See extra/qk/cooperative_stage_lanemap.py.
    _stage_w = getenv("DECODE_STAGE_COALESCE")
    try:
      if _stage_w:
        from extra.qk.cooperative_stage_lanemap import CooperativeStageLaneMap
        _lm = CooperativeStageLaneMap(total=TK * Hd, threads=THREADS, width=_stage_w, base_axis=60)
        _lm.validate()   # raise here (bad width / non-dividing total) -> fall through to the original staging
        st, wv = _lm.axes()
        i = _lm.elem_index(st, tid, wv)
    except ValueError:
      _stage_w = 0   # unsupported shape/width: fall back to the byte-identical one-element-per-thread staging
    if not _stage_w:
      st = UOp.range(STAGES, 4, axis_type=AxisType.REDUCE)
      i = st * THREADS + tid
    tt_stage = i // Hd
    e_stage = i % Hd
    t_stage = s * L + b * TK + tt_stage
    in_stage = (tt_stage < TK) & (t_stage < Tc)
    t_safe_stage = in_stage.where(t_stage, t_stage.const_like(0))
    _gate = () if _stage_w else (i < (TK * Hd),)   # W|TK*Hd divides evenly -> no bounds gate needed
    # K/V staging via the centralized loader (int8 dequant + rope-at-read folded in; both no-ops when off).
    kstore = ksh[i].store(kv_load(0, kvh, t_safe_stage, e_stage), *_gate)
    if staging == "KV_BOTH":
      vstore = vsh.after(kstore)[i].store(kv_load(1, kvh, t_safe_stage, e_stage), *_gate)
      bar = UOp.barrier(UOp.group(vstore.end(wv).end(st) if _stage_w else vstore.end(st)))
    else:
      # K_ONLY: barrier after K staging only; V read from global (L2-warmed by E_49152)
      bar = UOp.barrier(UOp.group(kstore.end(wv).end(st) if _stage_w else kstore.end(st)))
    def _dot_reduce(_tt):   # one token's dot (rp loop) -> warp-reduce -> scaled, masked score (the INDEPENDENT work)
      _dotp = UOp.placeholder((1,), _F32, 235, addrspace=AddrSpace.REG)
      _di = _dotp.after(b, _tt)[0].store(0.0); _dotp = _dotp.after(_di)
      _rp = UOp.range(RP, 6, axis_type=AxisType.REDUCE)
      _e2 = _rp * 64 + lane * 2
      _qp = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + _e2].cast(dtypes.half), q[h * Hd + _e2 + 1].cast(dtypes.half)))
      _kp = UOp(Ops.STACK, dtypes.half.vec(2), (ksh.after(bar)[_tt * Hd + _e2], ksh.after(bar)[_tt * Hd + _e2 + 1]))
      _d2 = UOp(Ops.CUSTOMI, _F32, (_dotp.after(_rp)[0], _qp, _kp), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
      _du = _dotp[0].store(_d2).end(_rp)
      return ((warp_reduce_sum(_dotp.after(_du)[0], lane, LANES) if getenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE", 0)
               else _warp_reduce_sum_staged(_dotp.after(_du)[0], lane, LANES)) * scale)
    if getenv("DECODE_ATTN_TILE_SPLIT_SCORE", 0):
      # REFACTOR (kernel-level recurrence pipelining): PASS 1 computes all TK independent dot+reduce scores into an
      # LDS score buffer (the ds_bpermute reduces pipeline back-to-back, no per-token serial merge stalling each),
      # PASS 2 runs the serial online-softmax merges reading the buffer. Lane 0 writes; lgkmcnt makes it warp-visible
      # (wave32 lockstep -> no barrier). Default-off (DECODE_ATTN_TILE_SPLIT_SCORE=1). Gated by the microgate + W==D.
      scsh = UOp.placeholder((WARPS * TK,), _F32, 236, addrspace=AddrSpace.LOCAL)
      tt1 = UOp.range(TK, 5, axis_type=AxisType.REDUCE)
      in_r1 = (s * L + b * TK + tt1) < Tc
      scst = scsh.after(b)[warp * TK + tt1].store(in_r1.where(_dot_reduce(tt1), _fc(-float("inf"))), lane.eq(0)).end(tt1)
      scsh = scsh.after(scst)
      tt = UOp.range(TK, 12, axis_type=AxisType.REDUCE)
      sc = scsh[warp * TK + tt]
      old_m = mx.after(tt)[0]; new_m = old_m.maximum(sc)
      corr = _fexp(old_m - new_m); p = _fexp(sc - new_m)   # sc=-inf for OOB -> corr=1, p=0 (mask folded in pass 1)
      dd = UOp.range(R, 7)
      d = lane * R + dd
      vd = (vsh.after(bar)[tt * Hd + d].cast(_F32) if staging == "KV_BOTH" else
            kv_load(1, kvh, s * L + b * TK + tt, d).cast(_F32))   # K_ONLY V from global via the centralized loader
      accu = acc[dd].store(acc.after(tt)[dd] * corr + p * vd).end(dd)
      denu = den.after(accu)[0].store(den.after(tt)[0] * corr + p)
      mxu = mx.after(denu)[0].store(new_m).end(tt).end(b)
    else:
      tt = UOp.range(TK, 5, axis_type=AxisType.REDUCE)
      in_r = (s * L + b * TK + tt) < Tc
      sc = in_r.where(_dot_reduce(tt), _fc(-float("inf")))
      old_m = mx.after(tt)[0]
      new_m = old_m.maximum(sc)
      corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
      p = in_r.where(_fexp(sc - new_m), _fc(0.0))
      dd = UOp.range(R, 7)
      d = lane * R + dd
      vd = (vsh.after(bar)[tt * Hd + d].cast(_F32) if staging == "KV_BOTH" else
            kv_load(1, kvh, s * L + b * TK + tt, d).cast(_F32))   # K_ONLY V from global via the centralized loader
      accu = acc[dd].store(acc.after(tt)[dd] * corr + p * vd).end(dd)
      denu = den.after(accu)[0].store(den.after(tt)[0] * corr + p)
      mxu = mx.after(denu)[0].store(new_m).end(tt).end(b)
    af, lf, mf = acc.after(mxu), den.after(mxu), mx.after(mxu)
    base = (h * S + s) * W
    dd2 = UOp.range(R, 8)
    d2 = lane * R + dd2
    pv = pout[base + d2].store(af[dd2]).end(dd2)
    ls = pout.after(pv)[base + Hd].store(lf[0], lane.eq(0))
    ms = pout.after(ls)[base + (Hd + 1)].store(mf[0], lane.eq(0))
    return ms.end(kvh, s, lane, warp).sink(arg=_fki(f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_max_kernel(Hq:int, MAXC:int, L:int, S, Tc):
  def kernel(pm:UOp, score:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S,  1, AxisType.GLOBAL)
    j = UOp.range(L, 2, axis_type=AxisType.REDUCE)
    t = s * L + j
    sc = (t < Tc).where(score[h * MAXC + t], _fc(-1e30))
    m = UOp.placeholder((1,), _F32, 100, addrspace=AddrSpace.REG)
    m = m.after(h, s)[0].set(-1e30)
    m = m[0].set(m.after(j)[0].maximum(sc), end=j)
    return pm[h * S + s].store(m[0]).end(h, s).sink(arg=_fki(f"flash_max_{Hq}"))
  return kernel

def flash_partial_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  # pout width W=Hd+1 per (h,s); column Hd folds the softmax denominator via a 1s-augmented v.
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, pm:UOp, score:UOp, vc:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S,  1, AxisType.GLOBAL)
    d = UOp.range(W,  2, AxisType.GLOBAL)
    kv = h // G; hs = h * S + s
    is_v = d < Hd
    m_s = pm[hs]
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    p = in_r.where(_fexp(score[h * MAXC + t] - m_s), _fc(0.0))
    # clamp the v index for out-of-range t to a valid (written) position so masked lanes read FINITE
    # data -- else p(=0) * vc[uninit cache](=Inf/NaN) = NaN poisons the accumulator.
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(vc[(kv * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((1,), _F32, 101, addrspace=AddrSpace.REG)
    acc = acc.after(h, s, d)[0].set(0.0)
    acc = acc[0].set(acc.after(j)[0] + p * vd, end=j)
    return pout[hs * W + d].store(acc[0]).end(h, s, d).sink(arg=_fki(f"flash_partial_{Hq}_{Hd}"))
  return kernel

def flash_prob_kernel(Hq:int, MAXC:int, L:int, S, Tc):
  """Variant 'hoisted': compute p[h,t] = exp(score[h,t] - pm[h,s]) ONCE per key (elementwise, no reduce),
  so flash_partial_v2 reads p instead of recomputing exp per output-dim lane (v1 recomputes it W=Hd+1 times).
  Out-of-range t -> 0 (so the masked v read in partial_v2 contributes nothing)."""
  def kernel(prob:UOp, pm:UOp, score:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S,  1, AxisType.GLOBAL)
    j = UOp.range(L,  2, AxisType.GLOBAL)
    t = s * L + j
    p = (t < Tc).where(_fexp(score[h * MAXC + t] - pm[h * S + s]), _fc(0.0))
    return prob[h * MAXC + t].store(p).end(h, s, j).sink(arg=_fki(f"flash_prob_{Hq}"))
  return kernel

def flash_partial_v2_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Variant 'hoisted' partial: pout[h,s,:] = sum_t prob[h,t]*v_aug[t,:]. No exp here (read from prob);
  column Hd folds the denominator via the 1s-augmented v, same as v1."""
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, prob:UOp, vc:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S,  1, AxisType.GLOBAL)
    d = UOp.range(W,  2, AxisType.GLOBAL)
    kv = h // G; hs = h * S + s
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    p = prob[h * MAXC + t]                                  # already 0 outside range (set in flash_prob)
    t_safe = in_r.where(t, t.const_like(0))                 # masked lanes read a valid (finite) v position
    vd = is_v.where(vc[(kv * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((1,), _F32, 101, addrspace=AddrSpace.REG)
    acc = acc.after(h, s, d)[0].set(0.0)
    acc = acc[0].set(acc.after(j)[0] + p * vd, end=j)
    return pout[hs * W + d].store(acc[0]).end(h, s, d).sink(arg=_fki(f"flash_partial_v2_{Hq}_{Hd}"))
  return kernel

def flash_partial_coop_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Variant 'gqa_coop': like v2 but the GLOBAL axis is the kv-head (not the query head), so V[kv,t,d] is read
  ONCE per (kv,split,d) thread and reused across the G=4 query heads via G register accumulators -- vs v2's
  per-query-head axis which re-reads V[kv] G times. Cuts V traffic G x (the llama flash_attn_tile reuse).
  Output layout identical to v2 (pout[(h*S+s)*W+d]), so flash_{max,prob,gmax,den,combine} are unchanged.
  Multi-reg-reduce pattern proven in extra/lds_attention_tile.py."""
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, prob:UOp, vc:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.GLOBAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(vc[(kvh * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))  # ONCE
    c = UOp.placeholder((G,), _F32, 110, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    p = in_r.where(prob[(kvh * G + g) * MAXC + t], _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_partial_coop_{Hq}_{Hd}"))
  return kernel

def flash_partial_coop_vec_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Variant 'gqa_coop_vec': identical math to gqa_coop, but the output-dim d is a LOCAL (workgroup-thread)
  axis instead of GLOBAL (grid). gqa_coop runs as 1-thread workgroups (flat_work_group_size(1,1)) -> the per-d
  V loads are scalar with no wavefront coalescing and ~1/32 lane use. Mapping d to LOCAL puts W=Hd+1 d-lanes in
  one workgroup, so adjacent lanes read adjacent V[...+d] -> coalesced fp16 loads + full lane utilization
  (llama's coalesced-load ingredient). Grid = Hkv x S. Output layout unchanged."""
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout:UOp, prob:UOp, vc:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)   # <-- LOCAL: d-lanes in a workgroup -> coalesced V loads
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(vc[(kvh * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 111, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    p = in_r.where(prob[(kvh * G + g) * MAXC + t], _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_partial_coop_vec_{Hq}_{Hd}"))
  return kernel

def flash_partial_coop_vec_kv_flat_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """gqa_coop_vec variant that reads V from a combined flat [2*Hkv*MAXC*Hd] KV buffer.
  V starts at index Hkv*MAXC*Hd in the flat buffer. Eliminates the V-slice materialization
  copy (E_49152_32_3) by accepting assigned_kv.reshape(...) instead of assigned_kv[1,0].reshape(...).
  The [1,0] indexing creates a non-contiguous view that tinygrad's callify cannot alias to the source
  buffer; the full reshape is aliasable. Route flag: DECODE_BYPASS_KV_SLICE=1. (EB-track)"""
  G = Hq // Hkv; W = Hd + 1
  V_OFF = Hkv * MAXC * Hd  # byte offset into combined [2, Hkv, MAXC, Hd] flat buffer
  def kernel(pout:UOp, prob:UOp, kv_flat:UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    v_idx = V_OFF + (kvh * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))
    vd = is_v.where(kv_flat[v_idx].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 111, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(G, 5)
    p = in_r.where(prob[(kvh * G + g) * MAXC + t], _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_partial_coop_vec_kv_flat_{Hq}_{Hd}"))
  return kernel

def flash_gmax_kernel(Hq:int, S):
  def kernel(gm:UOp, pm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 100, addrspace=AddrSpace.REG)
    g = g.after(h)[0].set(-1e30)
    g = g[0].set(g.after(s)[0].maximum(pm[h * S + s]), end=s)
    return gm[h].store(g[0]).end(h).sink(arg=_fki(f"flash_gmax_{Hq}"))
  return kernel

def flash_den_kernel(Hd:int, Hq:int, S):
  W = Hd + 1
  def kernel(den:UOp, pout:UOp, pm:UOp, gm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    gm_h = gm[h]
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    w = _fexp(pm[h * S + s] - gm_h)
    dd = UOp.placeholder((1,), _F32, 100, addrspace=AddrSpace.REG)
    dd = dd.after(h)[0].set(0.0)
    dd = dd[0].set(dd.after(s)[0] + w * pout[(h * S + s) * W + Hd], end=s)
    return den[h].store(dd[0]).end(h).sink(arg=_fki(f"flash_den_{Hq}"))
  return kernel

def flash_combine_kernel(Hd:int, Hq:int, S):
  W = Hd + 1
  def kernel(out:UOp, pout:UOp, pm:UOp, gm:UOp, den:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    gm_h = gm[h]; den_h = den[h]
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    w = _fexp(pm[h * S + s] - gm_h)
    num = UOp.placeholder((1,), _F32, 101, addrspace=AddrSpace.REG)
    num = num.after(h, d)[0].set(0.0)
    num = num[0].set(num.after(s)[0] + w * pout[(h * S + s) * W + d], end=s)
    return out[h * Hd + d].store(num[0] / den_h).end(h, d).sink(arg=_fki(f"flash_combine_{Hq}_{Hd}"))
  return kernel


def flash_combine_merged_kernel(Hd:int, Hq:int, S):
  """SPLIT-PRESERVING merge of flash_gmax + flash_den + flash_combine into ONE generated kernel per head.

  Does NOT touch flash_partial (the Hq*S partial phase stays fully parallel) — it only collapses the 3 small
  combine reduce kernels into one launch, eliminating the gm/dn global buffers and 2 kernel dispatches. gm (max
  over s), then den (sum over s), then out[h,d] (sum over s), sequential in one kernel; d is lane-sharded.
  Generated UOp, no handwritten kernel. See BoltBeam docs/attention-combine-reachability-audit-20260701.md."""
  W = Hd + 1
  LANES = 32
  if Hd % LANES != 0: raise ValueError(f"need Hd%{LANES}==0, got {Hd}")
  R = Hd // LANES

  def kernel(out:UOp, pout:UOp, pm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    s1 = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 100, addrspace=AddrSpace.REG)
    g = g.after(h)[0].set(-1e30)
    g = g[0].set(g.after(s1)[0].maximum(pm[h * S + s1]), end=s1)
    gm = g[0]
    s2 = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    dd = UOp.placeholder((1,), _F32, 101, addrspace=AddrSpace.REG)
    dd = dd.after(g)[0].set(0.0)
    dd = dd[0].set(dd.after(s2)[0] + _fexp(pm[h * S + s2] - gm) * pout[(h * S + s2) * W + Hd], end=s2)
    den = dd[0]
    rr = UOp.range(R, 3)
    d = lane * R + rr
    s3 = UOp.range(S, 4, axis_type=AxisType.REDUCE)
    num = UOp.placeholder((1,), _F32, 102, addrspace=AddrSpace.REG)
    num = num.after(dd, rr)[0].set(0.0)
    num = num[0].set(num.after(s3)[0] + _fexp(pm[h * S + s3] - gm) * pout[(h * S + s3) * W + d], end=s3)
    return out[h * Hd + d].store(num[0] / den).end(rr).end(h).sink(
      arg=_fki(f"flash_combine_merged_{Hq}_{Hd}"))

  return kernel

