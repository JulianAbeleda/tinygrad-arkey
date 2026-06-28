#!/usr/bin/env python3
"""Approach B: custom Flash-Decoding kernels for batch-1 GQA decode attention.
Splits the KV sequence into S chunks -> Hq*S workgroups (full GPU at batch 1), online softmax per split,
LSE reduction across splits. Decode-only (T=1). Exact up to fp reassociation.

Two kernels:
  flash_partial: workgroup=(head h, split s), 128 threads (one per head_dim d). Online softmax over the
                 split's keys -> partial out[h,s,:] (unnormalized) + m[h,s] (max) + l[h,s] (sum exp).
  flash_reduce:  workgroup=head h -> combine the S splits via the LSE formula -> out[h,:].
"""
from __future__ import annotations
import os

_SCORE_BROADCAST_SCRATCH = {}
_SCORE_BROADCAST_NO_GRAPH_PREFIXES = (
  "flash_pall_score_once_state_",
  "flash_pall_score_broadcast_pv_cols_",
  "flash_pall_score_broadcast_combine4_",
)

def _install_score_broadcast_no_graph_prefixes():
  existing = [p for p in os.environ.get("JIT_NO_GRAPH_KERNEL_PREFIXES", "").split(",") if p]
  merged = existing + [p for p in _SCORE_BROADCAST_NO_GRAPH_PREFIXES if p not in existing]
  os.environ["JIT_NO_GRAPH_KERNEL_PREFIXES"] = ",".join(merged)

def _score_broadcast_scratch(device, Hq:int, Hd:int, Hkv:int, MAXC:int, L:int, Smax:int):
  # Stable, realized scratch for the multi-kernel score-broadcast chain. These
  # buffers are intentionally kept out of TinyJit memory planning; the route is
  # effect-ordered through AFTER(custom_kernel) dependencies and reuses scratch
  # sequentially across block calls. This is not a throughput optimization.
  key = (device, Hq, Hd, Hkv, MAXC, L, Smax)
  if key not in _SCORE_BROADCAST_SCRATCH:
    state = Tensor.empty(Hq * Smax * 2, dtype=_F32, device=device).contiguous().realize()
    pvs = tuple(Tensor.empty(Hq * Smax * 32, dtype=_F32, device=device).contiguous().realize() for _ in range(4))
    out = Tensor.empty(Hq * Hd, dtype=_F32, device=device).contiguous().realize()
    _SCORE_BROADCAST_SCRATCH[key] = (state, *pvs, out)
  return _SCORE_BROADCAST_SCRATCH[key]

_HDR = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, pure)) float __ocml_exp2_f32(float);
#define Hd %d
#define S %d
#define G %d
#define MAXC %d
#define EXPF(x) __ocml_exp2_f32((x) * 1.4426950408889634f)
'''

# NOTE: each kernel MUST be compiled into its own lib -- tinygrad's dev.runtime(name, lib) reads the wrong
# kernarg size for a 2nd kernel in a multi-kernel lib (silent MMU fault). So two source fns, two compiles.
def flash_partial_src(Hd:int, Hq:int, Hkv:int, S:int, MAXC:int) -> str:
  return (_HDR % (Hd, S, Hq // Hkv, MAXC)) + f'''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,{Hd})))
void flash_partial(float* pout, float* pm, float* pl, _Float16* q, _Float16* kc, _Float16* vc, unsigned int Tc) {{
  unsigned int wg = (unsigned int)__ockl_get_group_id(0); int d = (int)__ockl_get_local_id(0);
  int h = wg / S; int s = wg % S; int kv = h / G;
  int per = ((int)Tc + S - 1) / S; int t0 = s*per; int t1 = t0+per; if (t1 > (int)Tc) t1 = (int)Tc;
  float scale = 1.0f / 11.313708498984761f;   // 1/sqrt(128)
  float m = -1e30f, l = 0.0f, acc = 0.0f;
  _Float16* qrow = q + (long)h*Hd;
  for (int t=t0; t<t1; t++) {{
    _Float16* krow = kc + ((long)kv*MAXC + t)*Hd;
    float dot = 0.0f;                                  // each thread computes the full q.k (no LDS/barrier)
    for (int e=0; e<Hd; e++) dot += (float)qrow[e] * (float)krow[e];
    dot *= scale;
    float mn = m > dot ? m : dot;
    float corr = EXPF(m - mn);
    float p = EXPF(dot - mn);
    l = l*corr + p;
    float vd = (float)vc[((long)kv*MAXC + t)*Hd + d];
    acc = acc*corr + p*vd;
    m = mn;
  }}
  pout[((long)h*S + s)*Hd + d] = acc;
  if (d == 0) {{ pm[h*S + s] = m; pl[h*S + s] = l; }}
}}
'''

def flash_reduce_src(Hd:int, Hq:int, Hkv:int, S:int, MAXC:int) -> str:
  return (_HDR % (Hd, S, Hq // Hkv, MAXC)) + f'''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,{Hd})))
void flash_reduce(float* out, float* pout, float* pm, float* pl) {{
  int h = (int)__ockl_get_group_id(0); int d = (int)__ockl_get_local_id(0);
  float gm = -1e30f;
  for (int s=0; s<S; s++) {{ float v = pm[h*S+s]; if (v > gm) gm = v; }}
  float num = 0.0f, den = 0.0f;
  for (int s=0; s<S; s++) {{ float w = EXPF(pm[h*S+s] - gm); num += pout[((long)h*S+s)*Hd+d]*w; den += pl[h*S+s]*w; }}
  out[h*Hd + d] = num / den;
}}
'''

# ============================================================================================
# UOp Flash-Decoding (the model-integrated path). The raw kernels above can't bridge into the
# JITted attention graph (custom_kernel takes a UOp builder, not raw C). These 5 UOp kernels do,
# and accept a SYMBOLIC split count S = cdiv(start_pos+1, L) over fixed-L chunks for occupancy.
#
# Structure (each kernel is ONE single-accumulator reduce -- the proven q4k_gemv_partial pattern;
# coupled/multi-accumulator reduces trip the linearizer's range-ordering, so the softmax is split
# across kernels):
#   flash_max:     pm[h,s]     = max_{t in split s} score[h,t]                  (per-split max)
#   flash_partial: pout[h,s,:] = sum_{t in split} exp(score-pm)*v_aug[t,:]      (col Hd = 1s denom)
#   flash_gmax:    gm[h]       = max_s pm[h,s]                                  (global max)
#   flash_den:     den[h]      = sum_s exp(pm-gm)*pout[h,s,Hd]                  (softmax denominator)
#   flash_combine: out[h,d]    = (sum_s exp(pm-gm)*pout[h,s,d]) / den[h]        (LSE reduction)
# Scores are precomputed via a matmul (grouped_q @ k^T) so the kernels never nest a q.k reduce.
# Buffers are Smax-sized (concrete, for placeholder_like); ranges/strides use the symbolic S<=Smax.
from tinygrad import Tensor, dtypes  # noqa: E402
from tinygrad.helpers import getenv  # noqa: E402
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp  # noqa: E402

_LOG2E = 1.4426950408889634
_F32 = dtypes.float32
def _fexp(x:UOp) -> UOp:
  arg = x * _LOG2E
  # DECODE_FAST_EXP2 (default-off): on the online-softmax carry chain the exp argument is ALWAYS <= 0
  # (old_m-new_m<=0, sc-new_m<=0), so the ocml range-reduction (v_cmp 0xc2fc0000 + 2x v_cndmask + v_ldexp
  # guarding the large-magnitude/denormal range) is dead weight ON the serial carry. Emit a bare v_exp_f32 via
  # the AMDGCN builtin -- 2^arg with no range reduction. For arg<<0 the instruction underflows to 0 (correct for
  # masked/saturated tokens, which are additionally where()-guarded), and arg>0 never occurs on this path.
  if getenv("DECODE_FAST_EXP2", 0): return UOp(Ops.CUSTOMI, arg.dtype, (arg,), arg="__builtin_amdgcn_exp2f({0})")
  return arg.exp2()
def _fc(v:float) -> UOp: return UOp.const(_F32, v)
def _fki(name:str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _ceildiv(a:int, b:int) -> int: return (a + b - 1) // b

# Single source of truth for accepted FLASH_VARIANT values (consumed by flash_decode_attention + model.py).
# 'gqa_coop' is the shipped default; 'hoisted'/'v1' are historical/fallback. Unknown -> raise (see below).
FLASH_DECODE_VARIANTS = ("v1", "hoisted", "gqa_coop", "gqa_coop_vec")
FLASH_DECODE_DEFAULT_VARIANT = "gqa_coop_vec"

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
    from extra.qk_lane_partition_reduce import LanePartition, lane_partition_reduce_sum
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
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
    from extra.amd_warp_reduce import warp_reduce_sum
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
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
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
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
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

def flash_state_gmax_kernel(Hd:int, Hq:int, S):
  W = Hd + 2; M_COL = Hd + 1
  def kernel(gm:UOp, pout:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 133, addrspace=AddrSpace.REG)
    g = g.after(h)[0].set(-1e30)
    g = g[0].set(g.after(s)[0].maximum(pout[(h * S + s) * W + M_COL]), end=s)
    return gm[h].store(g[0]).end(h).sink(arg=_fki(f"flash_state_gmax_{Hq}_{Hd}"))
  return kernel

def flash_state_combine_kernel(Hd:int, Hq:int, S):
  W = Hd + 2; L_COL = Hd; M_COL = Hd + 1
  def kernel(out:UOp, pout:UOp, gm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    gm_h = gm[h]
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    w = _fexp(pout[(h * S + s) * W + M_COL] - gm_h)
    num = UOp.placeholder((1,), _F32, 134, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 135, addrspace=AddrSpace.REG)
    num = num.after(h, d)[0].set(0.0)
    den = den.after(h, d)[0].set(0.0)
    upd = num[0].store(num.after(s)[0] + w * pout[(h * S + s) * W + d])
    upd = den.after(upd)[0].store(den.after(s)[0] + w * pout[(h * S + s) * W + L_COL]).end(s)
    nf, df = num.after(upd)[0], den.after(upd)[0]
    return out[h * Hd + d].store(nf / df).end(h, d).sink(arg=_fki(f"flash_state_combine_{Hq}_{Hd}"))
  return kernel

def flash_fused_state_combine_kernel(Hd:int, Hq:int, S):
  """P5 missing-primitive: fuse the global-max into the combine -> ONE dispatch instead of gmax+combine, and no gm
  buffer round-trip. Each (h,d) thread computes gm[h]=max_s M[s] inline (pass 1) then the log-sum-exp rescale
  (pass 2). M is read twice but in-kernel (no global gm buffer). Default-off (DECODE_ATTN_FUSED_COMBINE=1).
  docs/decode-attention-owned-lifecycle-missing-primitives-scope-20260627.md."""
  W = Hd + 2; L_COL = Hd; M_COL = Hd + 1
  def kernel(out:UOp, pout:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    s1 = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 136, addrspace=AddrSpace.REG)
    g = g.after(h, d)[0].set(-1e30)
    g = g[0].set(g.after(s1)[0].maximum(pout[(h * S + s1) * W + M_COL]), end=s1)   # pass 1: inline gmax
    gm_h = g[0]
    s2 = UOp.range(S, 3, axis_type=AxisType.REDUCE)
    w = _fexp(pout[(h * S + s2) * W + M_COL] - gm_h)
    num = UOp.placeholder((1,), _F32, 137, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 138, addrspace=AddrSpace.REG)
    num = num.after(g)[0].set(0.0); den = den.after(g)[0].set(0.0)
    upd = num[0].store(num.after(s2)[0] + w * pout[(h * S + s2) * W + d])
    upd = den.after(upd)[0].store(den.after(s2)[0] + w * pout[(h * S + s2) * W + L_COL]).end(s2)   # pass 2: rescale+norm
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
    from extra.amd_warp_reduce import warp_reduce_max
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
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

def flash_xlane_state_ml_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Token-sharded per-split online-softmax state only.

  Output state layout: state[(h*S+s)*2 + 0] = l, state[(h*S+s)*2 + 1] = m.
  This computes (m,l) once per (head, split), not once per output column.
  """
  G = Hq // Hkv; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(state:UOp, score:UOp) -> UOp:
    from extra.amd_warp_reduce import warp_reduce_max
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    j = r * LANES + lane
    t = s * L + j
    in_r = (j < L) & (t < Tc)
    t_safe = in_r.where(t, t.const_like(0))
    l = UOp.placeholder((G,), _F32, 139, addrspace=AddrSpace.REG)
    m = UOp.placeholder((G,), _F32, 140, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 3)
    init = l[zi].store(0.0).end(zi)
    zi2 = UOp.range(G, 4)
    init = m.after(init)[zi2].store(-float("inf")).end(zi2)
    l, m = l.after(init), m.after(init)
    g = UOp.range(G, 5)
    h = kvh * G + g
    old_m = m.after(r)[g]
    sc = in_r.where(score[h * MAXC + t_safe], old_m)
    mn = in_r.where(old_m.maximum(sc), old_m)
    corr = in_r.where(_fexp(old_m - mn), _fc(1.0))
    p = in_r.where(_fexp(sc - mn), _fc(0.0))
    upd = l[g].store(l.after(r)[g] * corr + p)
    upd = m.after(upd)[g].store(mn).end(g).end(r)
    g2 = UOp.range(G, 6)
    lf, mf = l.after(upd), m.after(upd)
    gm = warp_reduce_max(mf[g2], lane, LANES, 90)
    w = _fexp(mf[g2] - gm)
    l_all = _warp_reduce_sum_staged(lf[g2] * w, lane, LANES, 96)
    hs = (kvh * G + g2) * S + s
    col = UOp.range(2, 7, AxisType.GLOBAL)
    val = col.eq(0).where(l_all, gm)
    return state[hs * 2 + col].store(val, lane.eq(0)).end(col, g2).end(kvh, s).sink(
      arg=_fki(f"flash_xlane_state_ml_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_xlane_pv_from_state_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Token-sharded PV partial that reuses precomputed per-split m.

  Output layout: pv[(h*S+s)*Hd + d] = sum_t exp(score[h,t]-m[h,s]) * V[kv,t,d].
  """
  G = Hq // Hkv; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(pv:UOp, state:UOp, score:UOp, cache:UOp) -> UOp:
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(Hd, 2, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    r = UOp.range(R, 3, axis_type=AxisType.REDUCE)
    j = r * LANES + lane
    t = s * L + j
    in_r = (j < L) & (t < Tc)
    t_safe = in_r.where(t, t.const_like(0))
    g = UOp.range(G, 4)
    h = kvh * G + g
    m_s = state[(h * S + s) * 2 + 1]
    p = in_r.where(_fexp(score[h * MAXC + t_safe] - m_s), _fc(0.0))
    vd = cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + d].cast(_F32)
    acc = UOp.placeholder((G * Hd,), _F32, 144, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 5)
    init = acc[zi * Hd + d].store(0.0).end(zi)
    acc = acc.after(init)
    gd = g * Hd + d
    upd = acc[gd].store(acc.after(r)[gd] + p * vd).end(g).end(r)
    g2 = UOp.range(G, 6)
    part = _warp_reduce_sum_staged(acc.after(upd)[g2 * Hd + d], lane, LANES, 90)
    h2 = kvh * G + g2
    return pv[(h2 * S + s) * Hd + d].store(part, lane.eq(0)).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_xlane_pv_from_state_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_xlane_split_m_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(pm:UOp, score:UOp) -> UOp:
    from extra.amd_warp_reduce import warp_reduce_max
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

def flash_xlane_split_l_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(pl:UOp, pm:UOp, score:UOp) -> UOp:
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
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
    p = in_r.where(_fexp(score[h * MAXC + t_safe] - pm[h * S + s]), _fc(0.0))
    acc = UOp.placeholder((G,), _F32, 146, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4)
    init = acc[zi].store(0.0).end(zi)
    acc = acc.after(init)
    upd = acc[g].store(acc.after(r)[g] + p).end(g).end(r)
    g2 = UOp.range(G, 5)
    l_all = _warp_reduce_sum_staged(acc.after(upd)[g2], lane, LANES, 90)
    h2 = kvh * G + g2
    return pl[h2 * S + s].store(l_all, lane.eq(0)).end(g2).end(kvh, s).sink(arg=_fki(f"flash_xlane_split_l_{Hq}_{Hd}"))
  return kernel

def flash_xlane_pv_from_m_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  G = Hq // Hkv; W = Hd + 1; LANES = 32; R = _ceildiv(L, LANES)
  def kernel(pv:UOp, pm:UOp, score:UOp, cache:UOp) -> UOp:
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
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
  extra/qk_decode_attention_fused_xlane_score_pv_microgate.py.
  """
  if Hd % 64 != 0: raise ValueError(f"fused xlane score+PV requires Hd%%64==0, got {Hd}")
  G = Hq // Hkv; W = Hd + 2; LANES = 32; R = Hd // LANES; RP = Hd // 64
  scale = 1.0 / (Hd ** 0.5)
  def kernel(pout:UOp, q:UOp, cache:UOp) -> UOp:
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
    from extra.amd_warp_reduce import warp_reduce_sum
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

def flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Block-tiled generated decode candidate.

  Mirrors the owned tile's topology at the UOp level: one workgroup per (kvh, split), 4 warps per
  workgroup, one warp per GQA query head, TK=16 K/V rows staged in LDS, then online softmax + d-sharded PV.
  This is default-off and guarded by extra/qk_decode_attention_block_tile_microgate.py before route use.
  """
  if Hd % 64 != 0: raise ValueError(f"block tile requires Hd%%64==0, got {Hd}")
  G = Hq // Hkv; W = Hd + 2; LANES = 32; WARPS = 4; THREADS = LANES * WARPS; TK = 16
  if G != WARPS: raise ValueError(f"block tile expects G=={WARPS}, got {G}")
  R = Hd // LANES; RP = Hd // 64; STAGES = _ceildiv(TK * Hd, THREADS); NB = _ceildiv(L, TK)
  scale = 1.0 / (Hd ** 0.5)
  def kernel(pout:UOp, q:UOp, cache:UOp) -> UOp:
    from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged
    from extra.amd_warp_reduce import warp_reduce_sum
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    warp = UOp.special(WARPS, "lidx1")
    h = kvh * G + warp
    tid = warp * LANES + lane
    ksh = UOp.placeholder((TK * Hd,), dtypes.half, 230, addrspace=AddrSpace.LOCAL)
    vsh = UOp.placeholder((TK * Hd,), dtypes.half, 231, addrspace=AddrSpace.LOCAL)
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
    # one-element-per-thread staging is byte-identical. See extra/qk_cooperative_stage_lanemap.py.
    _stage_w = getenv("DECODE_STAGE_COALESCE")
    try:
      if _stage_w:
        from extra.qk_cooperative_stage_lanemap import CooperativeStageLaneMap
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
    kstore = ksh[i].store(cache[0, 0, kvh, t_safe_stage, e_stage].cast(dtypes.half), *_gate)
    vstore = vsh.after(kstore)[i].store(cache[1, 0, kvh, t_safe_stage, e_stage].cast(dtypes.half), *_gate)
    bar = UOp.barrier(UOp.group(vstore.end(wv).end(st) if _stage_w else vstore.end(st)))
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
      vd = vsh.after(bar)[tt * Hd + d].cast(_F32)
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
      vd = vsh.after(bar)[tt * Hd + d].cast(_F32)
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
    return ms.end(kvh, s).sink(arg=_fki(f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel

def flash_split_ml_gmax_kernel(Hq:int, S):
  def kernel(gm:UOp, pm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 148, addrspace=AddrSpace.REG)
    g = g.after(h)[0].set(-1e30)
    g = g[0].set(g.after(s)[0].maximum(pm[h * S + s]), end=s)
    return gm[h].store(g[0]).end(h).sink(arg=_fki(f"flash_split_ml_gmax_{Hq}"))
  return kernel

def flash_split_ml_combine_kernel(Hd:int, Hq:int, S):
  def kernel(out:UOp, pv:UOp, pm:UOp, pl:UOp, gm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    gm_h = gm[h]
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    w = _fexp(pm[h * S + s] - gm_h)
    num = UOp.placeholder((1,), _F32, 149, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 150, addrspace=AddrSpace.REG)
    num = num.after(h, d)[0].set(0.0)
    den = den.after(h, d)[0].set(0.0)
    upd = num[0].store(num.after(s)[0] + w * pv[(h * S + s) * Hd + d])
    upd = den.after(upd)[0].store(den.after(s)[0] + w * pl[h * S + s]).end(s)
    nf, df = num.after(upd)[0], den.after(upd)[0]
    return out[h * Hd + d].store(nf / df).end(h, d).sink(arg=_fki(f"flash_split_ml_combine_{Hq}_{Hd}"))
  return kernel

def flash_split_state_gmax_kernel(Hq:int, S):
  def kernel(gm:UOp, state:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    g = UOp.placeholder((1,), _F32, 141, addrspace=AddrSpace.REG)
    g = g.after(h)[0].set(-1e30)
    g = g[0].set(g.after(s)[0].maximum(state[(h * S + s) * 2 + 1]), end=s)
    return gm[h].store(g[0]).end(h).sink(arg=_fki(f"flash_split_state_gmax_{Hq}"))
  return kernel

def flash_split_state_combine_kernel(Hd:int, Hq:int, S):
  def kernel(out:UOp, pv:UOp, state:UOp, gm:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    gm_h = gm[h]
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    w = _fexp(state[(h * S + s) * 2 + 1] - gm_h)
    num = UOp.placeholder((1,), _F32, 142, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 143, addrspace=AddrSpace.REG)
    num = num.after(h, d)[0].set(0.0)
    den = den.after(h, d)[0].set(0.0)
    upd = num[0].store(num.after(s)[0] + w * pv[(h * S + s) * Hd + d])
    upd = den.after(upd)[0].store(den.after(s)[0] + w * state[(h * S + s) * 2 + 0]).end(s)
    nf, df = num.after(upd)[0], den.after(upd)[0]
    return out[h * Hd + d].store(nf / df).end(h, d).sink(arg=_fki(f"flash_split_state_combine_{Hq}_{Hd}"))
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

def flash_decode_attention(q:Tensor, k_full:Tensor, v_full:Tensor, Tc_b, Tc_u,
                           Hd:int, Hq:int, Hkv:int, MAXC:int, L:int=256, variant:str="v1") -> Tensor:
  """Batch-1 GQA decode attention via Flash-Decoding. Exact vs SDPA (up to fp reassociation).
  q:[Hq,Hd]  k_full,v_full:[Hkv,MAXC,Hd] (full KV cache buffers, concrete MAXC).
  Tc_b: bound symbolic context length (carries start_pos's value into var_vals, used for the score
        matmul slice).  Tc_u: same length as an UNbound DEFINE_VAR expr (used for the kernel ranges,
        so no BIND lands in a custom-kernel AST).  Returns [Hq,Hd] (float32).
  variant: 'hoisted' (default, exp computed once/key), 'gqa_coop' (hoisted + cooperative GQA V-reuse: V read
           once/group, ~3x on the partial), or 'v1' (legacy). Unknown -> raise, so a mistyped FLASH_VARIANT
           can't silently fall back and lose the shipped win."""
  if variant not in FLASH_DECODE_VARIANTS:
    raise ValueError(f"unknown flash variant {variant!r}; expected one of {FLASH_DECODE_VARIANTS} (check FLASH_VARIANT)")

  G = Hq // Hkv; W = Hd + 1; Smax = _ceildiv(MAXC, L); S = (Tc_u + L - 1) // L
  scale = 1.0 / (Hd ** 0.5)
  # precompute scores via a matmul over the symbolic-length KV slice, materialized into a concrete
  # [Hq,MAXC] buffer (the kernels need a concrete-shaped input). The matmul's bound Tc_b slice also
  # carries start_pos's value into var_vals, so the kernels' unbound DEFINE_VAR ranges resolve.
  qg = q.reshape(Hkv, G, Hd)
  ks = k_full[:, 0:Tc_b, :]
  scores = (qg @ ks.transpose(-1, -2)).reshape(Hq, Tc_b) * scale
  score_buf = Tensor.empty(Hq, MAXC, dtype=_F32)
  score_a = Tensor(score_buf.uop.after(score_buf[:, 0:Tc_b].uop.store(scores.cast(_F32).uop)))
  score_f = score_a.reshape(Hq * MAXC)
  vc_f = v_full.reshape(Hkv * MAXC * Hd)
  pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f, fxn=flash_max_kernel(Hq, MAXC, L, S, Tc_u))[0]
  if variant in ("hoisted", "gqa_coop", "gqa_coop_vec"):
    # exp computed once per key (flash_prob), then a pure weighted-sum partial (no per-d exp recompute).
    prob = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(pm, score_f, fxn=flash_prob_kernel(Hq, MAXC, L, S, Tc_u))[0]
    _partial = {"gqa_coop": flash_partial_coop_kernel, "gqa_coop_vec": flash_partial_coop_vec_kernel}.get(
      variant, flash_partial_v2_kernel)
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(prob, vc_f, fxn=_partial(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
  else:
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(pm, score_f, vc_f, fxn=flash_partial_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
  gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
  dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, S))[0]
  out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, gm, dn, fxn=flash_combine_kernel(Hd, Hq, S))[0]
  return out.reshape(Hq, Hd)

def flash_decode_attention_whole_cache(q:Tensor, cache_kv:Tensor, Tc_b, Tc_u,
                                       Hd:int, Hq:int, Hkv:int, MAXC:int, L:int=256) -> Tensor:
  """Generated decode-attention skeleton over the whole [2,1,Hkv,MAXC,Hd] KV cache.

  This avoids passing sliced K/V views into callify. It is an attribution/lifecycle skeleton, not a speed path.
  """
  use_vdot2 = bool(getenv("DECODE_ATTN_SCORE_VDOT2", 0))
  use_xlane = bool(getenv("DECODE_ATTN_SCORE_XLANE", 0))
  if use_vdot2: os.environ.setdefault("V_DOT2_LOWERING", "1")
  W = Hd + 1; Smax = _ceildiv(MAXC, L); S = (Tc_u + L - 1) // L
  q_f = q.reshape(Hq * Hd)
  cache_f = cache_kv.reshape(2 * Hkv * MAXC * Hd)
  if getenv("DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE", 0):
    from extra.qk_decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel
    if getenv("DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH", 1): _install_score_broadcast_no_graph_prefixes()
    chunks = getenv("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", 4)
    if chunks not in (1, 2, 3, 4): raise ValueError(f"DECODE_ATTN_SCORE_BROADCAST_CHUNKS must be 1, 2, 3, or 4, got {chunks}")
    if chunks != 4 and not getenv("DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS", 0):
      raise RuntimeError("DECODE_ATTN_SCORE_BROADCAST_CHUNKS<4 is diagnostic-only and cannot produce a full-width correct route")
    Tc_route = Tc_u
    S_route = Smax
    if getenv("DECODE_ATTN_SCORE_BROADCAST_SCRATCH", 1):
      state_s, pv0_s, pv1_s, pv2_s, pv3_s, out_s = _score_broadcast_scratch(q.device, Hq, Hd, Hkv, MAXC, L, Smax)
    else:
      state_s = Tensor.empty(Hq * Smax * 2, dtype=_F32, device=q.device)
      pv0_s = Tensor.empty(Hq * Smax * 32, dtype=_F32, device=q.device)
      pv1_s = Tensor.empty(Hq * Smax * 32, dtype=_F32, device=q.device)
      pv2_s = Tensor.empty(Hq * Smax * 32, dtype=_F32, device=q.device)
      pv3_s = Tensor.empty(Hq * Smax * 32, dtype=_F32, device=q.device)
      out_s = Tensor.empty(Hq * Hd, dtype=_F32, device=q.device)
    state = state_s.custom_kernel(q_f, cache_f,
      fxn=score_once_state_kernel(Hd, Hq, Hkv, MAXC, L, S_route, Tc_route))[0]
    pv0 = pv0_s.custom_kernel(state, q_f, cache_f,
      fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S_route, Tc_route, 32, 0))[0]
    pv1 = pv1_s.custom_kernel(state, q_f, cache_f,
      fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S_route, Tc_route, 32, 32))[0] if chunks >= 2 else pv0
    pv2 = pv2_s.custom_kernel(state, q_f, cache_f,
      fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S_route, Tc_route, 32, 64))[0] if chunks >= 3 else pv1
    pv3 = pv3_s.custom_kernel(state, q_f, cache_f,
      fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S_route, Tc_route, 32, 96))[0] if chunks >= 4 else pv2
    out = out_s.custom_kernel(state, pv0, pv1, pv2, pv3,
      fxn=flash_pall_score_broadcast_combine4_kernel(Hd, Hq, S_route))[0]
    return out.reshape(Hq, Hd)
  if getenv("DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE", 0):
    W = Hd + 2
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(q_f, cache_f,
      fxn=flash_pall_score_state_pv_lifecycle_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
    gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, S))[0]
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, gm, fxn=flash_state_combine_kernel(Hd, Hq, S))[0]
    return out.reshape(Hq, Hd)
  if getenv("DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE", 0):
    W = Hd + 2
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(q_f, cache_kv,
      fxn=flash_fused_score_state_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
    gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, S))[0]
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, gm, fxn=flash_state_combine_kernel(Hd, Hq, S))[0]
    return out.reshape(Hq, Hd)
  if getenv("DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE", 0):
    # physically-fast fused tile: score-once (e-shard fdot2 + cross-lane) + d-shard PV, buffer-identity
    # raw cache_kv. Split count S sets occupancy (docs/decode-fused-tile-occupancy-roofline-baseline.md):
    # default 48 == 4*CU/Hkv == owned route's DECODE_ATTN_AMDGCN_S -> ~4 workgroups/CU.
    W2 = Hd + 2
    # L is concrete (from MAXC + target split count); S is symbolic in Tc_u (mirrors the other routes,
    # which must not Python-eval the JIT-bound context length). target_s=48 == owned DECODE_ATTN_AMDGCN_S.
    target_s = getenv("DECODE_ATTN_FUSED_XLANE_SCORE_PV_S", 48)
    if getenv("DECODE_ATTN_BLOCK_TILE_FIXED_S", 0) and getenv("DECODE_ATTN_BLOCK_TILE", 0):
      # H2 occupancy experiment: keep the block-tile route at a concrete S grid for the measured ctx.
      # The UOp tile still needs concrete loop bounds, so L is provided explicitly by the gate/harness.
      # Example ctx512: S=48, L=ceildiv(512,48)=11 -> 384 workgroups instead of the current 48.
      smax_route = target_s
      s_route = target_s
      l_route = getenv("DECODE_ATTN_BLOCK_TILE_L", max(1, _ceildiv(MAXC, target_s)))
    else:
      l_route = max(1, _ceildiv(MAXC, target_s))
      s_route = (Tc_u + l_route - 1) // l_route
      smax_route = _ceildiv(MAXC, l_route)
    tile_builder = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel if getenv("DECODE_ATTN_BLOCK_TILE", 0) else \
      flash_fused_xlane_score_pv_tile_whole_cache_kernel
    po = Tensor.empty(Hq * smax_route * W2, dtype=_F32).custom_kernel(q_f, cache_kv,
      fxn=tile_builder(Hd, Hq, Hkv, MAXC, l_route, s_route, Tc_u))[0]
    if getenv("DECODE_ATTN_FUSED_COMBINE", 0):   # P5: one fused dispatch (inline gmax) instead of gmax+combine
      out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, fxn=flash_fused_state_combine_kernel(Hd, Hq, s_route))[0]
    else:
      gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, s_route))[0]
      out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, gm, fxn=flash_state_combine_kernel(Hd, Hq, s_route))[0]
    return out.reshape(Hq, Hd)
  if getenv("DECODE_ATTN_PHYSICAL_TILE_PALL_SCORE", 0):
    score_kernel = flash_pall_lds_crosslane_fdot2_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc_u)
  elif getenv("DECODE_ATTN_PHYSICAL_TILE_P1_SCORE", 0):
    score_kernel = flash_p1_crosslane_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc_u)
  else:
    score_kernel = flash_score_whole_cache_xlane_kernel(Hd, Hq, Hkv, MAXC, Tc_u) if use_xlane else \
      flash_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc_u, use_vdot2=use_vdot2)
  score_f = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(q_f, cache_f, fxn=score_kernel)[0]
  if getenv("DECODE_ATTN_TILE_PLACEHOLDER", 0):
    score_f = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(score_f, fxn=flash_tile_placeholder_kernel(Hd, Hq, MAXC, Tc_u))[0]
  if getenv("DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE", 0):
    W = Hd + 1
    pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f,
      fxn=flash_max_kernel(Hq, MAXC, L, S, Tc_u))[0]
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(pm, score_f, cache_f,
      fxn=flash_xlane_pv_from_m_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
    gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
    dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, S))[0]
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, gm, dn, fxn=flash_combine_kernel(Hd, Hq, S))[0]
    return out.reshape(Hq, Hd)
  if getenv("DECODE_ATTN_FUSED_PV_TILE", 0):
    W = Hd + 1
    pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f,
      fxn=flash_max_kernel(Hq, MAXC, L, S, Tc_u))[0]
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(pm, score_f, cache_f,
      fxn=flash_fused_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
    gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
    dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, S))[0]
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, gm, dn, fxn=flash_combine_kernel(Hd, Hq, S))[0]
    return out.reshape(Hq, Hd)
  use_online_state_pv_tile_xlane = getenv("DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE", 0)
  use_online_state_pv_tile = getenv("DECODE_ATTN_ONLINE_STATE_PV_TILE", 0) or use_online_state_pv_tile_xlane
  use_tile_score_max = getenv("DECODE_ATTN_TILE_SCORE_MAX", 0) or getenv("DECODE_ATTN_TILE_PROB", 0)
  if use_online_state_pv_tile:
    state_kernel = flash_online_state_pv_tile_xlane_whole_cache_kernel if use_online_state_pv_tile_xlane else \
      flash_online_state_pv_tile_whole_cache_kernel
    po = Tensor.empty(Hq * Smax * (Hd + 2), dtype=_F32).custom_kernel(score_f, cache_f,
      fxn=state_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
    gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, S))[0]
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, gm, fxn=flash_state_combine_kernel(Hd, Hq, S))[0]
    return out.reshape(Hq, Hd)
  if use_tile_score_max:
    pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f, fxn=flash_tile_score_max_kernel(Hd, Hq, MAXC, L, S, Tc_u))[0]
  else:
    pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f, fxn=flash_max_kernel(Hq, MAXC, L, S, Tc_u))[0]
  use_online_pv_tile = getenv("DECODE_ATTN_ONLINE_PV_TILE", 0)
  use_tile_prob_partial = getenv("DECODE_ATTN_TILE_PROB_PARTIAL_PV", 0) or use_online_pv_tile
  if use_tile_prob_partial:
    prob = None
  elif getenv("DECODE_ATTN_TILE_PROB", 0):
    prob = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(pm, score_f, fxn=flash_tile_prob_kernel(Hd, Hq, MAXC, L, S, Tc_u))[0]
  else:
    prob = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(pm, score_f, fxn=flash_prob_kernel(Hq, MAXC, L, S, Tc_u))[0]
  if use_tile_prob_partial:
    online_kernel = flash_online_pv_tile_whole_cache_kernel if use_online_pv_tile else flash_tile_prob_partial_pv_whole_cache_kernel
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(pm, score_f, cache_f,
      fxn=online_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
  else:
    partial_kernel = flash_tile_partial_pv_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u) if getenv("DECODE_ATTN_TILE_PARTIAL_PV", 0) else \
      flash_partial_coop_vec_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u)
    po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(prob, cache_f, fxn=partial_kernel)[0]
  gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
  dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, S))[0]
  out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, gm, dn, fxn=flash_combine_kernel(Hd, Hq, S))[0]
  return out.reshape(Hq, Hd)

if __name__ == "__main__":
  import numpy as np
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
  for Tc, S in [(3072, 8), (1024, 8), (777, 8), (100, 4)]:
    G = Hq // Hkv
    dev = Device["AMD"]; rng = np.random.default_rng(0)
    q = rng.standard_normal((Hq, Hd)).astype(np.float16)
    k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
    p_partial = dev.runtime("flash_partial", dev.compiler.compile(flash_partial_src(Hd, Hq, Hkv, S, MAXC)))
    p_reduce = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
    def buf(arr, dt):
      b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
    qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
    pout = Buffer("AMD", Hq*S*Hd, dtypes.float32).ensure_allocated()
    pm = Buffer("AMD", Hq*S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq*S, dtypes.float32).ensure_allocated()
    out = Buffer("AMD", Hq*Hd, dtypes.float32).ensure_allocated()
    p_partial(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hq*S,1,1), local_size=(Hd,1,1), vals=(Tc,), wait=True)
    p_reduce(out._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq,1,1), local_size=(Hd,1,1), wait=True)
    _o = np.empty(Hq*Hd, np.float32); out.copyout(memoryview(_o)); got = _o.reshape(Hq, Hd)
    # reference: per head, softmax(q·k[:Tc]/sqrt) @ v[:Tc]
    qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
    ref = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
      pw = np.exp(sc - sc.max()); pw /= pw.sum(); ref[h] = pw @ vf[kv]
    err = np.abs(got - ref).max()
    print(f"Tc={Tc} S={S}: max_err={err:.4g}  {'OK' if err < 2e-2 else 'FAIL'}")

  # UOp-path variants (the model-integrated path): v1 and hoisted must BOTH be exact, and identical to
  # each other (hoisted only changes WHERE exp is computed, not the math). Bound/unbound start_pos twins.
  print("\nUOp-path variants (v1 vs hoisted):")
  MAXC2 = 4608  # divisible by all swept L so flash_prob's [Hq,MAXC] buffer covers every t=s*L+j
  for Tc, L in [(512, 256), (1024, 256), (1024, 64), (3072, 256)]:
    G = Hq // Hkv
    rng = np.random.default_rng(1)
    qn = rng.standard_normal((Hq, Hd)).astype(np.float16)
    kn = rng.standard_normal((Hkv, MAXC2, Hd)).astype(np.float16)
    vn = rng.standard_normal((Hkv, MAXC2, Hd)).astype(np.float16)
    qf, kf, vf = qn.astype(np.float32), kn[:, :Tc].astype(np.float32), vn[:, :Tc].astype(np.float32)
    ref = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
      pw = np.exp(sc - sc.max()); pw /= pw.sum(); ref[h] = pw @ vf[kv]
    sp_b = UOp.variable("start_pos", 0, MAXC2 - 1).bind(Tc - 1); sp_u = UOp.variable("start_pos", 0, MAXC2 - 1)
    outs = {}
    for var in ("v1", "hoisted", "gqa_coop", "gqa_coop_vec"):
      o = flash_decode_attention(Tensor(qn), Tensor(kn), Tensor(vn), sp_b + 1, sp_u + 1,
                                 Hd, Hq, Hkv, MAXC2, L, variant=var).numpy()
      outs[var] = o
      e = float(np.abs(o - ref).max())
      print(f"  Tc={Tc} L={L} {var:8}: max_err={e:.4g} {'OK' if e < 2e-2 else 'FAIL'}")
    for var in ("hoisted", "gqa_coop", "gqa_coop_vec"):
      same = float(np.abs(outs['v1'] - outs[var]).max())
      print(f"  Tc={Tc} L={L} v1-vs-{var:8} max|diff|={same:.4g} {'IDENTICAL' if same == 0.0 else 'DIFFERS'}")
