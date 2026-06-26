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
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp  # noqa: E402

_LOG2E = 1.4426950408889634
_F32 = dtypes.float32
def _fexp(x:UOp) -> UOp: return (x * _LOG2E).exp2()
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
  score_kernel = flash_score_whole_cache_xlane_kernel(Hd, Hq, Hkv, MAXC, Tc_u) if use_xlane else \
    flash_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc_u, use_vdot2=use_vdot2)
  score_f = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(q_f, cache_f, fxn=score_kernel)[0]
  pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f, fxn=flash_max_kernel(Hq, MAXC, L, S, Tc_u))[0]
  prob = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(pm, score_f, fxn=flash_prob_kernel(Hq, MAXC, L, S, Tc_u))[0]
  po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(prob, cache_f,
    fxn=flash_partial_coop_vec_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
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
