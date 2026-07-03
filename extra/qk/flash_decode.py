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
from extra.qk.flash_common import _F32, _fexp, _fc, _fki, _ceildiv, Tensor, dtypes, getenv, AddrSpace, AxisType, KernelInfo, Ops, UOp  # noqa: F401
from extra.qk.flash_kernels import *   # noqa: F401,F403  (re-export kernel builders for external importers)

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

_LOG2E = 1.4426950408889634
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
  # SPLIT-PRESERVING combine merge (rollback = DECODE_ATTN_COMBINE_MERGED=0, default-off): flash_partial above
  # (Hq*S workgroups) is untouched; only the 3 small combine reduce kernels merge into one launch (drops the
  # gm/dn global buffers + 2 dispatches). This is NOT the refuted Hq-only fused route (that collapsed the partial).
  if getenv("DECODE_ATTN_COMBINE_MERGED", 0):
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, fxn=flash_combine_merged_kernel(Hd, Hq, S))[0]
    return out.reshape(Hq, Hd)
  gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
  dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, S))[0]
  out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, gm, dn, fxn=flash_combine_kernel(Hd, Hq, S))[0]
  return out.reshape(Hq, Hd)

def flash_decode_attention_kv_flat(q:Tensor, k_full:Tensor, kv_flat:Tensor, Tc_b, Tc_u,
                                    Hd:int, Hq:int, Hkv:int, MAXC:int, L:int=256) -> Tensor:
  """gqa_coop_vec flash-decode that eliminates the E_49152 V-slice materialization copy.

  K still uses k_full [Hkv,MAXC,Hd] for the score matmul (unchanged path).
  V is read from kv_flat [2*Hkv*MAXC*Hd] at offset Hkv*MAXC*Hd inside flash_partial_coop_vec_kv_flat.
  kv_flat = assigned_kv.reshape(2*Hkv*MAXC*Hd) — a contiguous reshape tinygrad can alias to cache_kv,
  avoiding the [1,0] indexing that forced the E_49152 copy. Route flag: DECODE_BYPASS_KV_SLICE=1. (EB-track)
  """
  G = Hq // Hkv; W = Hd + 1; Smax = _ceildiv(MAXC, L); S = (Tc_u + L - 1) // L
  scale = 1.0 / (Hd ** 0.5)
  qg = q.reshape(Hkv, G, Hd)
  ks = k_full[:, 0:Tc_b, :]
  scores = (qg @ ks.transpose(-1, -2)).reshape(Hq, Tc_b) * scale
  score_buf = Tensor.empty(Hq, MAXC, dtype=_F32)
  score_a = Tensor(score_buf.uop.after(score_buf[:, 0:Tc_b].uop.store(scores.cast(_F32).uop)))
  score_f = score_a.reshape(Hq * MAXC)
  pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f, fxn=flash_max_kernel(Hq, MAXC, L, S, Tc_u))[0]
  prob = Tensor.empty(Hq * MAXC, dtype=_F32).custom_kernel(pm, score_f, fxn=flash_prob_kernel(Hq, MAXC, L, S, Tc_u))[0]
  po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(
    prob, kv_flat, fxn=flash_partial_coop_vec_kv_flat_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
  if getenv("DECODE_ATTN_COMBINE_MERGED", 0):
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, fxn=flash_combine_merged_kernel(Hd, Hq, S))[0]
    return out.reshape(Hq, Hd)
  gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
  dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, S))[0]
  out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, gm, dn, fxn=flash_combine_kernel(Hd, Hq, S))[0]
  return out.reshape(Hq, Hd)

def flash_decode_g5_block_tile(q:Tensor, cache_kv:Tensor, Tc_b, Tc_u,
                               Hd:int, Hq:int, Hkv:int, MAXC:int, L:int=128, staging:str="KV_BOTH") -> Tensor:
  """G=5 block tile flash decode for 14B (Hq=40, Hkv=8, G=5). Sliced path: l_route=L, S=smax_route.

  Grid = Hkv × smax_route workgroups (concrete, using smax_route=ceildiv(MAXC,L) not symbolic s_route).
  Using symbolic s_route as S collapses the global axis to a serial inner loop (verified: single-WG
  serialization, 3 GB/s vs 960 GB/s peak). smax_route fixes this: all splits launch in parallel, OOB
  positions are masked by the existing in_stage < Tc check in the kernel body.
  Requires WARPS=G (already parameterized in the block tile kernel). Historical microgate primitive; not model-wired.
  staging="K_ONLY": stage only K in LDS, read V from global (L2-warm).
  """
  W2 = Hd + 2
  l_route = L
  smax_route = _ceildiv(MAXC, l_route)   # concrete grid bound; OOB positions masked by in_stage < Tc
  q_f = q.reshape(Hq * Hd)
  po = Tensor.empty(Hq * smax_route * W2, dtype=_F32).custom_kernel(
    q_f, cache_kv,
    fxn=flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, l_route, smax_route, Tc_u, staging=staging))[0]
  # Combine: opt-in single-kernel machine-search fused split-preserving combine (TG-P9/P14.8 generated UOp), which
  # replaces the 2-kernel flash_state_gmax + flash_state_combine lifecycle and removes the Hd-fold fexp redundancy
  # (Hq*Hd*S -> Hq*S). Uses the AMD baseline reduce/upcast lowering for the manual-END accumulator.
  if getenv("DECODE_G5_FUSED_COMBINE"):
    from extra.qk.live_split_geometry import flash_fused_gmax_combine_kernel
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, fxn=flash_fused_gmax_combine_kernel(Hd, Hq, smax_route, stride=smax_route))[0]
  else:
    gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, smax_route, stride=smax_route))[0]
    out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, gm, fxn=flash_state_combine_kernel(Hd, Hq, smax_route, stride=smax_route))[0]
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
    from extra.qk.decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel
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
    # default 48 == 4*CU/Hkv -> ~4 workgroups/CU.
    W2 = Hd + 2
    # L is concrete (from MAXC + target split count); S is symbolic in Tc_u (mirrors the other routes,
    # which must not Python-eval the JIT-bound context length). target_s=48 keeps ~4 workgroups/CU.
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
    combine_stride = s_route   # HIP tile packs partials at S=s_route stride; gmax/combine read at the same stride
    if getenv("DECODE_ATTN_NATIVE_ISA_BLOCK_TILE", 0) and getenv("DECODE_ATTN_BLOCK_TILE", 0):
      # Phase H: the block tile is compiled by the NATIVE AMD ISA backend (AMDISARenderer) and injected as a
      # precompiled Ops.PROGRAM graph node; gmax + combine stay on HIP. Default-off. L must be a concrete loop bound.
      # Phase N3F (dynamic-S): compile ONCE at the concrete Smax (partials stride + RANGE->gidx grid bound) but launch
      # only s_route = cdiv(Tc,L) split-workgroups (symbolic grid resolved at launch). With FIXED_S, s_route==smax_route
      # (static grid, unchanged). Partials are packed at the Smax stride; gmax/combine read s_route splits at Smax stride.
      if not isinstance(l_route, int):
        raise RuntimeError("DECODE_ATTN_NATIVE_ISA_BLOCK_TILE needs a concrete L loop bound; set DECODE_ATTN_BLOCK_TILE_FIXED_S/_L")
      from extra.qk.native_isa_block_tile_graph_node import native_isa_block_tile
      po = native_isa_block_tile(Tensor.empty(Hq * smax_route * W2, dtype=_F32), q_f, cache_kv,
                                 Hd, Hq, Hkv, MAXC, l_route, smax_route, Tc_u, s_grid=s_route)
      combine_stride = smax_route   # native tile compiled at Smax -> partials packed at Smax stride
    else:
      po = Tensor.empty(Hq * smax_route * W2, dtype=_F32).custom_kernel(q_f, cache_kv,
        fxn=tile_builder(Hd, Hq, Hkv, MAXC, l_route, s_route, Tc_u))[0]
    if getenv("DECODE_ATTN_FUSED_COMBINE", 0):   # P5: one fused dispatch (inline gmax) instead of gmax+combine
      out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, fxn=flash_fused_state_combine_kernel(Hd, Hq, s_route, stride=combine_stride))[0]
    else:
      gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, s_route, stride=combine_stride))[0]
      out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, gm, fxn=flash_state_combine_kernel(Hd, Hq, s_route, stride=combine_stride))[0]
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
