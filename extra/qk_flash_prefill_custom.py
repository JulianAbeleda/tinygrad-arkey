#!/usr/bin/env python3
"""Prefill v2 — Increment 2, Phase 2: fused single-head causal attention via custom_kernel (expressibility).

Proves a custom kernel can compute q·k -> softmax -> ·V WITHOUT materializing the [T,KV] scores (the point of
flash). Single head, concrete shapes, correctness only (no GQA/tiling/perf -- later phases).

Formulation verdict (measured on gfx1100):
  A  single-kernel ONLINE softmax (coupled m/l/acc over j)        -> REJECTED by the linearizer
       (codegen/late/linearizer.py: `assert y.src[1] not in x.backward_slice_with_self` -- the coupled
        multi-accumulator reduce flash-decode warned about).
  B  fused max+partial in ONE kernel via two SEQUENTIAL single-accumulator j-reduces (max, then a 1s-augmented
     weighted sum that folds the softmax denominator) + a tiny combine kernel = 2 kernels  -> WORKS, exact.
  C  flash-decode-style 3 separate single-accumulator kernels (max / partial / combine)    -> WORKS, exact.
We ship B (fewer kernels, no max buffer to HBM). Architectural consequence for Phase 3+: NO in-kernel online
softmax; use sequential max-then-weighted-sum passes (the q·k dot is recomputed -- the win is no score
materialization + good occupancy, not avoiding recompute).

Lessons reused from extra/qk_flash_decode.py: single-accumulator reduce per kernel (the q4k_gemv_partial
pattern), the 1s-augmented-V denominator trick (column Hd folds Σp), exp via exp2, and the NaN-clamp (a masked
key lane must read a valid cell -- here KV is concrete so all j are valid, but we clamp the augmented-column V
index anyway). Skeleton: the nested dot+accumulate of test/backend/test_custom_kernel.py:simple_qkv_kernel.
"""
from __future__ import annotations

import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo

_F32 = dtypes.float32
_NEG = -1e30
def _exp(x:UOp) -> UOp: return (x * 1.4426950408889634).exp2()  # exp2-based, like flash-decode
def _ki(name:str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())

def maxpartial_kernel(T:int, KV:int, Hd:int, scale:float, start_pos:int):
  # ONE kernel, two sequential single-accumulator reduces over j (causal: key j attends iff j <= start_pos+i):
  #   pass 1: m[i] = max_{j<=sp+i} (q_i·k_j)*scale              (single max accumulator; dot recomputed)
  #   pass 2: po[i,d] = Σ_{j<=sp+i} exp(s_ij - m[i]) * aug(d)   (single sum accumulator; 1s-aug: d==Hd -> Σp)
  # po is [T, Hd+1]; column Hd holds the softmax denominator. (combine_kernel divides.)
  W = Hd + 1
  def kernel(PO:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
    i = UOp.range(T, 0); d = UOp.range(W, 1)
    # pass 1 -- max
    j1 = UOp.range(KV, 2, axis_type=AxisType.REDUCE); e1 = UOp.range(Hd, 3, axis_type=AxisType.REDUCE)
    dot1 = UOp.placeholder((1,), _F32, 0, addrspace=AddrSpace.REG)
    dot1 = dot1.after(i, d, j1)[0].set(0.0)
    dot1 = dot1[0].set(dot1.after(e1)[0] + Q[i, e1].cast(_F32) * K[j1, e1].cast(_F32), end=e1)
    s1 = (j1 <= (start_pos + i)).where(dot1[0] * scale, UOp.const(_F32, _NEG))
    m = UOp.placeholder((1,), _F32, 1, addrspace=AddrSpace.REG)
    m = m.after(i, d)[0].set(_NEG)
    m = m[0].set(m.after(j1)[0].maximum(s1), end=j1)
    # pass 2 -- weighted sum (+ denom via 1s-aug)
    j2 = UOp.range(KV, 4, axis_type=AxisType.REDUCE); e2 = UOp.range(Hd, 5, axis_type=AxisType.REDUCE)
    dot2 = UOp.placeholder((1,), _F32, 2, addrspace=AddrSpace.REG)
    dot2 = dot2.after(i, d, j2)[0].set(0.0)
    dot2 = dot2[0].set(dot2.after(e2)[0] + Q[i, e2].cast(_F32) * K[j2, e2].cast(_F32), end=e2)
    p = (j2 <= (start_pos + i)).where(_exp(dot2[0] * scale - m[0]), UOp.const(_F32, 0.0))
    is_v = d < Hd
    vd = is_v.where(V[j2, is_v.where(d, d.const_like(0))].cast(_F32), UOp.const(_F32, 1.0))
    acc = UOp.placeholder((1,), _F32, 3, addrspace=AddrSpace.REG)
    acc = acc.after(i, d)[0].set(0.0)
    acc = acc[0].set(acc.after(j2)[0] + p * vd, end=j2)
    return PO[i * W + d].store(acc[0]).end(i, d).sink(arg=_ki(f"fp_maxpartial_{T}_{KV}_{Hd}"))
  return kernel

def combine_kernel(T:int, Hd:int):
  W = Hd + 1
  def kernel(O:UOp, PO:UOp) -> UOp:
    i = UOp.range(T, 0); d = UOp.range(Hd, 1)
    return O[i * Hd + d].store((PO[i * W + d] / PO[i * W + Hd]).cast(dtypes.float16)).end(i, d).sink(
      arg=_ki(f"fp_combine_{T}_{Hd}"))
  return kernel

def maxpartial_gqa_kernel(Hq:int, Hkv:int, T:int, KV:int, Hd:int, scale:float, start_pos:int):
  # Phase 4: the head dimension `h` is a range INSIDE the kernel (ONE launch covers all Hq heads -- no Python
  # per-head loop). GQA maps kv_head = h // G with NO repeat_interleave of K/V (each q-head reads its kv-head
  # directly). Same two sequential single-accumulator reduces as the single-head kernel, per (h,i,d).
  G = Hq // Hkv; W = Hd + 1
  def kernel(PO:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:   # PO:[Hq,T,W]  Q:[Hq,T,Hd]  K,V:[Hkv,KV,Hd]
    h = UOp.range(Hq, 0); i = UOp.range(T, 1); d = UOp.range(W, 2)
    kv = h // G
    # pass 1 -- max
    j1 = UOp.range(KV, 3, axis_type=AxisType.REDUCE); e1 = UOp.range(Hd, 4, axis_type=AxisType.REDUCE)
    dot1 = UOp.placeholder((1,), _F32, 0, addrspace=AddrSpace.REG)
    dot1 = dot1.after(h, i, d, j1)[0].set(0.0)
    dot1 = dot1[0].set(dot1.after(e1)[0] + Q[h, i, e1].cast(_F32) * K[kv, j1, e1].cast(_F32), end=e1)
    s1 = (j1 <= (start_pos + i)).where(dot1[0] * scale, UOp.const(_F32, _NEG))
    m = UOp.placeholder((1,), _F32, 1, addrspace=AddrSpace.REG)
    m = m.after(h, i, d)[0].set(_NEG)
    m = m[0].set(m.after(j1)[0].maximum(s1), end=j1)
    # pass 2 -- weighted sum (+ denom via 1s-aug)
    j2 = UOp.range(KV, 5, axis_type=AxisType.REDUCE); e2 = UOp.range(Hd, 6, axis_type=AxisType.REDUCE)
    dot2 = UOp.placeholder((1,), _F32, 2, addrspace=AddrSpace.REG)
    dot2 = dot2.after(h, i, d, j2)[0].set(0.0)
    dot2 = dot2[0].set(dot2.after(e2)[0] + Q[h, i, e2].cast(_F32) * K[kv, j2, e2].cast(_F32), end=e2)
    p = (j2 <= (start_pos + i)).where(_exp(dot2[0] * scale - m[0]), UOp.const(_F32, 0.0))
    is_v = d < Hd
    vd = is_v.where(V[kv, j2, is_v.where(d, d.const_like(0))].cast(_F32), UOp.const(_F32, 1.0))
    acc = UOp.placeholder((1,), _F32, 3, addrspace=AddrSpace.REG)
    acc = acc.after(h, i, d)[0].set(0.0)
    acc = acc[0].set(acc.after(j2)[0] + p * vd, end=j2)
    return PO[h, i, d].store(acc[0]).end(h, i, d).sink(arg=_ki(f"fp_maxpartial_gqa_{Hq}_{T}_{KV}_{Hd}"))
  return kernel

def combine_gqa_kernel(Hq:int, T:int, Hd:int):
  W = Hd + 1
  def kernel(O:UOp, PO:UOp) -> UOp:   # O:[Hq,T,Hd]  PO:[Hq,T,W]
    h = UOp.range(Hq, 0); i = UOp.range(T, 1); d = UOp.range(Hd, 2)
    return O[h, i, d].store((PO[h, i, d] / PO[h, i, Hd]).cast(dtypes.float16)).end(h, i, d).sink(
      arg=_ki(f"fp_combine_gqa_{Hq}_{T}_{Hd}"))
  return kernel

def flash_prefill_attention(q:Tensor, k:Tensor, v:Tensor, start_pos:int = 0) -> Tensor:
  """GQA multi-head causal attention, fused + score-free, heads covered INSIDE the kernel (2 programs total,
  not 2*Hq). q:[Hq,T,Hd]  k,v:[Hkv,KV,Hd] with KV=start_pos+T. Returns [Hq,T,Hd] fp16. Exact vs SDPA up to fp
  reassociation. Inputs should be contiguous (Phase-1 invariant)."""
  Hq, T, Hd = q.shape; Hkv, KV, _ = k.shape
  scale = 1.0 / math.sqrt(Hd)
  PO = Tensor.empty(Hq, T, Hd + 1, dtype=_F32).custom_kernel(
    q.contiguous(), k.contiguous(), v.contiguous(), fxn=maxpartial_gqa_kernel(Hq, Hkv, T, KV, Hd, scale, start_pos))[0]
  O = Tensor.empty(Hq, T, Hd, dtype=dtypes.float16).custom_kernel(PO, fxn=combine_gqa_kernel(Hq, T, Hd))[0]
  return O

def flash_prefill_attention_1h(q:Tensor, k:Tensor, v:Tensor, start_pos:int = 0) -> Tensor:
  """Single-head causal attention, fused (no score materialization). q:[T,Hd]  k,v:[KV,Hd] with KV=start_pos+T.
  Returns [T,Hd] fp16. Exact vs SDPA up to fp reassociation. Inputs should be contiguous (Phase-1 invariant)."""
  T, Hd = q.shape; KV = k.shape[0]
  scale = 1.0 / math.sqrt(Hd)
  PO = Tensor.empty(T * (Hd + 1), dtype=_F32).custom_kernel(
    q.contiguous(), k.contiguous(), v.contiguous(), fxn=maxpartial_kernel(T, KV, Hd, scale, start_pos))[0]
  O = Tensor.empty(T * Hd, dtype=dtypes.float16).custom_kernel(PO, fxn=combine_kernel(T, Hd))[0]
  return O.reshape(T, Hd)
