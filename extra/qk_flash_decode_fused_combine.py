#!/usr/bin/env python3
"""Fused-flash decode with IN-KERNEL LSE combine (attention-combine parity lever, route b).

The default gqa_coop_vec flash decode computes per-split partials (flash_partial over Hq*S workgroups) then
combines the S splits with 3 SEPARATE reduce kernels (flash_gmax/den/combine) = the ~12-24% attention_combine
reduce bucket. This kernel removes those 3 combine kernels by putting the S splits as WAVES in one workgroup per
head and doing the log-sum-exp (online-softmax) merge IN LDS -> out[h,:] directly. No external reduce, no `po`
global buffer.

GENERATED UOp kernel (native primitives: REG/LDS placeholders, s_barrier, cross-lane warp reduce) — NOT a
handwritten kernel. Reuses the fused_xlane online-softmax carry (acc/den/mx) + the LDS-int-slot combine pattern.

Layout: q[Hq*Hd] (f32/f16), cache[2*Hkv*MAXC*Hd] (K at 0, V at 1; f16), out[Hq*Hd] (f32).
Workgroup per head h (Hq groups); Smax waves x 32 lanes; wave w handles split w (idle if w*L>=Tc); lane d-shards
R=Hd/32 columns. Combine is unrolled over Smax (concrete).
"""
from __future__ import annotations
import math
from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.dtype import AddrSpace, dtypes
from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged

_F32 = dtypes.float32
_LOG2E = 1.4426950408889634
LANES = 32


def _fexp(x:UOp) -> UOp: return (x * UOp.const(_F32, _LOG2E)).exp2()
def _fc(v:float) -> UOp: return UOp.const(_F32, v)
def _ceildiv(a:int, b:int) -> int: return (a + b - 1) // b


def flash_decode_fused_combine_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, Tc):
  if Hd % LANES != 0: raise ValueError(f"need Hd%{LANES}==0, got {Hd}")
  G = Hq // Hkv
  R = Hd // LANES
  Smax = _ceildiv(MAXC, L)
  scale = 1.0 / (Hd ** 0.5)

  def kernel(out:UOp, q:UOp, cache:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    kvh = h // G
    tid = UOp.special(Smax * LANES, "lidx0")
    wave = tid // LANES                 # split index 0..Smax-1
    lane = tid % LANES                  # 0..31, d-shard

    # per-wave online-softmax state (this lane's R d-cols; den/mx scalar)
    acc = UOp.placeholder((R,), _F32, 200, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 201, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((1,), _F32, 202, addrspace=AddrSpace.REG)
    za = UOp.range(R, 1)
    init = acc.after(h)[za].store(0.0).end(za)
    init = den.after(init)[0].store(0.0)
    init = mx.after(init)[0].store(-float("inf"))
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)

    # online softmax over this wave's split (L keys)
    j = UOp.range(L, 2, axis_type=AxisType.REDUCE)
    t = wave * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    # score = q[h] . k[t] (d-shard + cross-lane reduce)
    dotp = UOp.placeholder((1,), _F32, 206, addrspace=AddrSpace.REG)
    dinit = dotp.after(h, j)[0].store(0.0)
    dotp = dotp.after(dinit)
    re = UOp.range(R, 3, axis_type=AxisType.REDUCE)
    e = lane * R + re
    qv = q[h * Hd + e].cast(_F32)
    kv = cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(_F32)
    dupd = dotp[0].store(dotp.after(re)[0] + qv * kv).end(re)
    partial = dotp.after(dupd)[0]
    sc_full = _warp_reduce_sum_staged(partial, lane, LANES) * _fc(scale)
    sc = in_r.where(sc_full, _fc(-float("inf")))
    old_m = mx.after(j)[0]
    new_m = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
    p = in_r.where(_fexp(sc - new_m), _fc(0.0))
    dd = UOp.range(R, 4)
    d = lane * R + dd
    vd = cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + d].cast(_F32)
    accu = acc[dd].store(acc.after(j)[dd] * corr + p * vd).end(dd)
    denu = den.after(accu)[0].store(den.after(j)[0] * corr + p)
    mxu = mx.after(denu)[0].store(new_m).end(j)
    accf, denf, mxf = acc.after(mxu), den.after(mxu), mx.after(mxu)

    # write this wave's partial to LDS (int slots)
    lds_acc = UOp.placeholder((Smax * Hd,), _F32, 210, addrspace=AddrSpace.LOCAL)
    lds_den = UOp.placeholder((Smax,), _F32, 211, addrspace=AddrSpace.LOCAL)
    lds_mx = UOp.placeholder((Smax,), _F32, 212, addrspace=AddrSpace.LOCAL)
    dw = UOp.range(R, 5)
    aw = lds_acc.after(h)[wave * Hd + lane * R + dw].store(accf[dw]).end(dw)
    lw = lds_den.after(aw)[wave].store(denf[0], lane.eq(0))
    mw = lds_mx.after(lw)[wave].store(mxf[0], lane.eq(0))
    bar = mw.barrier()

    # in-kernel LSE combine across the Smax waves (unrolled; Smax concrete). gm = max_w mx_w;
    # den_tot = sum_w den_w*exp(mx_w-gm); num[d] = sum_w acc_w[d]*exp(mx_w-gm); out = num/den_tot.
    def _mxw(w): return lds_mx.after(bar)[w].load()
    def _denw(w): return lds_den.after(bar)[w].load()
    gm = _mxw(0)
    for w in range(1, Smax): gm = gm.maximum(_mxw(w))
    den_tot = _denw(0) * _fexp(_mxw(0) - gm)
    for w in range(1, Smax): den_tot = den_tot + _denw(w) * _fexp(_mxw(w) - gm)

    dd2 = UOp.range(R, 6)
    d2 = lane * R + dd2
    num = lds_acc.after(bar)[0 * Hd + d2].load() * _fexp(_mxw(0) - gm)
    for w in range(1, Smax): num = num + lds_acc.after(bar)[w * Hd + d2].load() * _fexp(_mxw(w) - gm)
    res = num / den_tot
    return out[h * Hd + d2].store(res, wave < 1).end(dd2).end(h).sink(
      arg=KernelInfo(name=f"flash_fused_combine_{Hq}_{Hd}", opts_to_apply=()))

  return kernel


def flash_decode_fused_combine(q, cache_kv, Tc_b, Tc_u, Hd:int, Hq:int, Hkv:int, MAXC:int, L:int=256):
  """Wrapper: one fused kernel per head -> out[Hq,Hd]. No external combine, no `po` global buffer."""
  from tinygrad import Tensor
  q_f = q.reshape(Hq * Hd)
  cache_f = cache_kv.reshape(2 * Hkv * MAXC * Hd)
  out = Tensor.empty(Hq * Hd, dtype=_F32, device=q.device).custom_kernel(
    q_f, cache_f, fxn=flash_decode_fused_combine_kernel(Hd, Hq, Hkv, MAXC, L, Tc_u))[0]
  return out.reshape(Hq, Hd)


def _microgate():
  import numpy as np
  from tinygrad import Tensor
  Hq, Hkv, Hd, MAXC, L, Tc = 8, 2, 128, 1024, 256, 600
  G = Hq // Hkv
  rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float32) * 0.3
  K = (rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float32) * 0.3)
  V = (rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float32) * 0.3)
  # reference
  ref = np.zeros((Hq, Hd), np.float32)
  for hh in range(Hq):
    kv = hh // G
    sc = (q[hh] @ K[kv, :Tc].T) / math.sqrt(Hd)
    w = np.exp(sc - sc.max()); w /= w.sum()
    ref[hh] = w @ V[kv, :Tc]
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float32)
  cache[0] = K; cache[1] = V
  qf = Tensor(q.reshape(Hq * Hd), device="AMD").cast(dtypes.float16).contiguous()
  cf = Tensor(cache.reshape(2 * Hkv * MAXC * Hd), device="AMD").cast(dtypes.float16).contiguous()
  got = Tensor.empty(Hq * Hd, dtype=_F32, device="AMD").custom_kernel(
    qf, cf, fxn=flash_decode_fused_combine_kernel(Hd, Hq, Hkv, MAXC, L, Tc))[0].numpy().reshape(Hq, Hd)
  denom = np.abs(ref).mean() + 1e-9
  rel_rmse = float(np.sqrt(((got - ref) ** 2).mean()) / denom)
  print(f"flash fused-combine microgate: rel_rmse={rel_rmse:.2e}")
  ok = rel_rmse < 2e-2   # f16 K/V -> looser tol than the exact GEMV combine
  print("INKLSE_MICROGATE_PASS" if ok else "INKLSE_MICROGATE_FAIL")
  return ok


if __name__ == "__main__":
  import sys
  sys.exit(0 if _microgate() else 2)
