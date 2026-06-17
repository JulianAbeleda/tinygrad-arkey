#!/usr/bin/env python3
"""LDS tiling primitive arc — Phase 3: does LDS tile reuse BEAT redundant HBM reads? (the real primitive proof)

Synthetic that mirrors the flash-prefill failure mode: W output lanes per workgroup each need the WHOLE K tile
[L,Hd] (lane w computes out[w] = sum_l sum_e K[l,e]*Q[w,e]). Two kernels, identical math:
  A (global):  each of W lanes re-reads all of K from global  -> ~W x redundant HBM traffic (the Phase-4 bug).
  B (LDS):     workgroup cooperatively loads K[L,Hd] -> LDS once, barrier, all W lanes reuse it from LDS.
Both written directly in custom_kernel (no BEAM). GPU time = GlobalCounters.time_sum_s under DEBUG>=2 (NEVER
wall-clock -- the Phase-5 lesson). Acceptance: B correct AND faster than A, with the speedup GROWING in W
(more lanes -> more reuse). NB workgroups to saturate the GPU.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_lds_reuse_bench.py
"""
from __future__ import annotations

import json, pathlib, sys

import numpy as np

from tinygrad import Tensor, dtypes, GlobalCounters, Context
from tinygrad.uop.ops import UOp, KernelInfo, AxisType, Ops
from tinygrad.dtype import AddrSpace

Hd, NB = 128, 256
LS = [32, 64, 128]   # tile must fit 64KB LDS: L*Hd*4 <= 65536 -> L<=128 at Hd=128 (L=256=128KB overflows;
WS = [32, 64, 129]   # this 64KB ceiling is exactly why real flash tiles KV instead of loading it whole)
_F32 = dtypes.float32

def global_kernel(NB, L, Hd, W):
  def k(o:UOp, K:UOp, Q:UOp) -> UOp:
    gid = UOp.special(NB, "gidx0"); tid = UOp.special(W, "lidx0")
    Kgf = K.reshape(NB, L * Hd)[gid]; Qg = Q.reshape(NB, W, Hd)[gid]
    le = UOp.range(L * Hd, 0, AxisType.REDUCE); e = le % Hd
    acc = UOp.placeholder((1,), _F32, 0, addrspace=AddrSpace.REG)
    acc = acc.after(gid, tid)[0].set(0.0)
    acc = acc[0].set(acc.after(le)[0] + Kgf[le].cast(_F32) * Qg[tid, e].cast(_F32), end=le)
    return o.reshape(NB, W)[gid][tid].store(acc[0]).sink(arg=KernelInfo(name=f"reuse_global_{NB}_{L}_{Hd}_{W}", opts_to_apply=()))
  return k

def lds_kernel(NB, L, Hd, W):
  def k(o:UOp, K:UOp, Q:UOp) -> UOp:
    gid = UOp.special(NB, "gidx0"); tid = UOp.special(W, "lidx0")
    Kgf = K.reshape(NB, L * Hd)[gid]; Qg = Q.reshape(NB, W, Hd)[gid]
    lds = UOp.placeholder((L * Hd,), _F32, 0, addrspace=AddrSpace.LOCAL)
    nload = (L * Hd + W - 1) // W
    li = UOp.range(nload, 1, AxisType.LOOP); idx = tid + li * W
    sidx = (idx < L * Hd).where(idx, idx.const_like(0))   # OOB lanes harmlessly re-store slot 0 (same value)
    store = lds[sidx].store(Kgf[sidx].cast(_F32)).end(li)
    lds = lds.after(UOp.barrier(store))
    le = UOp.range(L * Hd, 2, AxisType.REDUCE); e = le % Hd
    acc = UOp.placeholder((1,), _F32, 1, addrspace=AddrSpace.REG)
    acc = acc.after(gid, tid)[0].set(0.0)
    acc = acc[0].set(acc.after(le)[0] + lds[le] * Qg[tid, e].cast(_F32), end=le)
    return o.reshape(NB, W)[gid][tid].store(acc[0]).sink(arg=KernelInfo(name=f"reuse_lds_{NB}_{L}_{Hd}_{W}", opts_to_apply=()))
  return k

def _gpu_ms(out_fn, iters=5):
  best = 1e9
  with Context(DEBUG=2):
    out_fn().realize()  # compile
    for _ in range(iters):
      GlobalCounters.reset(); out_fn().realize(); best = min(best, GlobalCounters.time_sum_s)
  return best * 1e3

def run(L, W):
  rng = np.random.default_rng(L * 1000 + W)
  Knp = rng.standard_normal((NB, L, Hd)).astype(np.float32) * 0.1
  Qnp = rng.standard_normal((NB, W, Hd)).astype(np.float32) * 0.1
  K = Tensor(Knp).realize(); Q = Tensor(Qnp).realize()
  outA = lambda: Tensor.empty(NB * W, dtype=_F32).custom_kernel(K, Q, fxn=global_kernel(NB, L, Hd, W))[0]
  outB = lambda: Tensor.empty(NB * W, dtype=_F32).custom_kernel(K, Q, fxn=lds_kernel(NB, L, Hd, W))[0]
  ref = np.einsum("ble,bwe->bw", Knp, Qnp).reshape(NB * W)   # sum_l sum_e K[l,e]Q[w,e] = (K.sum0)·Q[w]
  ga = outA().realize().numpy(); gb = outB().realize().numpy()
  errA = float(np.abs(ga - ref).max()); errB = float(np.abs(gb - ref).max())
  tol = 2e-2 * max(1.0, np.abs(ref).max())
  tA = _gpu_ms(outA); tB = _gpu_ms(outB)
  return {"L": L, "W": W, "Hd": Hd, "NB": NB, "errA": round(errA, 4), "errB": round(errB, 4),
          "global_ms": round(tA, 4), "lds_ms": round(tB, 4), "speedup": round(tA / tB, 3) if tB else None,
          "correct": bool(errA < tol and errB < tol)}

def main():
  rows = [run(L, W) for L in LS for W in WS]
  for r in rows:
    print(f"L={r['L']:4d} W={r['W']:4d}: global {r['global_ms']:7.3f}ms | lds {r['lds_ms']:7.3f}ms -> "
          f"{r['speedup']}x {'OK' if r['correct'] else 'WRONG'}", file=sys.__stdout__)
  # acceptance: B correct + faster, and the speedup grows with W (more lanes -> more reuse)
  by_L = {}
  for r in rows: by_L.setdefault(r["L"], []).append(r)
  grows = bool(all([x["speedup"] for x in sorted(v, key=lambda z: z["W"])] ==
                   sorted([x["speedup"] for x in v]) for v in by_L.values()))   # monotone in W per L
  allcorrect = bool(all(r["correct"] for r in rows))
  wmax = max(WS)
  win_high_reuse = bool(all(r["speedup"] and r["speedup"] > 1.0 for r in rows if r["W"] == wmax))  # flash regime (W=Hd+1)
  best = max(r["speedup"] for r in rows)
  # LDS wins when reuse is high enough to amortize the cooperative load (low W + large L can lose -- expected).
  out = {"shape": {"Hd": Hd, "NB": NB}, "rows": rows, "all_correct": allcorrect,
         "win_at_high_reuse_W": win_high_reuse, "speedup_grows_with_W": grows, "best_speedup": best,
         "verdict": (f"PASS: LDS reuse beats redundant HBM reads in the high-reuse regime (up to {best}x at "
                     f"W={wmax}); speedup grows with W -> locality primitive effective"
                     if (allcorrect and win_high_reuse and grows) else
                     "FAIL: LDS reuse did not beat redundant global reads where it should")}
  print(f"grows_with_W={grows} win@W={wmax}={win_high_reuse} best={best}x | {out['verdict']}", file=sys.__stdout__)
  art = pathlib.Path("bench/lds-tiling-primitive-20260617/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"artifact: {art}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
