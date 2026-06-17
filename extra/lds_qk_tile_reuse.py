#!/usr/bin/env python3
"""LDS tiling primitive arc — Phase 4: prove q·k TILE REUSE (locality inside the real attention math).

Phase 3 proved LDS reuse beats redundant HBM reads on a synthetic. Phase 4 puts it inside the actual q·k
structure: per workgroup, T query lanes each compute scores against the SAME K tile [L,Hd]
(score[t,l] = sum_e Q[t,e]*K[l,e]). This is exactly the flash-prefill reread failure mode (each query/d lane
re-read K). Two custom-kernel variants, identical math:
  baseline: each of T lanes re-reads all of K[L,Hd] from global  -> ~T x redundant HBM traffic.
  LDS:      workgroup cooperatively loads K[L,Hd] -> LDS once, barrier, T lanes reuse it for the dot.
NOT softmax, NOT V accumulation, NOT model integration. GPU time = GlobalCounters.time_sum_s under DEBUG>=2.

64 KB LDS ceiling: L*Hd*4 <= 65536 -> L<=128 at Hd=128 (designed in). Run:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/lds_qk_tile_reuse.py
"""
from __future__ import annotations

import json, pathlib, sys

import numpy as np

from tinygrad import Tensor, dtypes, GlobalCounters, Context
from tinygrad.uop.ops import UOp, KernelInfo, AxisType, Ops
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import compile_linear

Hd, NB = 128, 256
LS = [64, 128]
TS = [16, 32]
_F32 = dtypes.float32

def qk_global_kernel(NB, L, Hd, T):
  def k(o:UOp, K:UOp, Q:UOp) -> UOp:                       # o:[NB,T,L]  K:[NB,L,Hd]  Q:[NB,T,Hd]
    gid = UOp.special(NB, "gidx0"); tid = UOp.special(T, "lidx0")
    Kg = K.reshape(NB, L, Hd)[gid]; Qg = Q.reshape(NB, T, Hd)[gid]; og = o.reshape(NB, T, L)[gid][tid]
    score = UOp.placeholder((L,), _F32, 0, addrspace=AddrSpace.REG)
    ii = UOp.range(L, 5); score = score.after(score[ii].store(0.0).end(ii))   # zero-init regs
    e = UOp.range(Hd, 6, AxisType.REDUCE); l = UOp.range(L, 7)
    acc = score[l].store(score.after(e)[l] + Qg[tid, e].cast(_F32) * Kg[l, e].cast(_F32)).end(l).end(e)
    lo = UOp.range(L, 8)
    return og[lo].store(score.after(acc)[lo]).end(lo).sink(arg=KernelInfo(name=f"qk_global_{NB}_{L}_{Hd}_{T}", opts_to_apply=()))
  return k

def qk_lds_kernel(NB, L, Hd, T):
  def k(o:UOp, K:UOp, Q:UOp) -> UOp:
    gid = UOp.special(NB, "gidx0"); tid = UOp.special(T, "lidx0")
    Kgf = K.reshape(NB, L * Hd)[gid]; Qg = Q.reshape(NB, T, Hd)[gid]; og = o.reshape(NB, T, L)[gid][tid]
    lds = UOp.placeholder((L * Hd,), _F32, 0, addrspace=AddrSpace.LOCAL)
    nload = (L * Hd + T - 1) // T
    li = UOp.range(nload, 1, AxisType.LOOP); idx = tid + li * T
    sidx = (idx < L * Hd).where(idx, idx.const_like(0))    # OOB lanes harmlessly re-store slot 0
    store = lds[sidx].store(Kgf[sidx].cast(_F32)).end(li)
    ldsK = lds.after(UOp.barrier(store)).reshape(L, Hd)
    score = UOp.placeholder((L,), _F32, 1, addrspace=AddrSpace.REG)
    ii = UOp.range(L, 5); score = score.after(score[ii].store(0.0).end(ii))   # zero-init regs
    e = UOp.range(Hd, 6, AxisType.REDUCE); l = UOp.range(L, 7)
    acc = score[l].store(score.after(e)[l] + Qg[tid, e].cast(_F32) * ldsK[l, e]).end(l).end(e)
    lo = UOp.range(L, 8)
    return og[lo].store(score.after(acc)[lo]).end(lo).sink(arg=KernelInfo(name=f"qk_lds_{NB}_{L}_{Hd}_{T}", opts_to_apply=()))
  return k

def _gpu_ms(out_fn, iters=5):
  best = 1e9
  with Context(DEBUG=2):
    out_fn().realize()
    for _ in range(iters):
      GlobalCounters.reset(); out_fn().realize(); best = min(best, GlobalCounters.time_sum_s)
  return best * 1e3

def _emits_lds(out_fn) -> bool:
  compiled = compile_linear(out_fn().schedule_linear())
  for call in compiled.src:
    p = call.src[0]
    if p.op is Ops.PROGRAM and "qk_lds" in p.arg.name:
      src = next((u.arg for u in p.toposort() if u.op is Ops.SOURCE), "")
      return "shared" in src and ("barrier" in src or "s_barrier" in src)
  return False

def run(L, T):
  rng = np.random.default_rng(L * 100 + T)
  Knp = (rng.standard_normal((NB, L, Hd)) * 0.1).astype(np.float32)
  Qnp = (rng.standard_normal((NB, T, Hd)) * 0.1).astype(np.float32)
  K = Tensor(Knp).realize(); Q = Tensor(Qnp).realize()
  gfn = lambda: Tensor.empty(NB * T * L, dtype=_F32).custom_kernel(K, Q, fxn=qk_global_kernel(NB, L, Hd, T))[0]
  lfn = lambda: Tensor.empty(NB * T * L, dtype=_F32).custom_kernel(K, Q, fxn=qk_lds_kernel(NB, L, Hd, T))[0]
  ref = np.einsum("bte,ble->btl", Qnp, Knp).reshape(-1)
  eg = float(np.abs(gfn().realize().numpy() - ref).max()); el = float(np.abs(lfn().realize().numpy() - ref).max())
  tol = 2e-2 * max(1.0, np.abs(ref).max())
  tg = _gpu_ms(gfn); tl = _gpu_ms(lfn)
  # bytes read estimate (fp32): baseline rereads K per lane; LDS loads K once/workgroup
  gbytes_base = NB * T * L * Hd * 4; gbytes_lds = NB * L * Hd * 4
  return {"L": L, "T": T, "Hd": Hd, "NB": NB, "errG": round(eg, 4), "errL": round(el, 4),
          "global_ms": round(tg, 4), "lds_ms": round(tl, 4), "speedup": round(tg / tl, 3) if tl else None,
          "global_K_bytes": gbytes_base, "lds_K_bytes": gbytes_lds, "K_traffic_ratio": round(gbytes_base / gbytes_lds, 1),
          "correct": bool(eg < tol and el < tol), "lds_emitted": bool(_emits_lds(lfn))}

def main():
  rows = [run(L, T) for L in LS for T in TS]
  for r in rows:
    print(f"L={r['L']:4d} T={r['T']:3d}: global {r['global_ms']:7.3f}ms | lds {r['lds_ms']:7.3f}ms -> "
          f"{r['speedup']}x | Ktraffic {r['K_traffic_ratio']}x {'OK' if r['correct'] else 'WRONG'} "
          f"lds_emitted={r['lds_emitted']}", file=sys.__stdout__)
  by_L = {}
  for r in rows: by_L.setdefault(r["L"], []).append(r)
  grows = bool(all([x["speedup"] for x in sorted(v, key=lambda z: z["T"])] == sorted([x["speedup"] for x in v])
                   for v in by_L.values()))
  allcorrect = bool(all(r["correct"] for r in rows)); emitted = bool(all(r["lds_emitted"] for r in rows))
  tmax = max(TS); win = bool(all(r["speedup"] and r["speedup"] > 1.0 for r in rows if r["T"] == tmax))
  best = max(r["speedup"] for r in rows)
  out = {"shape": {"Hd": Hd, "NB": NB}, "rows": rows, "all_correct": allcorrect, "lds_emitted": emitted,
         "win_at_high_reuse_T": win, "speedup_grows_with_T": grows, "best_speedup": best,
         "verdict": (f"PASS: LDS q.k tile reuse beats global reread (up to {best}x at T={tmax}); grows with T; "
                     f"shared+barrier emitted -> Phase 5 one-tile attention justified"
                     if (allcorrect and emitted and win and grows) else
                     "FAIL: LDS q.k reuse did not beat global reread where it should")}
  print(f"grows_with_T={grows} win@T={tmax}={win} emitted={emitted} best={best}x | {out['verdict']}", file=sys.__stdout__)
  art = pathlib.Path("bench/lds-tiling-primitive-20260617/phase4-qk/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"artifact: {art}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
