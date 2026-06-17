#!/usr/bin/env python3
"""LDS tiling primitive arc — Phase 5: prove ONE flash-attention tile (q.k + softmax + V) with K/V in LDS.

Does the proven LDS q.k reuse (Phase 4) survive adding softmax + V accumulation? Single KV tile, single head,
output [T,Hd]. Two variants of the SAME kernel: `lds` (K,V cooperatively loaded into LDS once, reused by all T
query lanes) vs `global` (each query lane re-reads K,V from HBM -- the Phase-5-correction failure mode). Each
query lane = one thread; per-lane output via the REG c_regs idiom. Online softmax (coupled m/l/acc) is
linearizer-rejected, so we use the sequential single-accumulator formulation: pass1 max, pass2 weighted-V with
a 1s-augmented denom, then combine. NOT full flash-prefill, NOT model integration.

64KB LDS: K+V in fp16 = 2*L*Hd*2 <= 65536 -> L<=128 at Hd=128 (designed in). GPU time = time_sum_s @ DEBUG>=2.
Reference: extra/gemm/amd_flash_attention.py (full LDS+WMMA flash). Run:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/lds_attention_tile.py
"""
from __future__ import annotations

import json, math, pathlib, sys

import numpy as np

from tinygrad import Tensor, dtypes, GlobalCounters, Context
from tinygrad.uop.ops import UOp, KernelInfo, AxisType, Ops
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import compile_linear

Hd, NB = 128, 256
LS = [64, 128]
TS = [16, 32]
_F32 = dtypes.float32
def _exp(x): return (x * 1.4426950408889634).exp2()

def attn_tile_kernel(NB, L, Hd, T, lds:bool, causal:bool):
  scale = 1.0 / math.sqrt(Hd); W = Hd + 1
  def k(O:UOp, K:UOp, Q:UOp, V:UOp) -> UOp:
    gid = UOp.special(NB, "gidx0"); tid = UOp.special(T, "lidx0")
    Qg = Q.reshape(NB, T, Hd)[gid]; Og = O.reshape(NB, T, Hd)[gid][tid]
    if lds:
      Kf = K.reshape(NB, L * Hd)[gid]; Vf = V.reshape(NB, L * Hd)[gid]
      ldsK = UOp.placeholder((L * Hd,), dtypes.float16, 0, addrspace=AddrSpace.LOCAL)
      ldsV = UOp.placeholder((L * Hd,), dtypes.float16, 1, addrspace=AddrSpace.LOCAL)
      nload = (L * Hd + T - 1) // T
      li = UOp.range(nload, 1, AxisType.LOOP); idx = tid + li * T; sidx = (idx < L * Hd).where(idx, idx.const_like(0))
      loads = UOp.group(ldsK[sidx].store(Kf[sidx]), ldsV[sidx].store(Vf[sidx])).end(li)
      bar = UOp.barrier(loads); KT = ldsK.after(bar).reshape(L, Hd); VT = ldsV.after(bar).reshape(L, Hd)
    else:
      KT = K.reshape(NB, L, Hd)[gid]; VT = V.reshape(NB, L, Hd)[gid]   # each lane re-reads from global
    def dot(l, rng):
      e = UOp.range(Hd, rng, AxisType.REDUCE)
      acc = UOp.placeholder((1,), _F32, rng, addrspace=AddrSpace.REG); acc = acc.after(l)[0].set(0.0)
      acc = acc[0].set(acc.after(e)[0] + Qg[tid, e].cast(_F32) * KT[l, e].cast(_F32), end=e)
      s = acc[0] * scale
      return (l > tid).where(UOp.const(_F32, -1e30), s) if causal else s
    # pass 1: m = max_l score
    l1 = UOp.range(L, 2, AxisType.REDUCE)
    m = UOp.placeholder((1,), _F32, 10, addrspace=AddrSpace.REG); m = m[0].set(-1e30)
    m = m[0].set(m.after(l1)[0].maximum(dot(l1, 3)), end=l1)
    # pass 2: out_reg[W] = sum_l exp(score - m) * Vaug  (1s-aug: col Hd = denom)
    out_reg = UOp.placeholder((W,), _F32, 11, addrspace=AddrSpace.REG)
    ii = UOp.range(W, 5); out_reg = out_reg.after(out_reg[ii].store(0.0).end(ii))
    l2 = UOp.range(L, 6, AxisType.REDUCE); d = UOp.range(W, 8)
    p = _exp(dot(l2, 7) - m[0])
    vaug = (d < Hd).where(VT[l2, (d < Hd).where(d, d.const_like(0))].cast(_F32), UOp.const(_F32, 1.0))
    acc = out_reg[d].store(out_reg.after(l2)[d] + p * vaug).end(d).end(l2)
    do = UOp.range(Hd, 9); fin = out_reg.after(acc)
    return Og[do].store((fin[do] / fin[Hd]).cast(dtypes.float16)).end(do).sink(
      arg=KernelInfo(name=f"attn_tile_{'lds' if lds else 'glb'}_{NB}_{L}_{Hd}_{T}{'_c' if causal else ''}", opts_to_apply=()))
  return k

def _gpu_ms(out_fn, iters=5):
  best = 1e9
  with Context(DEBUG=2):
    out_fn().realize()
    for _ in range(iters):
      GlobalCounters.reset(); out_fn().realize(); best = min(best, GlobalCounters.time_sum_s)
  return best * 1e3

def _emits_lds(out_fn) -> bool:
  for call in compile_linear(out_fn().schedule_linear()).src:
    p = call.src[0]
    if p.op is Ops.PROGRAM and "attn_tile_lds" in p.arg.name:
      src = next((u.arg for u in p.toposort() if u.op is Ops.SOURCE), "")
      return "shared" in src and ("barrier" in src or "s_barrier" in src)
  return False

def run(L, T, causal):
  rng = np.random.default_rng(L * 100 + T + (7 if causal else 0))
  Knp = (rng.standard_normal((NB, L, Hd)) * 0.3).astype(np.float16)
  Qnp = (rng.standard_normal((NB, T, Hd)) * 0.3).astype(np.float16)
  Vnp = (rng.standard_normal((NB, L, Hd)) * 0.3).astype(np.float16)
  K = Tensor(Knp).realize(); Q = Tensor(Qnp).realize(); V = Tensor(Vnp).realize()
  s = np.einsum("bte,ble->btl", Qnp.astype(np.float32), Knp.astype(np.float32)) / math.sqrt(Hd)
  if causal:
    mask = (np.arange(L)[None, :] > np.arange(T)[:, None])[None]
    s = np.where(mask, -np.inf, s)
  s = s - s.max(-1, keepdims=True); pr = np.exp(s); pr = pr / pr.sum(-1, keepdims=True)
  ref = np.einsum("btl,ble->bte", pr, Vnp.astype(np.float32)).reshape(-1)
  gfn = lambda: Tensor.empty(NB * T * Hd, dtype=dtypes.float16).custom_kernel(K, Q, V, fxn=attn_tile_kernel(NB, L, Hd, T, False, causal))[0]
  lfn = lambda: Tensor.empty(NB * T * Hd, dtype=dtypes.float16).custom_kernel(K, Q, V, fxn=attn_tile_kernel(NB, L, Hd, T, True, causal))[0]
  eg = float(np.abs(gfn().realize().numpy().astype(np.float32) - ref).max())
  el = float(np.abs(lfn().realize().numpy().astype(np.float32) - ref).max())
  tol = 2e-2 * max(1.0, np.abs(ref).max())
  tg = _gpu_ms(gfn); tl = _gpu_ms(lfn)
  return {"L": L, "T": T, "causal": causal, "errG": round(eg, 4), "errL": round(el, 4),
          "global_ms": round(tg, 4), "lds_ms": round(tl, 4), "speedup": round(tg / tl, 3) if tl else None,
          "correct": bool(eg < tol and el < tol), "lds_emitted": bool(_emits_lds(lfn))}

def main():
  rows = [run(L, T, c) for c in (False, True) for L in LS for T in TS]
  for r in rows:
    print(f"L={r['L']:4d} T={r['T']:3d} causal={int(r['causal'])}: global {r['global_ms']:7.3f}ms | "
          f"lds {r['lds_ms']:7.3f}ms -> {r['speedup']}x | err {max(r['errG'],r['errL'])} "
          f"{'OK' if r['correct'] else 'WRONG'} lds_emitted={r['lds_emitted']}", file=sys.__stdout__)
  allcorrect = bool(all(r["correct"] for r in rows)); emitted = bool(all(r["lds_emitted"] for r in rows))
  win = bool(all(r["speedup"] and r["speedup"] > 1.0 for r in rows))
  best = max(r["speedup"] for r in rows); causal_ok = bool(all(r["correct"] for r in rows if r["causal"]))
  out = {"shape": {"Hd": Hd, "NB": NB}, "rows": rows, "all_correct": allcorrect, "lds_emitted": emitted,
         "lds_faster_everywhere": win, "best_speedup": best, "causal_ok": causal_ok,
         "verdict": (f"PASS: LDS attention tile (q.k+softmax+V) correct + beats global-reread (up to {best}x), "
                     f"causal ok, shared+barrier emitted -> tiled multi-KV (Phase 6) justified"
                     if (allcorrect and emitted and win and causal_ok) else
                     "PARTIAL/FAIL: see rows (correct/emitted/faster/causal)")}
  print(f"correct={allcorrect} emitted={emitted} faster_everywhere={win} causal_ok={causal_ok} best={best}x\n{out['verdict']}", file=sys.__stdout__)
  art = pathlib.Path("bench/lds-tiling-primitive-20260617/phase5-attention-tile/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"artifact: {art}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
