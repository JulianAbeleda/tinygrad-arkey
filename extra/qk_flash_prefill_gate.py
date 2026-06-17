#!/usr/bin/env python3
"""Prefill v2 — Increment 2, Stage 0 make-or-break gate: flash-style prefill attention.

Increment 1 made the dense FFN fast; attention (SDPA) is now the prefill bottleneck and grows with context
(8B warm v2 forward: 241ms @ sp=0 -> 1202ms @ sp=3072, attention ~51% there). tinygrad SDPA (tensor.py:1197)
materializes the full [T, start_pos+T] scores, softmaxes, then @v -> ~4% peak, memory-bound, and the symbolic
KV blocks the concrete-shape/warmstart-TC lever that fixed the FFN.

This gate compares, on the REAL 8B attention shapes (Hq=32, Hkv=8, Hd=128, T=512, causal, GQA, fp16), at
KV in {512, 1024, 3584} (i.e. start_pos in {0, 512, 3072}):
  1. baseline SDPA (the current path) -- anchor.
  2. KV-tiled online-softmax (flash-2) in tinygrad ops: fixed-L concrete tiles, running max/sum/acc, causal
     per tile. Each tile's q@k_tile^T is a concrete [T,L] matmul (warmstart-TC eligible) -- no full-score
     materialization. No custom kernel.

GATE: approach #2 must be EXACT vs SDPA (max abs err within fp16 tolerance) AND >=3x faster on the long-KV
shape. PASS -> wire it into _attention (Stage 1). FAIL -> consider a raw-HIP fused flash kernel, or bank
Increment 1 as-is (attention stays SDPA; the short-prompt 13x already holds).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_flash_prefill_gate.py
"""
from __future__ import annotations

import json, math, sys, time

Hq, Hkv, Hd, T = 32, 8, 128, 512
PEAK_TF = 83.6
KVS = [512, 1024, 3584]   # start_pos = KV - T
LTILE = 512               # KV tile size (concrete)

def _causal_mask(start_pos:int, KV:int):
  # query i (abs start_pos+i) attends key j iff j <= start_pos+i. matches model.py triu(start_pos+1).
  from tinygrad import Tensor, dtypes
  qi = Tensor.arange(T).reshape(T, 1)
  kj = Tensor.arange(KV).reshape(1, KV)
  return (kj > (start_pos + qi)).where(Tensor(-float("inf")), Tensor(0.0)).cast(dtypes.float16)

def sdpa(q, k, v, mask):
  return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)

def tiled_flash(q, k, v, start_pos:int, KV:int):
  # q:[1,Hq,T,Hd] k,v:[1,Hkv,KV,Hd] fp16. flash-2 online softmax over fixed-L KV tiles (fp32 accumulators).
  from tinygrad import Tensor, dtypes
  G = Hq // Hkv
  qh = q[0]                                       # [Hq,T,Hd]
  kh = k[0].repeat_interleave(G, 0)               # [Hq,KV,Hd] (GQA expand, like SDPA)
  vh = v[0].repeat_interleave(G, 0)
  scale = 1.0 / math.sqrt(Hd)
  m = Tensor.full((Hq, T, 1), -float("inf"), dtype=dtypes.float32)
  l = Tensor.zeros(Hq, T, 1, dtype=dtypes.float32)
  acc = Tensor.zeros(Hq, T, Hd, dtype=dtypes.float32)
  qi = Tensor.arange(T).reshape(1, T, 1)
  for t0 in range(0, KV, LTILE):
    t1 = min(t0 + LTILE, KV); L = t1 - t0
    kt = kh[:, t0:t1, :]; vt = vh[:, t0:t1, :]                       # [Hq,L,Hd]
    s = (qh @ kt.transpose(1, 2)).cast(dtypes.float32) * scale       # [Hq,T,L] concrete matmul
    kj = Tensor.arange(L).reshape(1, 1, L) + t0
    s = (kj > (start_pos + qi)).where(Tensor(-float("inf")), s)      # causal mask for this tile
    m_new = m.maximum(s.max(axis=-1, keepdim=True))                  # [Hq,T,1]
    corr = (m - m_new).exp()
    p = (s - m_new).exp()                                            # [Hq,T,L]
    l = l * corr + p.sum(axis=-1, keepdim=True)
    acc = acc * corr + (p.cast(dtypes.float16) @ vt).cast(dtypes.float32)   # [Hq,T,Hd]
    m = m_new
  out = (acc / l).cast(dtypes.float16)                              # [Hq,T,Hd]
  return out.reshape(1, Hq, T, Hd)

def _time(fn, iters=8):
  from tinygrad import GlobalCounters
  fn().realize()
  ts = []
  for _ in range(iters):
    GlobalCounters.reset(); t0 = time.perf_counter(); fn().realize(); ts.append(time.perf_counter() - t0)
  return min(ts)

def run_shape(KV:int) -> dict:
  from tinygrad import Tensor, dtypes
  start_pos = KV - T
  Tensor.manual_seed(0)
  q = Tensor.randn(1, Hq, T, Hd, dtype=dtypes.float16).realize()
  k = Tensor.randn(1, Hkv, KV, Hd, dtype=dtypes.float16).realize()
  v = Tensor.randn(1, Hkv, KV, Hd, dtype=dtypes.float16).realize()
  mask = _causal_mask(start_pos, KV).reshape(1, 1, T, KV).realize()
  ref = sdpa(q, k, v, mask).realize()
  out = tiled_flash(q, k, v, start_pos, KV).realize()
  err = float((ref - out).abs().max().item())
  rmean = float(ref.abs().mean().item())
  t_sdpa = _time(lambda: sdpa(q, k, v, mask))
  t_tiled = _time(lambda: tiled_flash(q, k, v, start_pos, KV))
  flops = 2 * 2 * Hq * T * KV * Hd  # QK + PV (full; causal ~halves real work)
  return {"KV": KV, "start_pos": start_pos, "max_abs_err": round(err, 5), "ref_abs_mean": round(rmean, 5),
          "sdpa_ms": round(t_sdpa * 1e3, 2), "tiled_ms": round(t_tiled * 1e3, 2),
          "speedup": round(t_sdpa / t_tiled, 2),
          "sdpa_pct_peak": round(100 * flops / t_sdpa / 1e12 / PEAK_TF, 1),
          "tiled_pct_peak": round(100 * flops / t_tiled / 1e12 / PEAK_TF, 1)}

def main():
  out = {"shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "T": T, "Ltile": LTILE}, "rows": []}
  for KV in KVS:
    r = run_shape(KV); out["rows"].append(r)
    print(f"KV={KV:5d} (sp={r['start_pos']:5d}): err={r['max_abs_err']:.4f} (|ref|~{r['ref_abs_mean']:.3f}) "
          f"sdpa {r['sdpa_ms']:.1f}ms ({r['sdpa_pct_peak']}%) | tiled {r['tiled_ms']:.1f}ms "
          f"({r['tiled_pct_peak']}%) -> {r['speedup']}x", file=sys.__stdout__)
  TOL = 0.02  # fp16 attention tolerance (online softmax is fp-reassociative; dNLL is the real gate later)
  long = next(r for r in out["rows"] if r["KV"] == max(KVS))
  exact = all(r["max_abs_err"] <= TOL for r in out["rows"])
  fast = long["speedup"] >= 3.0
  out["gate_pass"] = exact and fast
  print("GATE:", f"PASS (tiled flash exact within {TOL} AND {long['speedup']}x on KV={long['KV']} -> Stage 1)"
        if out["gate_pass"] else
        f"FAIL (exact={exact} max_err_ok, long_speedup={long['speedup']}x) -> consider raw-HIP kernel or bank Inc1",
        file=sys.__stdout__)
  print(json.dumps(out))

if __name__ == "__main__":
  main()
