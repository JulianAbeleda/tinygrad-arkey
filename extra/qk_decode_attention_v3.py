#!/usr/bin/env python3
# STATUS: v3 REFUTED on perf at decode shapes (docs/qk-decode-attention-v3-result-20260617.md). KEPT for the
# hoisted-baseline measurement method (Phase 0) only; the WMMA/LDS v3 kernel was not built (regime mismatch).
"""decode_attention_v3 — isolated build + perf gate (Phase 0 baseline first).

Phase 0: measure the CURRENT shipped hoisted flash-decode primitive (the real baseline v3 must beat by >=1.3x)
at true decode shapes (T=1, Hq=32, Hkv=8, Hd=128, KV in {512,1024,2048,4096}), plus a numpy SDPA correctness
reference. GPU time = DEBUG=2 device tm (the locked method), warm, summed over the attention kernels.

The v3 candidate (Phase 1) is benchmarked against THIS hoisted number, not against SDPA.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_v3.py
"""
from __future__ import annotations
import io, json, os, pathlib, re, sys, contextlib, math
import numpy as np

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+(\S+)\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us")
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4608
G = Hq // Hkv
KVS = [512, 1024, 2048, 4096]

def _gpu_tm(realize_fn, warm=3):
  from tinygrad import Context, GlobalCounters
  for _ in range(warm): realize_fn()
  best = None
  for _ in range(5):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); realize_fn()
    tot, dev = 0.0, set()
    for l in buf.getvalue().splitlines():
      if (m := _LINE.search(_ANSI.sub("", l))): tot += float(m.group(3)); dev.add(m.group(1))
    best = tot if best is None else min(best, tot)
  return best, sorted(dev)

def hoisted_baseline():
  from tinygrad import Tensor, UOp, dtypes
  from extra.qk_flash_decode import flash_decode_attention
  rng = np.random.default_rng(0)
  qn = rng.standard_normal((Hq, Hd)).astype(np.float16)
  kn = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  vn = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  q, k, v = Tensor(qn).realize(), Tensor(kn).realize(), Tensor(vn).realize()
  rows = []
  for KV in KVS:
    sp_b = UOp.variable("start_pos", 0, MAXC - 1).bind(KV - 1); sp_u = UOp.variable("start_pos", 0, MAXC - 1)
    def build(): return flash_decode_attention(q, k, v, sp_b + 1, sp_u + 1, Hd, Hq, Hkv, MAXC, 128, variant="hoisted")
    got = build().numpy()
    # SDPA correctness ref
    qf = qn.astype(np.float32); ref = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kv = h // G; sc = (qf[h] @ kn[kv, :KV].astype(np.float32).T) / math.sqrt(Hd)
      p = np.exp(sc - sc.max()); p /= p.sum(); ref[h] = p @ vn[kv, :KV].astype(np.float32)
    rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9))
    tm, dev = _gpu_tm(lambda: build().realize())
    rows.append({"KV": KV, "hoisted_gpu_us": round(tm, 1), "rel_err_vs_sdpa": round(rel, 5), "devices": dev})
    print(f"  KV={KV:5}: hoisted {tm:7.1f}us  rel_err {rel:.2g}  dev={dev}", file=sys.stderr)
  return rows

def main():
  from tinygrad import Device
  assert Device.DEFAULT == "AMD"
  print("=== Phase 0: hoisted flash baseline (the number v3 must beat by >=1.3x) ===", file=sys.stderr)
  rows = hoisted_baseline()
  out = {"shape": {"Hq": Hq, "Hkv": Hkv, "G": G, "Hd": Hd, "T": 1}, "kvs": KVS,
         "method": "DEBUG=2 device tm, warm, min-of-5; isolated flash_decode_attention(variant=hoisted, L=128)",
         "baseline": rows}
  art = pathlib.Path("bench/qk-decode-attention-v3-isolated/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"\nartifact: {art}", file=sys.__stderr__)
  print("@@DONE@@", file=sys.__stderr__)

if __name__ == "__main__":
  main()
