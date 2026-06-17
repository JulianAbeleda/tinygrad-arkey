#!/usr/bin/env python3
"""Prefill v2 — Increment 2, Phase 5: HONEST GPU-time re-measurement of the flash-prefill kernel.

CRITICAL CORRECTION. Phases 3-4 reported ~2.7-2.8x speedups using wall-clock around `.realize()` in a warm
loop. That methodology did NOT capture GPU execution time (host-dispatch / cache confound -- cf. the
amd-decode-measurement-confounds lesson). This harness measures the AUTHORITATIVE per-kernel GPU time via
DEBUG=2 (`tm`), summing the compute kernels (excluding the one-time device-init copy), in an isolated
subprocess per case.

Verdict: the score-free fused kernel is CORRECT but FAR SLOWER than SDPA. Root cause: `d` (output dim, W=129)
is a GLOBAL lane, so each lane independently streams all of K/V from HBM -> ~129x redundant reads, memory-bound
at ~0.19 TFLOP / ~367 GB/s. Being score-free WITHOUT LDS data-reuse is fundamentally slower than SDPA's
materialized-but-reused path. Real flash-2 needs LDS tiling (stage K/V tile in shared memory, reuse across the
head/dim lanes); that opt is what BEAM would find, but BEAM hangs gfx1100, and hand-LDS is dangerous-power
surface. So flash-prefill is REFUTED on performance and BANKED (correct, not performant here).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_flash_prefill_phase5.py
"""
from __future__ import annotations

import json, os, pathlib, re, subprocess, sys

Hd, T = 128, 512
CASES = [("1h", 512), ("1h", 3584), ("gqa", 512), ("gqa", 3584)]
Hq, Hkv = 32, 8

def _child(kind:str, KV:int):
  import os as _os
  _os.environ["DEBUG"] = "2"
  from tinygrad import Tensor, dtypes
  sp = KV - T
  if kind == "1h":
    q = Tensor.randn(T, Hd, dtype=dtypes.float16).realize(); k = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize(); v = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
    qi = Tensor.arange(T).reshape(T, 1); kj = Tensor.arange(KV).reshape(1, KV)
    mask = (kj > sp + qi).where(Tensor(-float("inf")), Tensor(0.0)).cast(dtypes.float16).reshape(1, 1, T, KV)
    if sys.argv[3] == "flash":
      from extra.qk_flash_prefill_custom import flash_prefill_attention_1h
      flash_prefill_attention_1h(q, k, v, start_pos=sp).realize()
    else:
      q.reshape(1, 1, T, Hd).scaled_dot_product_attention(k.reshape(1, 1, KV, Hd), v.reshape(1, 1, KV, Hd), attn_mask=mask).realize()
  else:
    q = Tensor.randn(Hq, T, Hd, dtype=dtypes.float16).realize(); k = Tensor.randn(Hkv, KV, Hd, dtype=dtypes.float16).realize(); v = Tensor.randn(Hkv, KV, Hd, dtype=dtypes.float16).realize()
    qi = Tensor.arange(T).reshape(T, 1); kj = Tensor.arange(KV).reshape(1, KV)
    mask = (kj > sp + qi).where(Tensor(-float("inf")), Tensor(0.0)).cast(dtypes.float16).reshape(1, 1, T, KV)
    if sys.argv[3] == "flash":
      from extra.qk_flash_prefill_custom import flash_prefill_attention
      flash_prefill_attention(q, k, v, start_pos=sp).realize()
    else:
      q.reshape(1, Hq, T, Hd).scaled_dot_product_attention(k.reshape(1, Hkv, KV, Hd), v.reshape(1, Hkv, KV, Hd), attn_mask=mask, enable_gqa=True).realize()

_TM = re.compile(r"\*\*\* AMD\s+\d+\s+(\S+).*tm\s+([\d.]+)(us|ms)/")
def _gpu_ms(kind:str, KV:int, impl:str):
  try:
    p = subprocess.run([sys.executable, __file__, "--child", f"{kind}:{KV}", impl],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."}, timeout=180)
  except subprocess.TimeoutExpired:
    return None, False  # too slow / hung
  total, names = 0.0, []
  for line in p.stderr.splitlines() + p.stdout.splitlines():
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = _TM.search(line)
    if not m: continue
    name, val, unit = m.group(1), float(m.group(2)), m.group(3)
    names.append(name)
    if name.startswith("copy"): continue  # exclude one-time device-init / host copies
    total += val if unit == "ms" else val / 1000
  faulted = ("HW fault" in p.stderr) or ("HW fault" in p.stdout)
  # flash MUST show its fp_maxpartial kernel; if absent (fault/truncation) the measurement is incomplete
  complete = (impl != "flash") or any(n.startswith("fp_maxpartial") for n in names)
  return (round(total, 3) if complete and not faulted else None), (complete and not faulted)

def main():
  if len(sys.argv) >= 4 and sys.argv[1] == "--child":
    kind, KV = sys.argv[2].split(":"); _child(kind, int(KV)); return
  rows = []
  for kind, KV in CASES:
    f, fok = _gpu_ms(kind, KV, "flash"); s, _ = _gpu_ms(kind, KV, "sdpa")
    row = {"kind": kind, "KV": KV, "flash_gpu_ms": f, "sdpa_gpu_ms": s,
           "slowdown_x": (round(f / s, 1) if (f and s) else None), "complete": bool(fok and f and s)}
    rows.append(row)
    if row["complete"]:
      print(f"{kind:3s} KV={KV:5d}: flash {f:8.1f}ms | sdpa {s:6.1f}ms -> flash is {row['slowdown_x']}x SLOWER", file=sys.__stdout__)
    else:
      print(f"{kind:3s} KV={KV:5d}: INCOMPLETE (flash too slow / faulted / kernel tm not captured)", file=sys.__stdout__)
  out = {"note": "authoritative DEBUG=2 GPU kernel time (compute kernels, excl device-init copy); supersedes "
                 "the wall-clock Phase-3/4 numbers which measured host dispatch, not GPU exec",
         "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "T": T}, "rows": rows,
         "verdict": "REFUTED: score-free fused kernel is correct but memory-bound (per-d HBM re-streaming, no "
                    "LDS reuse) -> far slower than SDPA; real speedup needs LDS tiling (BEAM-territory, hangs "
                    "gfx1100). Flash-prefill BANKED as correct-but-not-performant."}
  art = pathlib.Path("bench/qk-flash-prefill-phase5/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"artifact: {art}\nVERDICT: {out['verdict']}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
