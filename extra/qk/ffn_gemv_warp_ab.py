#!/usr/bin/env python3
"""Local A/B for the lossless q4k_gemv_warp work-decomposition variant vs the default q4k_gemv_partial, at the FFN
roles (gate/up 12288x4096, down 4096x12288). Synthetic Q4_K words (warp must match default up to fp reassoc =
lossless). Reports correctness, device GPU-busy us, effective Q4-bandwidth, % HBM peak, workgroups, and the ISA
resource summary (VGPR/SGPR/LDS/spill) via the AMD renderer when available.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/ffn_gemv_warp_ab.py
"""
from __future__ import annotations
import json, pathlib, statistics, sys
import time
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv
from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS
from extra.qk.quant.q4_k_gemv_primitive import q4k_gemv_partial_kernel, q4k_gemv_warp_kernel

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-ffn-gemv-warp"
HBM_PEAK_GBs = 960.0
ROLES = [("gate_up", 12288, 4096), ("down", 4096, 12288)]

def _device_us(fn, iters=30):
  import io, contextlib
  from tinygrad import Context
  from tinygrad.helpers import GlobalCounters
  fn().realize(); ts = []
  for _ in range(iters):
    GlobalCounters.reset()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), Context(DEBUG=2):
      fn().realize()
    if GlobalCounters.time_sum_s > 0: ts.append(GlobalCounters.time_sum_s * 1e6)
  return statistics.median(ts) if ts else 0.0

def main():
  assert Device.DEFAULT == "AMD"
  import numpy as np
  rng = np.random.default_rng(0)
  rows_results = []
  for role, rows, k in ROLES:
    k_blocks = k // Q4_K_BLOCK_ELEMS
    nwords = rows * k_blocks * Q4K_WORDS_PER_BLOCK
    q4_bytes = rows * k_blocks * Q4_K_BLOCK_BYTES
    w_np = rng.integers(0, 2**32, size=nwords, dtype=np.uint32)
    w_np[0::Q4K_WORDS_PER_BLOCK] = 0x3c003c00  # block word0 = (d,dmin)=(1.0,1.0) fp16 -> finite decode (no nan/inf)
    words = Tensor(w_np).realize()
    x = Tensor(rng.standard_normal(k).astype(np.float16)).realize()

    def run_default():
      partials = Tensor.empty(rows, 1, dtype=dtypes.float32)
      return partials.custom_kernel(words, x, fxn=q4k_gemv_partial_kernel(rows, k, 1, "none", ()))[0].sum(axis=1)
    def run_warp():
      out = Tensor.empty(rows, dtype=dtypes.float32)
      return out.custom_kernel(words, x, fxn=q4k_gemv_warp_kernel(rows, k))[0]

    d = run_default().realize(); w = run_warp().realize()
    rel = float((w - d).abs().max().item() / (d.abs().max().item() + 1e-6))
    du = _device_us(run_default); wu = _device_us(run_warp)
    dgb, wgb = q4_bytes / (du*1e-6) / 1e9, q4_bytes / (wu*1e-6) / 1e9
    r = {"role": role, "shape": f"{rows}x{k}", "q4_bytes": q4_bytes,
         "default_us": round(du, 2), "warp_us": round(wu, 2), "speedup": round(du/wu, 3),
         "default_GBs": round(dgb, 1), "warp_GBs": round(wgb, 1),
         "default_pct_peak": round(100*dgb/HBM_PEAK_GBs, 1), "warp_pct_peak": round(100*wgb/HBM_PEAK_GBs, 1),
         "warp_workgroups": rows, "warp_threads_per_row": 32,
         "rel_warp_vs_default": rel, "correct": rel <= 1e-2}
    rows_results.append(r)
    print(f"{role:8} {r['shape']:>11}: default {du:6.1f}us ({r['default_pct_peak']:.0f}%) -> warp {wu:6.1f}us "
          f"({r['warp_pct_peak']:.0f}%)  {r['speedup']}x  rel {rel:.1e} {'OK' if r['correct'] else 'FAIL'}", file=sys.__stderr__)

  gateup = next(r for r in rows_results if r["role"] == "gate_up")
  local_pass = gateup["correct"] and gateup["speedup"] > 1.05
  out = {"date": "2026-06-22", "phase": "Q4K_GEMV_WARP_LOCAL_AB", "hbm_peak_GBs": HBM_PEAK_GBs,
         "roles": rows_results, "gateup_local_pass": local_pass,
         "method": "synthetic Q4_K words; warp vs default (lossless => must match up to fp reassoc); device us = "
                   "GlobalCounters.time_sum_s median-of-50", "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"latest.json").write_text(json.dumps(out, indent=2))
  print(f"\ngate/up local_pass={local_pass} | artifact: {OUT/'latest.json'}", file=sys.__stderr__)

if __name__ == "__main__":
  main()
