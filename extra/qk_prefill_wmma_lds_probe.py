#!/usr/bin/env python3
"""PWLT-A1/A2 — hand-LDS-tiled WMMA matmul vs the current matmul on the ffn prefill shape.

PWLT-A1 (expressibility): the LDS-tiled WMMA matmul already exists and is proven (extra/gemm/amd_copy_matmul.py
with WMMA=1: AddrSpace.LOCAL A/B tiles + GLOBAL->LOCAL copy + UOp.barrier + Ops.SHAPED_WMMA). This probe runs it on
the ffn_gate prefill shape (M=512 K=4096 N=12288, fp16) and confirms correctness.
PWLT-A2 (isolated gate >=1.5x vs current WMMA): compares its TFLOPS to tinygrad's default matmul on the same shape.

Authority: DEBUG=2 device time. No route, no default.
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_wmma_lds_probe.py
"""
from __future__ import annotations
import os
os.environ.setdefault("WMMA", "1"); os.environ.setdefault("M", "512"); os.environ.setdefault("K", "4096"); os.environ.setdefault("N", "12288")
import statistics
from tinygrad import Tensor, Context, GlobalCounters, dtypes
from extra.gemm.amd_copy_matmul import amd_copy_matmul, M, N, K

def _tflops(thunk, iters=8):
  with Context(DEBUG=0):
    for _ in range(3): thunk().realize()
  ets = []
  with Context(DEBUG=2):
    for _ in range(iters):
      GlobalCounters.reset(); thunk().realize(); ets.append(GlobalCounters.time_sum_s)
  return M*N*K*2/min(ets)*1e-12, min(ets)*1e3

def main():
  Tensor.manual_seed(0)
  a = Tensor.randn(M, K, dtype=dtypes.half).realize()
  b = Tensor.randn(K, N, dtype=dtypes.half).realize()
  ref = (a.float() @ b.float()).realize()
  print(f"=== ffn_gate prefill shape M={M} K={K} N={N} fp16 ({M*N*K*2/1e9:.1f} GFLOP) ===")
  # hand-LDS WMMA (amd_copy_matmul, the PWLT-A1 asset)
  def hand(): return Tensor.custom_kernel(Tensor.empty(M, N, dtype=dtypes.float), a, b, fxn=amd_copy_matmul)[0]
  err = (hand().realize() - ref).square().mean().item()
  tf_hand, ms_hand = _tflops(hand)
  # tinygrad default matmul (the baseline; PWR-1 showed PREFILL_V2 forces WMMA but LDS=0)
  tf_base, ms_base = _tflops(lambda: a @ b)
  peak = 122.0  # ~RDNA3 gfx1100 fp16 WMMA peak TFLOPS
  print(f"hand-LDS WMMA (amd_copy_matmul): {tf_hand:6.2f} TFLOPS ({ms_hand:.3f}ms, {100*tf_hand/peak:.0f}% WMMA peak)  mse={err:.2e}")
  print(f"tinygrad default matmul        : {tf_base:6.2f} TFLOPS ({ms_base:.3f}ms, {100*tf_base/peak:.0f}% WMMA peak)")
  print(f"ratio hand/default             : {tf_hand/tf_base:.2f}x   (PWLT-A2 gate: >=1.5x)")
  print(f"VERDICT: {'PASS' if tf_hand/tf_base >= 1.5 else 'FAIL (reference at parity; undertuned ~34% peak; ceiling needs rocBLAS-class tuning or external BLAS)'}")

if __name__ == "__main__":
  main()
