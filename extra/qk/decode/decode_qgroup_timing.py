#!/usr/bin/env python3
"""Compare shipped G=5 decode attention with llama-style 2+2+1 query-head grouping."""
from __future__ import annotations

import argparse, json, statistics, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from extra.qk.decode.flash_decode_attention_executor import flash_decode_live_split_block_tile

HKV, G, HD, HQ, MAXC, SPLITS = 8, 5, 128, 40, 4608, 48


def benchmark(tc: int, grouped_splits: int, query_group_size: int, warmups: int, samples: int) -> dict:
  rng = np.random.default_rng(20260727 + tc)
  q_np = rng.normal(0, 0.04, (HKV, G, HD)).astype(np.float16)
  cache_np = np.zeros((2, 1, HKV, MAXC, HD), dtype=np.float16)
  cache_np[:, :, :, :tc, :] = rng.normal(0, 0.04, (2, 1, HKV, tc, HD)).astype(np.float16)
  q = Tensor(q_np.reshape(HQ, HD), device="AMD")
  cache = Tensor(cache_np, device="AMD")

  baseline = TinyJit(lambda: flash_decode_live_split_block_tile(
    q, cache, tc, HD, HQ, HKV, MAXC, SPLITS, staging="KV_BOTH", fused_combine=True))
  grouped = TinyJit(lambda: flash_decode_live_split_block_tile(
    q, cache, tc, HD, HQ, HKV, MAXC, grouped_splits, staging="KV_BOTH", fused_combine=True,
    query_group_size=query_group_size))

  ref = baseline().numpy().astype(np.float32)
  got = grouped().numpy().astype(np.float32)

  def measure(fn) -> list[float]:
    for _ in range(warmups): fn().realize()
    Device["AMD"].synchronize()
    raw = []
    for _ in range(samples):
      Device["AMD"].synchronize(); begin = time.perf_counter_ns()
      fn().realize(); Device["AMD"].synchronize()
      raw.append((time.perf_counter_ns() - begin) / 1e6)
    return raw

  baseline_ms = measure(baseline)
  grouped_ms = measure(grouped)
  abs_err = np.abs(ref - got)
  return {
    "Tc": tc, "baseline_splits": SPLITS, "grouped_splits": grouped_splits,
    "baseline_warps_per_workgroup": 5, "grouped_warps_per_workgroup": query_group_size,
    "warmups": warmups, "samples": samples,
    "baseline_raw_ms": baseline_ms, "baseline_median_ms": statistics.median(baseline_ms),
    "grouped_raw_ms": grouped_ms, "grouped_median_ms": statistics.median(grouped_ms),
    "grouped_vs_baseline_percent": 100 * (statistics.median(grouped_ms) / statistics.median(baseline_ms) - 1),
    "max_abs_error": float(abs_err.max()), "mean_abs_error": float(abs_err.mean()),
    "outputs_finite": bool(np.isfinite(ref).all() and np.isfinite(got).all()),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--Tc", type=int, choices=(128, 512, 1024, 4096), required=True)
  ap.add_argument("--grouped-splits", type=int, required=True)
  ap.add_argument("--query-group-size", type=int, choices=(1, 2, 3, 4), default=2)
  ap.add_argument("--warmups", type=int, default=3)
  ap.add_argument("--samples", type=int, default=3)
  args = ap.parse_args()
  print(json.dumps(benchmark(args.Tc, args.grouped_splits, args.query_group_size, args.warmups, args.samples), sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
