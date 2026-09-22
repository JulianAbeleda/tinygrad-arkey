#!/usr/bin/env python3
"""Generalized unprofiled HCQ dispatch slope for an arbitrary captured cubin.

This is a parameterized version of nv_hcq_dispatch_slope.py.  It submits N
chained QMDs for a caller-supplied cubin/grid/block/buffer-set and measures the
GPU drain slope for two arms:

  plain      N QMDs, no timestamps          (production unprofiled path)
  timestamp  N QMDs, 2 timestamps per QMD   (production profiled path)

The `plain` slope is the clean chained HCQ duration `C`; subtracting the exact
cubin body `B` yields the clean dispatch component `D = C - B`.  The
`timestamp - plain` slope is the profiling instrumentation tax.  It is
measurement tooling only and changes no production model/runtime path.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _median(values: list[float]) -> float | None:
  return statistics.median(values) if values else None


def _slope(xs: list[int], ys: list[float]) -> tuple[float, float]:
  n = len(xs)
  mx = sum(xs) / n
  my = sum(ys) / n
  den = sum((x - mx) ** 2 for x in xs)
  slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
  intercept = my - slope * mx
  return slope, intercept


def _alloc(dev, size: int):
  from tinygrad.device import BufferSpec
  return dev.allocator._alloc(size, BufferSpec())


def _make_queue(dev):
  from tinygrad.runtime.ops_nv import NVComputeQueue
  q = NVComputeQueue(queue_idx=0)
  q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
          shared_mem_window=dev.shared_mem_window)
  q.wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  return q


def _run_batch(dev, prg, bufs, vals, n: int, arm: str, grid, block, timeout_s: float) -> dict:
  q = _make_queue(dev)
  args_states = [prg.fill_kernargs(bufs, vals=vals) for _ in range(n)]

  t0 = time.perf_counter_ns()
  if arm == "plain":
    for args_state in args_states:
      q.exec(prg, args_state, grid, block)
  elif arm == "timestamp":
    st_sig = dev.new_signal()
    en_sig = dev.new_signal()
    for args_state in args_states:
      q.timestamp(st_sig)
      q.exec(prg, args_state, grid, block)
      q.timestamp(en_sig)
  else:
    raise ValueError(arm)
  target = dev.next_timeline()
  q.signal(dev.timeline_signal, target)
  t1 = time.perf_counter_ns()
  q.submit(dev)
  t2 = time.perf_counter_ns()
  dev.synchronize(timeout=int(timeout_s * 1000))
  t3 = time.perf_counter_ns()
  return {"enqueue_us": (t1 - t0) / 1e3, "submit_us": (t2 - t1) / 1e3, "drain_us": (t3 - t2) / 1e3}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--cubin", type=pathlib.Path, required=True)
  ap.add_argument("--symbol", required=True)
  ap.add_argument("--grid", required=True, help="comma-separated x,y,z")
  ap.add_argument("--block", required=True, help="comma-separated x,y,z")
  ap.add_argument("--buf-sizes", required=True, help="comma-separated bytes, one per buffer")
  ap.add_argument("--vals", default="", help="comma-separated int32 scalar arguments after buffers")
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--timeout-s", type=float, default=60.0)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  cubin_bytes = args.cubin.read_bytes()
  prg = NVProgram(dev, args.symbol, cubin_bytes)

  sizes = [int(x) for x in args.buf_sizes.split(",")]
  vals = tuple(int(x) for x in args.vals.split(",") if x)
  bufs = tuple(_alloc(dev, sz) for sz in sizes)
  for buf in bufs:
    dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))
  dev.synchronize()

  grid = tuple(int(x) for x in args.grid.split(","))
  block = tuple(int(x) for x in args.block.split(","))
  assert len(grid) == 3 and len(block) == 3

  rows: list[dict] = []
  for arm in ("plain", "timestamp"):
    for n in NS:
      samples: list[dict] = []
      for rep in range(args.warmup + args.reps):
        samples.append(_run_batch(dev, prg, bufs, vals, n, arm, grid, block, args.timeout_s))
      measured = samples[args.warmup:]
      rows.append({
        "arm": arm, "n": n,
        "enqueue_us_median": _median([r["enqueue_us"] for r in measured]),
        "submit_us_median": _median([r["submit_us"] for r in measured]),
        "drain_us_median": _median([r["drain_us"] for r in measured]),
        "samples": measured,
      })

  slopes: dict = {}
  for arm in ("plain", "timestamp"):
    subset = [r for r in rows if r["arm"] == arm]
    xs = [r["n"] for r in subset]
    ys = [r["drain_us_median"] for r in subset]
    slope, intercept = _slope(xs, ys)
    slopes[f"{arm}_drain_us_per_kernel"] = slope
    slopes[f"{arm}_drain_intercept_us"] = intercept
    host_slope, _ = _slope(xs, [r["enqueue_us_median"] for r in subset])
    slopes[f"{arm}_host_enqueue_us_per_kernel"] = host_slope

  payload = {
    "schema": "tinygrad.nv_hcq_dispatch_slope_general.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": "NVComputeQueue chained QMD replay of a caller-supplied production cubin; plain = unprofiled, timestamp = 2 HCQ timestamps/kernel",
    "cubin": str(args.cubin),
    "cubin_sha256": subprocess.check_output(["sha256sum", str(args.cubin)], text=True).split()[0],
    "symbol": args.symbol,
    "grid": list(grid), "block": list(block), "buf_sizes": sizes, "vals": list(vals),
    "ns": NS, "warmup": args.warmup, "reps": args.reps,
    "rows": rows, "slopes": slopes,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"slopes": slopes}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
