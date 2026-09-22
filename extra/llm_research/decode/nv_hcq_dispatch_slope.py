#!/usr/bin/env python3
"""Measure the unprofiled HCQ dispatch slope against the timestamp-bracketed
(profiled) arm using the exact production Q/K norm cubin.

The retained production capture reports the Q/K norm command wall as ~2.5 us
while the same cubin replayed through a CUDA driver context executes the pure
SASS body in ~1.18 us. That ~1.3 us residual is either the NV HCQ QMD
dispatch cost (present with and without profiling) or the two per-kernel
`timestamp` semaphore commands HCQ profiling inserts around each `exec`.

This probe submits N chained QMDs on a real NVComputeQueue and measures the
GPU drain slope for two arms:

  plain      N QMDs, no timestamps          (production unprofiled path)
  timestamp  N QMDs, 2 timestamps per QMD   (production profiled path)

Subtracting the measured pure-cubin body from the `plain` slope yields the
unprofiled per-launch dispatch tax. The `timestamp - plain` slope is the
profiling-instrumentation tax. It is measurement tooling only and changes no
production model, renderer, scheduler, or runtime file.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CUBIN = ROOT / "docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/reduce_output_rmsnorm_8_128.cubin"
GRID = (8, 1, 1)
BLOCK = (2, 16, 1)
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _median(values: list[float]) -> float | None:
  return statistics.median(values) if values else None


def _slope(xs: list[int], ys: list[float]) -> tuple[float, float]:
  # Ordinary least squares: y = slope * x + intercept.
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


def _compile_floor(dev) -> bytes:
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  src = """
extern "C" __global__ void nv_hcq_dispatch_floor(unsigned int* out) {
  if (threadIdx.x == 0u && blockIdx.x == 0u) out[0] = blockIdx.x + 1u;
}
"""
  return NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_hcq_dispatch_floor_v1").compile(src)


def _run_batch(dev, prg, bufs, n: int, arm: str, timeout_s: float):
  q = _make_queue(dev)
  args_states = [prg.fill_kernargs(bufs) for _ in range(n)]

  t0 = time.perf_counter_ns()
  if arm == "plain":
    for args_state in args_states:
      q.exec(prg, args_state, GRID, BLOCK)
  elif arm == "timestamp":
    st_sig = dev.new_signal()
    en_sig = dev.new_signal()
    for args_state in args_states:
      q.timestamp(st_sig)
      q.exec(prg, args_state, GRID, BLOCK)
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

  return {
    "enqueue_us": (t1 - t0) / 1e3,
    "submit_us": (t2 - t1) / 1e3,
    "drain_us": (t3 - t2) / 1e3,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--timeout-s", type=float, default=30.0)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  cubin_bytes = CUBIN.read_bytes()
  qk_prog = NVProgram(dev, "reduce_output_rmsnorm_8_128", cubin_bytes)

  out = _alloc(dev, 4096)
  x = _alloc(dev, 4096)
  w = _alloc(dev, 4096)
  for buf in (out, x, w):
    dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))
  dev.synchronize()
  qk_bufs = (out, x, w)

  floor_prog = NVProgram(dev, "nv_hcq_dispatch_floor", _compile_floor(dev))
  floor_out = _alloc(dev, 4)
  dev.allocator._copyin(floor_out, memoryview(b"\x00" * 4))
  dev.synchronize()

  rows: list[dict] = []
  for prog_name, prog, bufs, grid, block in (
      ("qk_cubin", qk_prog, qk_bufs, GRID, BLOCK),
      ("floor_noop", floor_prog, (floor_out,), (1, 1, 1), (32, 1, 1))):
    for arm in ("plain", "timestamp"):
      for n in NS:
        samples: list[dict] = []
        for rep in range(args.warmup + args.reps):
          # The floor kernel has a different grid/block; qk uses the module
          # constants. This branch keeps the loop body explicit.
          row = _run_batch(dev, prog, bufs, n, arm, args.timeout_s) if prog_name == "qk_cubin" else \
                _run_batch_geometry(dev, prog, bufs, n, arm, grid, block, args.timeout_s)
          samples.append(row)
        measured = samples[args.warmup:]
        rows.append({
          "program": prog_name,
          "arm": arm,
          "n": n,
          "enqueue_us_median": _median([r["enqueue_us"] for r in measured]),
          "submit_us_median": _median([r["submit_us"] for r in measured]),
          "drain_us_median": _median([r["drain_us"] for r in measured]),
          "samples": measured,
        })

  slopes: dict = {}
  for prog_name in ("qk_cubin", "floor_noop"):
    for arm in ("plain", "timestamp"):
      subset = [r for r in rows if r["program"] == prog_name and r["arm"] == arm]
      xs = [r["n"] for r in subset]
      ys = [r["drain_us_median"] for r in subset]
      slope, intercept = _slope(xs, ys)
      slopes[f"{prog_name}_{arm}_drain_us_per_kernel"] = slope
      slopes[f"{prog_name}_{arm}_drain_intercept_us"] = intercept
      host_slope, _ = _slope(xs, [r["enqueue_us_median"] for r in subset])
      slopes[f"{prog_name}_{arm}_host_enqueue_us_per_kernel"] = host_slope

  payload = {
    "schema": "tinygrad.nv_hcq_dispatch_slope.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      text=True).strip(),
    "method": "NVComputeQueue chained QMD replay of the exact production reduce_output_rmsnorm_8_128 cubin; "
              "plain arm = production unprofiled path, timestamp arm = two HCQ profile timestamp semaphores per kernel",
    "cubin_sha256": subprocess.check_output(["sha256sum", str(CUBIN)], text=True).split()[0],
    "grid": list(GRID),
    "block": list(BLOCK),
    "ns": NS,
    "warmup": args.warmup,
    "reps": args.reps,
    "reference_pure_cubin_body_us": {"reduce_output_rmsnorm_8_128": 1.1962},
    "rows": rows,
    "slopes": slopes,
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"rows": rows, "slopes": slopes}, indent=2))
  return 0


def _run_batch_geometry(dev, prg, bufs, n: int, arm: str, grid, block, timeout_s: float):
  # Floor kernel uses a caller-provided geometry; otherwise identical to
  # _run_batch. Kept separate so the qk path never accidentally changes grid.
  q = _make_queue(dev)
  args_states = [prg.fill_kernargs(bufs) for _ in range(n)]
  t0 = time.perf_counter_ns()
  if arm == "plain":
    for args_state in args_states:
      q.exec(prg, args_state, grid, block)
  else:
    st_sig = dev.new_signal()
    en_sig = dev.new_signal()
    for args_state in args_states:
      q.timestamp(st_sig)
      q.exec(prg, args_state, grid, block)
      q.timestamp(en_sig)
  target = dev.next_timeline()
  q.signal(dev.timeline_signal, target)
  t1 = time.perf_counter_ns()
  q.submit(dev)
  t2 = time.perf_counter_ns()
  dev.synchronize(timeout=int(timeout_s * 1000))
  t3 = time.perf_counter_ns()
  return {"enqueue_us": (t1 - t0) / 1e3, "submit_us": (t2 - t1) / 1e3,
          "drain_us": (t3 - t2) / 1e3}


if __name__ == "__main__":
  raise SystemExit(main())
