#!/usr/bin/env python3
"""Reproduce HCQ profile per-kernel duration for a clean chained replay.

The retained production HCQ profile reports the Q/K norm command wall as
~2.5 us per kernel. This probe enqueues N exact production
``reduce_output_rmsnorm_8_128`` QMDs back-to-back with the same two distinct
timestamp signals per kernel that HCQ profiling uses, then reads each
``end.timestamp - start.timestamp`` in microseconds. A clean chain without
the production dependency graph shows whether the ~2.5 us is profiling
instrumentation, QMD dispatch, or something the production schedule adds.
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

CUBIN = ROOT / "docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/reduce_output_rmsnorm_8_128.cubin"
GRID = (8, 1, 1)
BLOCK = (2, 16, 1)
NS = [8, 16, 32, 64, 128]


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


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--warmup", type=int, default=1)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  prog = NVProgram(dev, "reduce_output_rmsnorm_8_128", CUBIN.read_bytes())
  out = _alloc(dev, 4096)
  x = _alloc(dev, 4096)
  w = _alloc(dev, 4096)
  for buf in (out, x, w):
    dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))
  dev.synchronize()
  bufs = (out, x, w)

  rows: list[dict] = []
  for n in NS:
    samples: list[dict] = []
    for rep in range(args.warmup + args.reps):
      q = _make_queue(dev)
      args_states = [prog.fill_kernargs(bufs) for _ in range(n)]
      start_sigs = [dev.new_signal() for _ in range(n)]
      end_sigs = [dev.new_signal() for _ in range(n)]
      for args_state, st_sig, en_sig in zip(args_states, start_sigs, end_sigs):
        q.timestamp(st_sig)
        q.exec(prog, args_state, GRID, BLOCK)
        q.timestamp(en_sig)
      target = dev.next_timeline()
      q.signal(dev.timeline_signal, target)
      t0 = time.perf_counter_ns()
      q.submit(dev)
      dev.synchronize()
      t1 = time.perf_counter_ns()
      durations = [float(en.timestamp - st.timestamp) for st, en in zip(start_sigs, end_sigs)]
      samples.append({
        "drain_us": (t1 - t0) / 1e3,
        "duration_us_median": statistics.median(durations),
        "duration_us_mean": sum(durations) / len(durations),
        "duration_us_min": min(durations),
        "duration_us_max": max(durations),
      })
    measured = samples[args.warmup:]
    rows.append({
      "n": n,
      "drain_us_median": statistics.median([s["drain_us"] for s in measured]),
      "duration_us_median_of_medians": statistics.median([s["duration_us_median"] for s in measured]),
      "duration_us_mean": sum(s["duration_us_mean"] for s in measured) / len(measured),
      "duration_us_min": min(s["duration_us_min"] for s in measured),
      "duration_us_max": max(s["duration_us_max"] for s in measured),
      "samples": measured,
    })

  payload = {
    "schema": "tinygrad.nv_hcq_profile_per_kernel.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      text=True).strip(),
    "method": "exact production reduce_output_rmsnorm_8_128 cubin, N chained QMDs, two distinct "
              "timestamp signals per kernel (HCQ profile replica), read end.timestamp - start.timestamp",
    "cubin_sha256": subprocess.check_output(["sha256sum", str(CUBIN)], text=True).split()[0],
    "grid": list(GRID),
    "block": list(BLOCK),
    "warmup": args.warmup,
    "reps": args.reps,
    "reference_production_profile_duration_us": 2.5,
    "rows": rows,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(rows, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
