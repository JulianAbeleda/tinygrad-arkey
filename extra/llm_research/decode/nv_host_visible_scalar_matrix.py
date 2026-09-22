#!/usr/bin/env python3
"""Qualify GPU-written, CPU-visible scalar allocation classes on NV.

This is a measurement-only gate for direct token delivery.  A compute kernel
writes a value plus guard words, signals completion, and the CPU reads the
mapping without an intervening DMA copy or device-wide synchronize.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SRC = r'''
extern "C" __global__ void host_visible_scalar(unsigned int *out, unsigned int value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = 0x13579bdfu;
    out[1] = value;
    out[2] = value ^ 0xa5a5a5a5u;
    out[3] = 0xfdb97531u;
  }
}
'''


def _queue(dev, program, buf, value):
  from tinygrad.runtime.ops_nv import NVComputeQueue
  q = NVComputeQueue(queue_idx=0)
  q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
          shared_mem_window=dev.shared_mem_window)
  q.wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  q.exec(program, program.fill_kernargs((buf,), (value,)), (1, 1, 1), (32, 1, 1))
  done = dev.new_signal()
  q.signal(done, 1).bind(dev)
  return q, done


def _words(buf):
  view = buf.cpu_view().view(size=16, fmt="I")
  return [int(view[i]) for i in range(4)]


def _expected(value): return [0x13579bdf, value, value ^ 0xa5a5a5a5, 0xfdb97531]


def run(stress: int, reps: int) -> dict:
  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

  dev = Device["NV"]
  cubin = NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_host_visible_scalar_matrix_v1").compile(SRC)
  program = NVProgram(dev, "host_visible_scalar", cubin)
  specs = {
    "mapped_vram_cached": {"cpu_access": True},
    "pinned_host_cached": {"host": True},
    "system_uncached": {"uncached": True, "cpu_access": True},
  }
  values = [0x00000000, 0xffffffff, 0x01234567, 0x89abcdef,
            0x55555555, 0xaaaaaaaa, 0xdeadbeef, 0x31415926]
  rows = {}
  for name, spec in specs.items():
    buf = dev.iface.alloc(0x1000, **spec)
    queues = [_queue(dev, program, buf, value) for value in values]
    errors, samples = [], []
    for i in range(stress + reps):
      value = values[(i * 5 + i // len(values)) % len(values)]
      q, done = queues[values.index(value)]
      done.value = 0
      begin = time.perf_counter_ns()
      q.submit(dev)
      done.wait(1)
      got = _words(buf)
      elapsed = (time.perf_counter_ns() - begin) / 1e3
      if got != _expected(value): errors.append({"iteration": i, "value": value, "got": got, "expected": _expected(value)})
      if i >= stress: samples.append(elapsed)
    rows[name] = {
      "allocation": spec, "stress_iterations": stress, "errors": errors[:16],
      "error_count": len(errors), "pass": not errors, "samples_us": samples,
      "median_us": statistics.median(samples), "minimum_us": min(samples),
    }
    dev.iface.free(buf)
  return {
    "schema": "tinygrad.nv_host_visible_scalar_matrix.v1",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "device": str(dev), "completion_boundary": "compute signal wait; no DMA copy or device synchronize",
    "stress": stress, "reps": reps, "rows": rows, "pass": all(row["pass"] for row in rows.values()),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--stress", type=int, default=10000)
  ap.add_argument("--reps", type=int, default=101)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  result = run(args.stress, args.reps)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
