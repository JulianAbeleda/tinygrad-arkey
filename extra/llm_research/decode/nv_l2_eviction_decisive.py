#!/usr/bin/env python3
"""Decisive L2-eviction probe for the native NV HCQ path.

The rotation probes could not separate an L2-resident working set from an
evicted one: a low-MLP pointer chase was TLB-dominated and a high-MLP chase
saturated the memory system.  This tool isolates the one question that gates
the whole cache-state adjudication:

    can a native-HCQ kernel observe L2 residency at all?

Protocol, all through the same stable timestamp path as the first probe:

    1. SM-touch buffer A (32 MiB) and evict buffer E (256 MiB) so every page is
       mapped and the MMU TLB is warm.
    2. chase A  ->  latency L1 (A is L2-resident after the touch)
    3. stream-read E (256 MiB > 96 MiB L2)  ->  evicts A
    4. chase A  ->  latency L2 (A should now be DRAM-resident)

If L2 is measurably larger than L1, L2 residency is observable and the
rotation control can be rebuilt on it.  If L1 ~= L2, cache state is
``UNMEASURED`` with this harness.

Measurement tooling only; no production code path is touched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import _alloc, _make_queue  # noqa: E402
from nv_r_residual_rotation_positive import (  # noqa: E402
  _compile_chase, CHASE_WORDS, CHASE_MASK, CHASE_STRIDE,
)

SCHEMA = "tinygrad.nv_l2_eviction_decisive.v1"
EVICT_MIB = 256
EVICT_BYTES = EVICT_MIB * 1024 * 1024
EVICT_WORDS = EVICT_BYTES // 4
EVICT_BLOCK = 256
EVICT_GRID = ((EVICT_WORDS + EVICT_BLOCK - 1) // EVICT_BLOCK, 1, 1)
CHASE_GRID = (1, 1, 1)
CHASE_BLOCK = 256


def _compile_stream_read(dev, words: int, tag: str):
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  src = f"""
extern "C" __global__ void nv_l2_stream_{tag}(const float* __restrict__ d, float* __restrict__ out) {{
  unsigned int i = blockIdx.x * {EVICT_BLOCK}u + threadIdx.x;
  if (i < {words}u) {{
    float v = d[i];
    if (v < 0.0f) out[0] = v;
  }}
}}
"""
  return NVRTCCompiler(dev.arch, ptx=False, cache_key=f"nv_l2_stream_{tag}_v1").compile(src)


def _run_single(dev, prg, args_state, grid, block, timeout_s):
  q = _make_queue(dev)
  st = dev.new_signal()
  en = dev.new_signal()
  q.timestamp(st)
  q.exec(prg, args_state, grid, block)
  q.timestamp(en)
  target = dev.next_timeline()
  q.signal(dev.timeline_signal, target)
  q.submit(dev)
  dev.synchronize(timeout=int(timeout_s * 1000))
  return float(en.timestamp - st.timestamp)


def _run_many(dev, prg, args_states, grid, block, n, timeout_s):
  """Stable per-launch timestamp bracket, median over n reps."""
  starts = [dev.new_signal() for _ in range(n)]
  ends = [dev.new_signal() for _ in range(n)]
  for a, s, e in zip(args_states, starts, ends):
    q = _make_queue(dev)
    q.timestamp(s)
    q.exec(prg, a, grid, block)
    q.timestamp(e)
    target = dev.next_timeline()
    q.signal(dev.timeline_signal, target).submit(dev)
    dev.synchronize(timeout=int(timeout_s * 1000))
  return [float(e.timestamp - s.timestamp) for s, e in zip(starts, ends)]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--reps", type=int, default=24)
  ap.add_argument("--timeout-s", type=float, default=180.0)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  chase = NVProgram(dev, "nv_l2_chase_32mib", _compile_chase(dev))
  stream = NVProgram(dev, f"nv_l2_stream_evict{EVICT_MIB}mib", _compile_stream_read(dev, EVICT_WORDS, f"evict{EVICT_MIB}mib"))
  touch_a = NVProgram(dev, "nv_l2_stream_touch32mib", _compile_stream_read(dev, CHASE_WORDS, "touch32mib"))
  out = _alloc(dev, 4)
  dev.allocator._copyin(out, memoryview(bytearray(4)))

  # 32 MiB permutation for the chase, 256 MiB streaming buffer for eviction.
  a = _alloc(dev, CHASE_WORDS * 4)
  pat = ((np.arange(CHASE_WORDS, dtype=np.uint32) + CHASE_STRIDE) & CHASE_MASK)
  dev.allocator._copyin(a, memoryview(pat))
  evict = _alloc(dev, EVICT_BYTES)
  dev.allocator._copyin(evict, memoryview(bytearray(EVICT_BYTES)))
  dev.synchronize()

  # SM-touch both so every page is mapped (TLB warm) before latency timing.
  touch_grid = ((CHASE_WORDS + EVICT_BLOCK - 1) // EVICT_BLOCK, 1, 1)
  _run_single(dev, touch_a, touch_a.fill_kernargs((a, out)), touch_grid, (EVICT_BLOCK, 1, 1), args.timeout_s)
  _run_single(dev, stream, stream.fill_kernargs((evict, out)), EVICT_GRID, (EVICT_BLOCK, 1, 1), args.timeout_s)

  # L1: chase A while it is L2-resident.
  l1 = _run_many(dev, chase, [chase.fill_kernargs((a, out)) for _ in range(args.reps)],
                 CHASE_GRID, (CHASE_BLOCK, 1, 1), args.reps, args.timeout_s)
  # Evict: stream-read 256 MiB (three times the 96 MiB L2).
  _run_many(dev, stream, [stream.fill_kernargs((evict, out)) for _ in range(3)],
            EVICT_GRID, (EVICT_BLOCK, 1, 1), 3, args.timeout_s)
  # L2: chase A after eviction.
  l2 = _run_many(dev, chase, [chase.fill_kernargs((a, out)) for _ in range(args.reps)],
                 CHASE_GRID, (CHASE_BLOCK, 1, 1), args.reps, args.timeout_s)

  payload = {
    "schema": SCHEMA,
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": ("SM-touch TLB warmup, then chase A (L2-resident), stream-read 256 MiB "
               "to evict, chase A again; per-launch timestamp brackets, median"),
    "chase_words": CHASE_WORDS,
    "chase_grid": list(CHASE_GRID), "chase_block": [CHASE_BLOCK, 1, 1],
    "evict_mib": EVICT_MIB,
    "reps": args.reps,
    "l1_resident_us": [round(x, 3) for x in l1],
    "l1_median_us": round(statistics.median(l1), 3),
    "l2_evicted_us": [round(x, 3) for x in l2],
    "l2_median_us": round(statistics.median(l2), 3),
    "eviction_delta_us": round(statistics.median(l2) - statistics.median(l1), 3),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: v for k, v in payload.items()
                    if not isinstance(v, list)}, indent=2), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
