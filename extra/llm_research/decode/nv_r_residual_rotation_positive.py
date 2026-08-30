#!/usr/bin/env python3
"""Latency-bound positive eviction control for the rotation probe.

A streaming read of a 16-32 MiB buffer was not L2-sensitive on the native HCQ
path: its duration stayed at DRAM bandwidth whether or not the buffer should
have been resident.  This tool uses a dependent pointer chase instead, which
is latency-bound and therefore separates an L2-resident working set from one
that has been evicted:

  hot       chase the same 32 MiB permutation every launch (resident in L2)
  rotating  chase disjoint 32 MiB permutations whose aggregate exceeds 2x L2,
            so each permutation is evicted before it is touched again

The permutation is a single cycle ``d[i] = (i + STRIDE) & mask`` with an odd
STRIDE, so successive hops land far apart and defeat prefetching.  If this
control does not show a clear rotating > hot delta, cache state stays
``UNMEASURED`` and the target rotation rows cannot be interpreted as a cache
measurement.

Measurement tooling only; no production code path is touched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import struct
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import _alloc, _make_queue  # noqa: E402

SCHEMA = "tinygrad.nv_r_residual_rotation_positive.v1"
L2_MIB = 96
TWO_X_L2_BYTES = 2 * L2_MIB * 1024 * 1024
CHASE_MIB = 32
CHASE_WORDS = CHASE_MIB * 1024 * 1024 // 4  # power of two
CHASE_MASK = CHASE_WORDS - 1
CHASE_STRIDE = 0x1F83D9AB  # odd, >> 2^23, so hops wrap far across the buffer
CHASE_BLOCK = 256
CHASE_GRID = (128, 1, 1)  # 32768 threads x 256 hops = full 32 MiB touched/launch
CHASE_ITERS = 256


def _copies_for(size: int) -> int:
  return max(1, (TWO_X_L2_BYTES + size - 1) // size + 1)


def _compile_chase(dev):
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  src = f"""
extern "C" __global__ void nv_r_rot_chase_{CHASE_MIB}mib(const unsigned int* __restrict__ d, unsigned int* __restrict__ out) {{
  unsigned int mask = {CHASE_MASK}u;
  unsigned int x = d[(blockIdx.x * {CHASE_BLOCK}u + threadIdx.x) & mask];
  for (unsigned int k = 0; k < {CHASE_ITERS}u; k++) x = d[x & mask];
  if (x == 0xDEADBEEFu) out[0] = x;
}}
"""
  return NVRTCCompiler(dev.arch, ptx=False, cache_key=f"nv_r_rot_chase_{CHASE_MIB}mib_v1").compile(src)


def _med(vals):
  return round(statistics.median(vals), 3)


def _run_arm(dev, prg, arg_states, grid, block, n, timeout_s):
  start_sigs = [dev.new_signal() for _ in range(n)]
  end_sigs = [dev.new_signal() for _ in range(n)]
  for args_state, st_sig, en_sig in zip(arg_states, start_sigs, end_sigs):
    q = _make_queue(dev)
    q.timestamp(st_sig)
    q.exec(prg, args_state, grid, block)
    q.timestamp(en_sig)
    target = dev.next_timeline()
    q.signal(dev.timeline_signal, target).submit(dev)
    dev.synchronize(timeout=int(timeout_s * 1000))
  return [round(float(en.timestamp - st.timestamp), 3)
          for st, en in zip(start_sigs, end_sigs)]


def _protocol(dev, prg, hot_states, rot_states, grid, block, warmup, n, timeout_s):
  out = {}
  for name, seq in (("H/C/H", ("hot", "rot", "hot")),
                    ("C/H/C", ("rot", "hot", "rot"))):
    arms = []
    for arm in seq:
      states = hot_states if arm == "hot" else rot_states
      durs = _run_arm(dev, prg, states, grid, block, n, timeout_s)
      kept = durs[warmup:]
      arms.append({"arm": arm, "durations_us": durs, "kept_us": kept,
                   "median_us": _med(kept), "mean_us": round(sum(kept) / len(kept), 3)})
    hot = arms[2] if name == "H/C/H" else arms[1]
    rot = arms[1] if name == "H/C/H" else arms[2]
    out[name] = {
      "arms": arms,
      "hot_median_us": hot["median_us"],
      "rotating_median_us": rot["median_us"],
      "eviction_delta_us": round(rot["median_us"] - hot["median_us"], 3),
    }
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=32)
  ap.add_argument("--warmup", type=int, default=6)
  ap.add_argument("--iters", type=int, default=CHASE_ITERS)
  ap.add_argument("--timeout-s", type=float, default=180.0)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  prg = NVProgram(dev, f"nv_r_rot_chase_{CHASE_MIB}mib", _compile_chase(dev))
  out_buf = _alloc(dev, 4)
  dev.allocator._copyin(out_buf, memoryview(bytearray(4)))

  pattern = ((np.arange(CHASE_WORDS, dtype=np.uint32) + CHASE_STRIDE) & CHASE_MASK)
  assert (pattern & ~np.uint32(CHASE_MASK)).sum() == 0
  pattern_mv = memoryview(pattern)

  n_copies = _copies_for(CHASE_WORDS * 4)
  copies = [_alloc(dev, CHASE_WORDS * 4) for _ in range(n_copies)]
  for c in copies:
    dev.allocator._copyin(c, pattern_mv)
  dev.synchronize()

  hot_states = [prg.fill_kernargs((copies[0], out_buf)) for _ in range(args.n)]
  rot_states = [prg.fill_kernargs((copies[k % n_copies], out_buf))
                for k in range(args.n)]
  prot = _protocol(dev, prg, hot_states, rot_states, CHASE_GRID,
                   (CHASE_BLOCK, 1, 1), args.warmup, args.n, args.timeout_s)

  payload = {
    "schema": SCHEMA,
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": ("dependent pointer-chase latency control; hot pins one 32 MiB "
               "permutation, rotating round-robins disjoint copies with aggregate "
               "> 2x 96 MiB L2; reverse-bracketed H/C/H and C/H/C"),
    "l2_mib": L2_MIB,
    "two_x_l2_bytes": TWO_X_L2_BYTES,
    "chase_mib": CHASE_MIB,
    "chase_words": CHASE_WORDS,
    "stride": CHASE_STRIDE,
    "grid": list(CHASE_GRID), "block": [CHASE_BLOCK, 1, 1],
    "iters": args.iters,
    "n_per_arm": args.n,
    "warmup_discard": args.warmup,
    "copies": n_copies,
    "aggregate_bytes": n_copies * CHASE_WORDS * 4,
    "protocols": prot,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
