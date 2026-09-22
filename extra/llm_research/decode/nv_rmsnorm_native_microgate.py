#!/usr/bin/env python3
"""Default-off, included-cost native-NV RMSNorm topology microgate.

This deliberately uses realized activation and affine-weight buffers.  It answers a
narrow question which the full-token Path-3 run could not: is the one-kernel native
RMSNorm itself viable when no opaque-boundary materialization is involved?  It does
not claim an e2e recovery; the token path has lazy producer views and is measured
separately.
"""
from __future__ import annotations

import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes, nn

DIM, EPS = 4096, 1e-6

def _kernel_names(out: Tensor) -> list[str]:
  linear, _ = out.linear_with_vars()
  return [x.src[0].arg.name for x in linear.src]

def run(replays: int=1000, reps: int=7) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  rng = np.random.default_rng(20260805)
  x_np = rng.normal(0, .2, (1, DIM)).astype(np.float16)
  w_np = rng.normal(1, .05, (DIM,)).astype(np.float16)
  x = Tensor(x_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()
  w = Tensor(w_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()
  ordinary, native = nn.RMSNorm(DIM, eps=EPS), nn.RMSNorm(DIM, eps=EPS)
  ordinary.weight, native.weight = w, w
  ordinary._rmsnorm_native_promoted, native._rmsnorm_native_promoted = False, True

  @TinyJit
  def baseline(a: Tensor): return ordinary(a)
  @TinyJit
  def candidate(a: Tensor): return native(a)

  # Establish both captured graphs before introspection/timing.
  baseline(x).realize(); base = baseline(x).realize()
  candidate(x).realize(); cand = candidate(x).realize()
  Device[dev].synchronize()
  base_np, cand_np = base.numpy().astype(np.float32), cand.numpy().astype(np.float32)

  def timed(fn):
    values = []
    for _ in range(reps):
      Device[dev].synchronize(); start = time.perf_counter_ns()
      for _ in range(replays): fn(x).realize()
      Device[dev].synchronize()
      values.append((time.perf_counter_ns()-start)/1e3/replays)
    return values

  # A/B/A avoids crediting a one-sided clock state to the candidate.
  a, b, c = timed(baseline), timed(candidate), timed(baseline)
  midpoint = (statistics.median(a)+statistics.median(c))/2
  max_abs = float(np.max(np.abs(base_np-cand_np)))
  return {
    "schema":"tinygrad.nv_rmsnorm_native_microgate.v1", "device":str(dev),
    "git_commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "contract":{"shape":[1,DIM], "x_dtype":"float16", "weight_dtype":"float16", "eps":EPS,
      "realized_inputs":True, "boundary_materialization_in_scope":False},
    "payload":{"x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest(), "weight_sha256":hashlib.sha256(w_np.tobytes()).hexdigest()},
    "topology":{"baseline":_kernel_names(ordinary(x)), "candidate":_kernel_names(native(x))},
    "correctness":{"max_abs":max_abs, "atol":3e-3, "pass":max_abs <= 3e-3},
    "timing":{"unit":"us_per_graph_replay", "replays":replays, "reps":reps,
      "control_a":a, "candidate_b":b, "control_c":c, "control_midpoint_median":midpoint,
      "candidate_median":statistics.median(b), "delta":statistics.median(b)-midpoint},
  }

def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--replays", type=int, default=1000); ap.add_argument("--reps", type=int, default=7); ap.add_argument("--out")
  args = ap.parse_args(); result = run(args.replays, args.reps); rendered = json.dumps(result, indent=2, sort_keys=True)
  if args.out:
    with open(args.out, "w") as f: f.write(rendered+"\n")
  print(rendered)
  return 0 if result["correctness"]["pass"] else 1

if __name__ == "__main__": raise SystemExit(main())
