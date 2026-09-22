#!/usr/bin/env python3
"""Exactness and isolated timing gate for the one-CTA finite-fp32 vocab argmax."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.packed_argmax import native_argmax_finite_fp32

N = 151936


def _payloads() -> dict[str, np.ndarray]:
  rng = np.random.default_rng(20260824)
  random = rng.normal(0, 7, (1, N)).astype(np.float32)
  ties = rng.normal(-10, 1, (1, N)).astype(np.float32); ties[0, [0, 17, N-1]] = np.float32(123.5)
  signed_zero = np.full((1, N), -1.0, dtype=np.float32); signed_zero[0, 0] = -0.0; signed_zero[0, 1] = 0.0
  all_equal = np.full((1, N), np.float32(-3.25), dtype=np.float32)
  extrema = rng.normal(0, 1, (1, N)).astype(np.float32)
  extrema[0, 7], extrema[0, N-3] = np.finfo(np.float32).max, np.finfo(np.float32).max
  return {"random":random, "first_index_ties":ties, "signed_zero":signed_zero,
          "all_equal":all_equal, "finite_extrema":extrema}


def run(replays:int, reps:int) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  @TinyJit
  def legacy(x): return x.argmax(-1, keepdim=True)
  def make_native(threads:int):
    @TinyJit
    def call(x): return native_argmax_finite_fp32(x, threads)
    return call
  calls = {threads:make_native(threads) for threads in (256, 512, 1024)}

  exact = {}
  for name, payload in _payloads().items():
    x = Tensor(payload, dtype=dtypes.float32, device=dev).contiguous().realize()
    want = int(legacy(x).realize().item()); rows = {}
    for threads, call in calls.items(): rows[str(threads)] = int(call(x).realize().item())
    np_want = int(np.argmax(payload, axis=-1)[0])
    rows["legacy"] = want; rows["numpy"] = np_want
    rows["pass"] = want == np_want and all(rows[str(t)] == want for t in calls)
    exact[name] = rows
  if not all(row["pass"] for row in exact.values()): raise RuntimeError(f"semantic gate failed: {exact}")

  payload = _payloads()["random"]
  x = Tensor(payload, dtype=dtypes.float32, device=dev).contiguous().realize()
  for _ in range(100):
    legacy(x).realize()
    for call in calls.values(): call(x).realize()
  Device[dev].synchronize()
  order = ("legacy_a", "256", "512", "1024", "legacy_c")
  samples = {}
  for label in order:
    call = legacy if label.startswith("legacy") else calls[int(label)]
    vals = []
    for _ in range(reps):
      Device[dev].synchronize(); begin = time.perf_counter_ns()
      for _ in range(replays): call(x).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-begin)/1e3/replays)
    samples[label] = vals
  control = statistics.median((statistics.median(samples["legacy_a"]), statistics.median(samples["legacy_c"])))
  medians = {label:statistics.median(vals) for label, vals in samples.items()}
  best_threads = min((256, 512, 1024), key=lambda threads:medians[str(threads)])
  result = {"schema":"tinygrad.nv_native_argmax_microgate.v1", "device":str(dev),
    "git_commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape":[1, N], "dtype":"float32", "finite_only":True, "exact":exact,
    "payload_sha256":hashlib.sha256(payload.tobytes()).hexdigest(),
    "timing":{"unit":"us_per_argmax_host_synchronized", "replays":replays, "reps":reps,
      "samples":samples, "medians":medians, "control_midpoint":control, "best_threads":best_threads,
      "best_recovery_us":control-medians[str(best_threads)],
      "best_speedup_pct":(control/medians[str(best_threads)]-1)*100.0}}
  return result


def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--replays", type=int, default=1000)
  ap.add_argument("--reps", type=int, default=7); ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args(); result = run(args.replays, args.reps)
  args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
