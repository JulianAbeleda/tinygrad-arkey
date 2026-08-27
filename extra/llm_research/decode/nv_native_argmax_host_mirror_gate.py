#!/usr/bin/env python3
"""Exactness and synchronized latency gate for a pinned-host argmax mirror."""
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _host_tensor(dev):
  from tinygrad import Tensor, dtypes
  from tinygrad.device import BufferSpec
  from tinygrad.uop.ops import UOp
  u = UOp.new_buffer(dev.device, 1, dtypes.int32)
  u.buffer.options = BufferSpec(host=True, nolru=True)
  u.buffer.ensure_allocated()
  return Tensor(u)


def _host_value(t) -> int:
  return int(t.uop.buffer.get_buf(t.device).cpu_view().view(size=4, fmt="i")[0])


def run(replays: int, reps: int) -> dict:
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.llm.packed_argmax import native_argmax_finite_fp32, native_argmax_finite_fp32_host_mirror

  dev, n = Device["NV"], 151936
  rng = np.random.default_rng(20260827)
  payloads = []
  for i in range(17):
    x = rng.normal(0, 5, (1, n)).astype(np.float32)
    winner = (i * 7919 + 17) % n
    x[0, winner] = np.float32(1000 + i)
    payloads.append((x, winner))

  mirror = _host_tensor(dev)

  @TinyJit
  def control(x): return native_argmax_finite_fp32(x, 1024)

  @TinyJit
  def candidate(x, host): return native_argmax_finite_fp32_host_mirror(x, host, 1024)

  exact = []
  tensors = [Tensor(x, dtype=dtypes.float32, device="NV").contiguous().realize() for x, _ in payloads]
  for x, (_, want) in zip(tensors, payloads):
    got_control = int(control(x).realize().item())
    got_gpu, got_mirror = candidate(x, mirror)
    # Realize only the GPU result.  The mirror store must survive as part of
    # the same opaque call; production does not retain a second graph output.
    got_gpu.realize(); dev.synchronize()
    host_value = _host_value(mirror)
    gpu_value = int(got_gpu.item())
    exact.append({"want": want, "control": got_control, "gpu": gpu_value, "mirror": host_value,
                  "pass": want == got_control == gpu_value == host_value})
  if not all(row["pass"] for row in exact): raise RuntimeError(f"exactness failed: {exact}")

  x = tensors[0]
  for _ in range(100):
    control(x).realize().item()
    gpu, host = candidate(x, mirror); Tensor.realize(gpu, host); dev.synchronize(); _host_value(mirror)

  samples = {"control_a": [], "candidate": [], "control_c": []}
  for label in samples:
    for _ in range(reps):
      begin = time.perf_counter_ns()
      if label == "candidate":
        for _ in range(replays):
          gpu, host = candidate(x, mirror); Tensor.realize(gpu, host); dev.synchronize(); value = _host_value(mirror)
      else:
        for _ in range(replays): value = int(control(x).realize().item())
      elapsed = (time.perf_counter_ns() - begin) / 1e3 / replays
      if value != payloads[0][1]: raise RuntimeError(f"timed value mismatch: {value}")
      samples[label].append(elapsed)
  medians = {key: statistics.median(vals) for key, vals in samples.items()}
  midpoint = statistics.median((medians["control_a"], medians["control_c"]))
  return {
    "schema": "tinygrad.nv_native_argmax_host_mirror_gate.v1",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape": [1, n], "dtype": "float32", "replays": replays, "reps": reps,
    "exact": exact, "samples_us": samples, "medians_us": medians,
    "control_midpoint_us": midpoint, "candidate_us": medians["candidate"],
    "recovery_us": midpoint - medians["candidate"],
    "speedup_pct": (midpoint / medians["candidate"] - 1) * 100,
    "pass": all(row["pass"] for row in exact) and medians["candidate"] < midpoint,
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--replays", type=int, default=50)
  ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args(); result = run(args.replays, args.reps)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
