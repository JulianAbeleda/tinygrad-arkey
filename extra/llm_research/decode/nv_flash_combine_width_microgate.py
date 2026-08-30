#!/usr/bin/env python3
"""Native-NV exact/timing gate for wider flash-combine blocks."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.flash_decode_attention import FlashCombineSpec
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program

HQ, HD, SPLITS, W = 32, 128, 48, 130


def _program(width: int) -> KernelProgram:
  spec = FlashCombineSpec(HD, HQ, SPLITS, output_fp16=True, lane_width=width)
  return KernelProgram("research.flash_combine_width", f"width{width}", KernelProgramProvenance.RESEARCH_ONLY,
                       spec.emit(), output_spec=OutputSpec((HQ * HD,), dtypes.float16))


def run(replays: int, reps: int) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  rng = np.random.default_rng(20260824)
  payload = np.empty((HQ, SPLITS, W), dtype=np.float32)
  payload[..., :HD] = rng.normal(0, 0.3, (HQ, SPLITS, HD)).astype(np.float32)
  payload[..., HD] = rng.uniform(0.1, 8.0, (HQ, SPLITS)).astype(np.float32)
  payload[..., HD + 1] = rng.normal(0, 2.0, (HQ, SPLITS)).astype(np.float32)
  src = Tensor(payload.reshape(-1), dtype=dtypes.float32, device=dev).contiguous().realize()
  widths = (32, 64, 128)
  programs = {width: _program(width) for width in widths}

  def make_call(width: int):
    program = programs[width]
    @TinyJit
    def call(x):
      return execute_research_program(Tensor.empty((HQ * HD,), dtype=dtypes.float16, device=dev),
                                      x, program=program)
    return call

  calls = {width: make_call(width) for width in widths}
  outputs = {}
  for width in widths:
    calls[width](src).realize(); outputs[width] = calls[width](src).realize().numpy()
  Device[dev].synchronize()
  control = outputs[32].view(np.uint16)
  exact = {str(width): {
    "bitwise_equal": bool(np.array_equal(control, outputs[width].view(np.uint16))),
    "mismatched_fp16_words": int(np.count_nonzero(control != outputs[width].view(np.uint16))),
    "max_abs_delta": float(np.max(np.abs(outputs[32].astype(np.float32) - outputs[width].astype(np.float32)))),
    "sha256": hashlib.sha256(outputs[width].tobytes()).hexdigest(),
  } for width in widths}
  if not all(x["bitwise_equal"] for x in exact.values()): raise RuntimeError(f"exactness gate failed: {exact}")

  for _ in range(500):
    for width in widths: calls[width](src).realize()
  Device[dev].synchronize()
  samples = {}
  for width in (32, 64, 128, 32):
    vals = []
    for _ in range(reps):
      Device[dev].synchronize(); begin = time.perf_counter_ns()
      for _ in range(replays): calls[width](src).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns() - begin) / 1e3 / replays)
    samples.setdefault(str(width), []).append(vals)
  control_a, control_c = samples["32"]
  control_midpoint = (statistics.median(control_a) + statistics.median(control_c)) / 2
  medians = {"32_control_midpoint": control_midpoint,
             "64": statistics.median(samples["64"][0]), "128": statistics.median(samples["128"][0])}
  best_width = min((64, 128), key=lambda width: medians[str(width)])
  result = {
    "schema": "tinygrad.nv_flash_combine_width_microgate.v1",
    "device": str(dev), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape": {"heads": HQ, "head_dim": HD, "splits": SPLITS, "output_dtype": "float16"},
    "payload_sha256": hashlib.sha256(payload.tobytes()).hexdigest(), "exact": exact,
    "timing": {"unit": "us_per_launch_host_synchronized", "replays": replays, "reps": reps,
               "samples": samples, "medians": medians, "best_width": best_width,
               "best_recovery_us": control_midpoint - medians[str(best_width)]},
  }
  return result


def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--replays", type=int, default=1000)
  ap.add_argument("--reps", type=int, default=7); ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args(); result = run(args.replays, args.reps)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
