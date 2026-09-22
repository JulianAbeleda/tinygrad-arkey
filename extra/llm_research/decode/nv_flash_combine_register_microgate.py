#!/usr/bin/env python3
"""Native-NV exact/timing gate for register-broadcast Flash combine weights."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program

HQ, HD, W = 32, 128, 130


def _program(register_weights: bool, splits:int) -> KernelProgram:
  emitter = flash_fused_gmax_combine_kernel(HD, HQ, splits, output_fp16=True, lane_width=128,
                                            register_weights=register_weights)
  arm = "register" if register_weights else "shared"
  return KernelProgram("research.flash_combine_register", arm, KernelProgramProvenance.RESEARCH_ONLY,
                       emitter, output_spec=OutputSpec((HQ * HD,), dtypes.float16))


def run(replays: int, reps: int, splits:int) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  rng = np.random.default_rng(20260827)
  payload = np.empty((HQ, splits, W), dtype=np.float32)
  payload[..., :HD] = rng.normal(0, 0.3, (HQ, splits, HD)).astype(np.float32)
  payload[..., HD] = rng.uniform(0.1, 8.0, (HQ, splits)).astype(np.float32)
  payload[..., HD + 1] = rng.normal(0, 2.0, (HQ, splits)).astype(np.float32)
  src = Tensor(payload.reshape(-1), dtype=dtypes.float32, device=dev).contiguous().realize()
  programs = {arm: _program(arm == "register", splits) for arm in ("shared", "register")}

  def make_call(arm: str):
    program = programs[arm]
    @TinyJit
    def call(x):
      return execute_research_program(Tensor.empty((HQ * HD,), dtype=dtypes.float16, device=dev), x, program=program)
    return call

  calls = {arm: make_call(arm) for arm in programs}
  outputs = {}
  for arm in calls:
    calls[arm](src).realize(); outputs[arm] = calls[arm](src).realize().numpy()
  Device[dev].synchronize()
  shared, register = outputs["shared"].view(np.uint16), outputs["register"].view(np.uint16)
  exact = {
    "bitwise_equal": bool(np.array_equal(shared, register)),
    "mismatched_fp16_words": int(np.count_nonzero(shared != register)),
    "max_abs_delta": float(np.max(np.abs(outputs["shared"].astype(np.float32) - outputs["register"].astype(np.float32)))),
    "shared_sha256": hashlib.sha256(outputs["shared"].tobytes()).hexdigest(),
    "register_sha256": hashlib.sha256(outputs["register"].tobytes()).hexdigest(),
  }
  if not exact["bitwise_equal"]: raise RuntimeError(f"exactness gate failed: {exact}")

  for _ in range(500):
    calls["shared"](src).realize(); calls["register"](src).realize()
  Device[dev].synchronize()
  samples: dict[str, list[list[float]]] = {}
  for arm in ("register", "shared", "register"):
    vals = []
    for _ in range(reps):
      Device[dev].synchronize(); begin = time.perf_counter_ns()
      for _ in range(replays): calls[arm](src).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns() - begin) / 1e3 / replays)
    samples.setdefault(arm, []).append(vals)
  candidate = (statistics.median(samples["register"][0]) + statistics.median(samples["register"][1])) / 2
  control = statistics.median(samples["shared"][0])
  result = {
    "schema": "tinygrad.nv_flash_combine_register_microgate.v1",
    "device": str(dev), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape": {"heads": HQ, "head_dim": HD, "splits": splits, "stride": W, "output_dtype": "float16"},
    "payload_sha256": hashlib.sha256(payload.tobytes()).hexdigest(), "exact": exact,
    "timing": {"unit": "us_per_launch_host_synchronized", "replays": replays, "reps": reps, "samples": samples,
               "candidate_register_midpoint": candidate, "control_shared": control,
               "recovery_us": control - candidate, "ratio": candidate / control},
    "verdict": "PRIMITIVE_PASS" if candidate < control else "NO_GO_PRIMITIVE",
  }
  return result


def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--replays", type=int, default=2000)
  ap.add_argument("--reps", type=int, default=9); ap.add_argument("--splits", type=int, default=8)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args(); result = run(args.replays, args.reps, args.splits)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
