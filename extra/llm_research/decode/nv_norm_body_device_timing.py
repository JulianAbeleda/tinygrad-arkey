#!/usr/bin/env python3
"""Device-side (CUPTI/nsys) timing of decode norm kernels.

Compares the bitwise serial-chain reduce-output body against the warp-reduce
native body at the three decode norm shapes (q 32x128, k 8x128, ffn 1x4096).
Answers two things for the "relax the bitwise contract" decision: (a) is the
warp-reduce body faster at the device level, and (b) does it flip the output.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess, time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import Ops, ReduceOutputSpec
from tinygrad.llm.kernel_program import (
  KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program)
from tinygrad.codegen.late.reduce_output import emit_reduce_output
from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel

SHAPES = {"q_32x128": (32, 128, dtypes.float32), "k_8x128": (8, 128, dtypes.float32),
          "ffn_1x4096": (1, 4096, dtypes.float16), "out_1x4096": (1, 4096, dtypes.float32)}
EPS = 1e-6


def _reduce_spec(rows: int, dim: int, out_dtype) -> ReduceOutputSpec:
  if rows == 1:
    warps, lanes, per_lane = 16, 32, 8
  else:
    warps, lanes = rows, 32
    per_lane = rows * dim // (warps * lanes)
  return ReduceOutputSpec(rows=rows, dim=dim, eps=EPS, out_dtype=out_dtype,
                          affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                          warps=warps, lanes=lanes, per_lane=per_lane)


def _native_spec(rows: int, dim: int, out_dtype) -> DecodeRMSNormSpec:
  warps = 16 if (rows, dim) == (1, 4096) else 1
  return DecodeRMSNormSpec(rows=rows, dim=dim, eps=EPS, lane_width=32, warps_per_row=warps,
                           x_dtype=dtypes.float16, weight_dtype=dtypes.float16,
                           out_dtype=out_dtype, x_rank=1, native=True)


def _programs() -> dict[str, tuple[KernelProgram, int, int]]:
  out = {}
  for name, (rows, dim, out_dtype) in SHAPES.items():
    serial = KernelProgram(
      "research.nv_norm_body_device_timing", f"serial_{name}",
      KernelProgramProvenance.RESEARCH_ONLY,
      emit_reduce_output(_reduce_spec(rows, dim, out_dtype), dtypes.float16, dtypes.float16),
      output_spec=OutputSpec((rows * dim,), out_dtype))
    warp = KernelProgram(
      "research.nv_norm_body_device_timing", f"warp_{name}",
      KernelProgramProvenance.RESEARCH_ONLY,
      emit_decode_rmsnorm_kernel(_native_spec(rows, dim, out_dtype)),
      output_spec=OutputSpec((rows * dim,), out_dtype))
    out[f"serial_{name}"] = (serial, rows, dim)
    out[f"warp_{name}"] = (warp, rows, dim)
  return out


def _inputs(rows: int, dim: int) -> tuple[Tensor, Tensor]:
  rng = np.random.default_rng(20260819)
  x = rng.normal(0, 0.2, rows * dim).astype(np.float16)
  w = rng.normal(1, 0.05, dim).astype(np.float16)
  return (Tensor(x, device="CUDA").contiguous().realize(),
          Tensor(w, device="CUDA").contiguous().realize())


def run(replays: int = 400, warmup: int = 20, only: str | None = None) -> dict:
  if Device.DEFAULT != "CUDA":
    raise RuntimeError(f"DEV=CUDA required for CUPTI timing, got {Device.DEFAULT}")
  programs = _programs()
  if only is not None:
    if only not in programs:
      raise ValueError(f"unknown kernel {only!r}; choose from {sorted(programs)}")
    programs = {only: programs[only]}
  result = {"schema": "tinygrad.nv_norm_body_device_timing.v1",
            "device": str(Device.DEFAULT),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "replays": replays, "warmup": warmup, "runs": {}}
  ref_sha = {}
  for name, (program, rows, dim) in programs.items():
    x, w = _inputs(rows, dim)
    out_dtype = program.output_spec.dtype
    dst = Tensor.empty(rows * dim, dtype=out_dtype, device="CUDA")
    res = execute_research_program(dst, x, w, program=program)
    res.realize(); Device["CUDA"].synchronize()
    arr = np.asarray(res.numpy())
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    key = name.split("_", 1)[1]
    ref_sha.setdefault(key, {"serial": None, "warp": None})
    ref_sha[key]["serial" if name.startswith("serial") else "warp"] = sha
    ref_sha[key]["out_dtype"] = str(out_dtype)
    for _ in range(warmup):
      execute_research_program(dst, x, w, program=program).realize()
    Device["CUDA"].synchronize()
    for _ in range(replays):
      execute_research_program(dst, x, w, program=program).realize()
    Device["CUDA"].synchronize()
    result["runs"][name] = {"rows": rows, "dim": dim, "out_dtype": str(out_dtype), "sha256": sha,
                            "nsys_marker": f"START_{name}_x{replays}_END"}
    time.sleep(0.05)
  for key, shas in ref_sha.items():
    shas["same_sha"] = shas["serial"] == shas["warp"]
  result["sha_comparison"] = ref_sha
  return result


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--replays", type=int, default=400)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--only")
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  got = run(args.replays, args.warmup, args.only)
  print(json.dumps(got, indent=2, sort_keys=True))
  if args.out:
    args.out.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
