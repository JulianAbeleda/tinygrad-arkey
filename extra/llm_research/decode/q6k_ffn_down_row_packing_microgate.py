#!/usr/bin/env python3
"""Exact native-NV microgate for packing multiple Q6 FFN-down rows per CTA."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.llm.q6k_ffn_down_mmvq import K, ROWS, emit_q6k_four_warp_fp16_direct
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs


def _program(rows_per_block:int) -> KernelProgram:
  return KernelProgram("research.q6k_ffn_down_row_packing", f"rpb{rows_per_block}",
    KernelProgramProvenance.RESEARCH_ONLY, emit_q6k_four_warp_fp16_direct(rows_per_block=rows_per_block))


def run(replays:int, reps:int) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  halfs_np = _make_q6k_halfs(ROWS, K, 2026082401)
  x_np = np.random.default_rng(2026082402).normal(0, 0.2, K).astype(np.float16)
  h_np = np.random.default_rng(2026082403).normal(0, 0.05, ROWS).astype(np.float32)
  halfs = Tensor(halfs_np.copy(), dtype=dtypes.uint16, device=dev).contiguous().realize()
  x = Tensor(x_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()
  h = Tensor(h_np.copy(), dtype=dtypes.float32, device=dev).contiguous().realize()

  def make_fn(rows_per_block:int):
    program = _program(rows_per_block)
    @TinyJit
    def fn(w:Tensor, a:Tensor, residual:Tensor):
      return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
        w, a, residual, program=program)
    return fn

  fns = {rpb:make_fn(rpb) for rpb in (1, 2, 4, 8)}
  outputs = {}
  for rpb, fn in fns.items():
    fn(halfs, x, h).realize()
    outputs[rpb] = fn(halfs, x, h).realize().numpy().astype(np.float32)
  Device[dev].synchronize()

  exact = {str(rpb):{"bitwise_identical":bool(np.array_equal(outputs[1].view(np.uint32), outputs[rpb].view(np.uint32))),
                     "max_abs_diff":float(np.max(np.abs(outputs[1]-outputs[rpb]))),
                     "finite":bool(np.isfinite(outputs[rpb]).all())} for rpb in (2, 4, 8)}

  def timed(fn) -> list[float]:
    samples=[]
    for _ in range(reps):
      Device[dev].synchronize(); start=time.perf_counter_ns()
      for _ in range(replays): fn(halfs, x, h).realize()
      Device[dev].synchronize(); samples.append((time.perf_counter_ns()-start)/1e3/replays)
    return samples

  samples = {str(rpb):timed(fns[rpb]) for rpb in (1, 2, 4, 8)}
  medians = {rpb:statistics.median(samples[str(rpb)]) for rpb in (1, 2, 4, 8)}
  best = min((2, 4, 8), key=medians.__getitem__)
  result = {"schema":"tinygrad.q6k_ffn_down_row_packing_microgate.v1", "device":str(dev),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "payload":{"q6_sha256":hashlib.sha256(halfs_np.view(np.uint8)).hexdigest(),
      "x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "one_change":"CTA rows; four warps and arithmetic order per row are unchanged",
    "correctness":exact, "timing":{"unit":"us_per_launch_host_synchronized", "replays":replays,
      "reps":reps, "samples":samples, "medians":{str(k):v for k,v in medians.items()},
      "best_rows_per_block":best, "best_recovery_us":medians[1]-medians[best]}}
  return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=300)
  ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out",type=pathlib.Path); args=ap.parse_args()
  result=run(args.replays,args.reps); text=json.dumps(result,indent=2,sort_keys=True)
  if args.out: args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(text+"\n")
  print(text)
  return 0 if all(x["bitwise_identical"] and x["finite"] for x in result["correctness"].values()) else 1


if __name__ == "__main__": raise SystemExit(main())
