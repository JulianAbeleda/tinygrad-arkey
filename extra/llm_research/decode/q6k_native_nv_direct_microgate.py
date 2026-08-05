#!/usr/bin/env python3
"""Default-off native-NV Q6_K 1024x4096 contiguous-output microgate."""
from __future__ import annotations

import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import emit_q6k_gemv_kernel, q6k_spec_for_role
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference


def _program(spec):
  return KernelProgram("research.q6k_native_nv_direct", spec.kernel_name,
    KernelProgramProvenance.RESEARCH_ONLY, emit_q6k_gemv_kernel(spec))


def run(replays:int, reps:int, row_tile:int) -> dict:
  dev, rows, k = Device.DEFAULT, 1024, 4096
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  halfs_np = _make_q6k_halfs(rows, k, 20260805)
  x_np = np.random.default_rng(20260805).normal(0, .2, k).astype(np.float16)
  halfs = Tensor(halfs_np.copy(), dtype=dtypes.uint16, device=dev).contiguous().realize()
  x = Tensor(x_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()
  baseline_spec = q6k_spec_for_role(rows, k, role="attn_kv", parts=4, use_coop=False, reduction="external_sum")
  direct_spec = q6k_spec_for_role(rows, k, role="attn_kv", parts=1, row_tile=row_tile, use_coop=True, reduction="in_kernel")

  @TinyJit
  def baseline(w:Tensor, a:Tensor):
    parts = execute_research_program(Tensor.empty((rows, 4), dtype=dtypes.float32, device=dev), w, a,
                                     program=_program(baseline_spec))
    return parts.sum(axis=1).contiguous()

  @TinyJit
  def direct(w:Tensor, a:Tensor):
    return execute_research_program(Tensor.empty((rows,), dtype=dtypes.float32, device=dev), w, a,
                                    program=_program(direct_spec))

  # Eager execution then capture/build.
  baseline(halfs, x).realize(); base_out = baseline(halfs, x).realize()
  direct(halfs, x).realize(); direct_out = direct(halfs, x).realize()
  Device[dev].synchronize()
  got_base, got_direct = base_out.numpy().astype(np.float32), direct_out.numpy().astype(np.float32)
  raw = halfs_np.view(np.uint8)
  weights = q6_k_reference(Tensor(raw.copy(), dtype=dtypes.uint8), rows*k).numpy().astype(np.float32).reshape(rows, k)
  ref = weights @ x_np.astype(np.float32)

  def timed(fn) -> list[float]:
    samples = []
    for _ in range(reps):
      Device[dev].synchronize(); st = time.perf_counter_ns()
      for _ in range(replays): fn(halfs, x).realize()
      Device[dev].synchronize(); samples.append((time.perf_counter_ns()-st)/1e3/replays)
    return samples

  a, b, c = timed(baseline), timed(direct), timed(baseline)
  base_mid = (statistics.median(a)+statistics.median(c))/2
  err = np.abs(got_direct-ref)
  base_err = np.abs(got_base-ref)
  cross = np.abs(got_direct-got_base)
  atol = max(2e-2, float(np.max(np.abs(ref)))*2e-4)
  return {
    "schema":"tinygrad.q6k_native_nv_direct_microgate.v1", "device":str(dev),
    "git_commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "payload":{"q6_sha256":hashlib.sha256(raw).hexdigest(), "x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "candidate":{"id":f"q6k_1024x4096_native_nv_direct_rt{row_tile}_v1", "spec":direct_spec.to_json()},
    "baseline":{"spec":baseline_spec.to_json(), "tail":"sum(axis=1).contiguous"},
    "correctness":{"atol":atol, "candidate_max_abs_ref":float(err.max()),
      "baseline_max_abs_ref":float(base_err.max()), "candidate_vs_baseline_max_abs":float(cross.max()),
      "pass":bool(err.max() <= atol)},
    "timing":{"unit":"us_per_graph_replay", "replays":replays, "reps":reps,
      "control_a":a, "candidate_b":b, "control_c":c, "control_midpoint_median":base_mid,
      "candidate_median":statistics.median(b), "delta":statistics.median(b)-base_mid},
  }


def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--replays", type=int, default=500)
  ap.add_argument("--reps", type=int, default=7); ap.add_argument("--row-tile", type=int, choices=(1, 2), default=2); ap.add_argument("--out")
  a = ap.parse_args(); result = run(a.replays, a.reps, a.row_tile); text = json.dumps(result, indent=2, sort_keys=True)
  if a.out:
    with open(a.out, "w") as f: f.write(text+"\n")
  print(text)
  return 0 if result["correctness"]["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
