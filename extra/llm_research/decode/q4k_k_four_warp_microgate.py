#!/usr/bin/env python3
"""Included-cost native-NV gate for exact four-warp Q4_K x fp16 K GEMV.

The control is the production attention-K path ``q4k_g3_lanemap_gemv_1024_4096``
(one warp per 1024-row output, 16 Q4_K blocks serialized inside a single lane
map).  The candidate is the already-landed exact four-warp Q4_K geometry
re-pointed at the K shape: 128 threads per output row, each warp owning four of
the row's 16 Q4_K blocks, no Q8 provider node, and a single contiguous fp32
output.  This is the row-starved case where four-warp ownership has a chance to
win; the 4096-row FFN shapes were already measured wall-neutral.

This is a measurement gate, not a route selector.  The candidate emitter lives
only in the imported research module and is never reachable from production
decode.
"""
from __future__ import annotations

import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from extra.llm_research.decode.q4k_exact_group_factorized import emit_q4k_exact_four_warp, oracle_gemv
from extra.llm_research.decode.route_class_numerics import _make_q4k_words

ROWS, K, WARPS_PER_ROW = 1024, 4096, 4
K_BLOCKS = K // 256
BLOCKS_PER_WARP = K_BLOCKS // WARPS_PER_ROW


def _program(name, emitter):
  return KernelProgram("research.q4k_k_four_warp", name,
    KernelProgramProvenance.RESEARCH_ONLY, emitter)


def run(replays: int = 200, reps: int = 7) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"):
    raise RuntimeError(f"native NV required, got {dev}")
  words_np, raw = _make_q4k_words(ROWS, K, 20260806)
  x_np = np.random.default_rng(20260806).normal(0, 0.2, K).astype(np.float16)
  words = Tensor(words_np.copy(), dtype=dtypes.uint32, device=dev).contiguous().realize()
  x = Tensor(x_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()

  baseline_program = _program("q4_installed", q4k_g3_lanemap_gemv_kernel(ROWS, K))
  candidate_program = _program("q4_exact_four_warp", emit_q4k_exact_four_warp(ROWS, K))

  @TinyJit
  def baseline(ww, xx):
    return execute_research_program(
      Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      ww, xx, program=baseline_program)

  @TinyJit
  def candidate(ww, xx):
    return execute_research_program(
      Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      ww, xx, program=candidate_program)

  baseline_out = baseline(words, x).realize()
  candidate_out = candidate(words, x).realize()
  Device[dev].synchronize()
  baseline_np = baseline_out.numpy().astype(np.float32)
  candidate_np = candidate_out.numpy().astype(np.float32)

  # Independent byte-layout/fp64 oracle on deterministic rows.  Neither check
  # reuses the candidate lane map or algebra.
  oracle_rows = np.asarray([0, 1, 17, 255, 511, 767, 1023], dtype=np.int64)
  row_words = K_BLOCKS * 36
  oracle_packed = np.concatenate([words_np[r * row_words:(r + 1) * row_words]
                                  for r in oracle_rows])
  oracle = oracle_gemv(oracle_packed, x_np, len(oracle_rows), K)
  selected = candidate_np[oracle_rows]
  oracle_max_abs = float(np.max(np.abs(selected - oracle)))
  oracle_tol = max(1e-3, float(np.max(np.abs(oracle))) * 1e-2)
  cross_max_abs = float(np.max(np.abs(candidate_np - baseline_np)))
  cross_rel_l2 = float(
    np.linalg.norm((candidate_np - baseline_np).astype(np.float64))
    / max(np.linalg.norm(baseline_np.astype(np.float64)), 1e-30))
  correctness = bool(
    np.isfinite(candidate_np).all()
    and oracle_max_abs <= oracle_tol
    and cross_rel_l2 <= 1e-3)
  if not correctness:
    raise RuntimeError(
      f"exact four-warp correctness failed: oracle {oracle_max_abs}/{oracle_tol}, "
      f"relL2 {cross_rel_l2}")

  for _ in range(200):
    baseline(words, x).realize()
    candidate(words, x).realize()

  def timed(fn):
    samples = []
    for _ in range(reps):
      Device[dev].synchronize()
      start = time.perf_counter_ns()
      for _ in range(replays):
        fn()
      Device[dev].synchronize()
      samples.append((time.perf_counter_ns() - start) / 1e3 / replays)
    return samples

  control_a = timed(lambda: baseline(words, x).realize())
  candidate_b = timed(lambda: candidate(words, x).realize())
  control_c = timed(lambda: baseline(words, x).realize())
  midpoint = (statistics.median(control_a) + statistics.median(control_c)) / 2
  cand_median = statistics.median(candidate_b)
  material_win = bool(cand_median <= 0.95 * midpoint)
  return {
    "schema": "tinygrad.q4k_k_four_warp_microgate.v1",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape": {"rows": ROWS, "k": K, "k_blocks": K_BLOCKS,
              "warps_per_row": WARPS_PER_ROW, "blocks_per_warp": BLOCKS_PER_WARP},
    "representation": {
      "weights": "production packed Q4_K uint32",
      "activation": "production fp16",
      "approximation": False,
    },
    "payload": {
      "q4_sha256": hashlib.sha256(raw.reshape(-1).tobytes()).hexdigest(),
      "x_sha256": hashlib.sha256(x_np.tobytes()).hexdigest(),
    },
    "census": {"control_programs_per_replay": 1, "candidate_programs_per_replay": 1},
    "correctness": {
      "pass": correctness,
      "oracle_rows": oracle_rows.tolist(),
      "oracle_max_abs": oracle_max_abs,
      "oracle_tol": oracle_tol,
      "candidate_vs_installed_max_abs": cross_max_abs,
      "candidate_vs_installed_relative_l2": cross_rel_l2,
      "full_logit_status": "not run; primitive is default-off and cannot be promoted from this gate",
    },
    "timing": {
      "unit": "us_per_kernel",
      "replays": replays,
      "reps": reps,
      "control_a": control_a,
      "candidate_b": candidate_b,
      "control_c": control_c,
      "control_midpoint_median": midpoint,
      "candidate_median": cand_median,
      "ratio": cand_median / midpoint,
      "delta": cand_median - midpoint,
      "gate": "PASS" if material_win else "FAIL",
      "material_win_threshold": "candidate <= 0.95 * control midpoint",
    },
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--replays", type=int, default=200)
  parser.add_argument("--reps", type=int, default=7)
  parser.add_argument("--out")
  args = parser.parse_args()
  result = run(args.replays, args.reps)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  print(encoded, flush=True)
  if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(encoded + "\n")
