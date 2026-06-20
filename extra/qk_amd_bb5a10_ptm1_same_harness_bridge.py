#!/usr/bin/env python3
# PTM-1 — same-harness authority bridge.
# Times the captured tinygrad authority kernel and the two hand-ASM P8 candidates (LDS DS64 macro,
# global-direct pipe T4x2) under ONE process, ONE clock, INTERLEAVED round-robin. This resolves whether
# the prior 43-vs-18-21 TFLOPS gap is real kernel quality or a harness/clock artifact (the 43.026 came
# from tinygrad's _time_program best-of-7, the 18-21 from host-wall perf_counter+synchronize).
from __future__ import annotations

import json, pathlib, re, subprocess, sys, time
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*m")

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.gemm.rdna3_wmma_matmul import _run_insts, _run_insts_lds, build_gemm_pipe
from extra.qk_amd_bb5a10_p8_tta3a_ds64_macro_conversion import build_converted_macro_insts
from tinygrad import Context, Device, Tensor
from tinygrad.codegen import to_program
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import getenv
from tinygrad.llm.model import _prefill_v2_opts
from tinygrad.uop.ops import Ops, UOp

OUT = ROOT / "bench/amd-broad-backend-roadmap"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
AUTHORITY_NAME = "r_16_192_32_2_2_2_2_4_32_2_8"
LDS_BYTES, BM, BN, THREADS = 8192, 128, 128, 128


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sclk_reading() -> str | None:
  try:
    out = subprocess.run(["rocm-smi", "--showgpuclocks"], capture_output=True, text=True, timeout=10).stdout
    lines = [ln.strip() for ln in out.splitlines() if "sclk" in ln.lower()]
    return " | ".join(lines) if lines else None
  except Exception:
    return None


def build_authority() -> tuple[UOp, dict[str, Any], list[Any]]:
  # Recompile the captured authority kernel from its exact shape + prefill_v2 opts (identical to the
  # BB-5a.8 capture probe). More robust than raw-ELF VA patching; yields correctly-ordered buffers and
  # the same r_16_192_32_..._8 kernel. helper_realized_ast returns (ast, [out, a, b]).
  from test.backend.test_linearizer import helper_realized_ast
  from test.helpers import replace_opts
  a = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  b = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(a @ b)
  prg = to_program(replace_opts(ast, tuple(_prefill_v2_opts(N, K))), Device["AMD"].renderer)
  call = prg.call(*[UOp.from_buffer(buf) for buf in bufs])
  linear = UOp(Ops.LINEAR, src=(call,))
  raw_name = getattr(prg.arg, "name", None) or getattr(prg.arg, "function_name", None) or ""
  clean_name = ANSI.sub("", raw_name)  # prg.arg.name carries terminal color escapes
  meta = {
    "name": clean_name,
    "global_size": list(getattr(prg.arg, "global_size", []) or []),
    "local_size": list(getattr(prg.arg, "local_size", []) or []),
    "expected_name": AUTHORITY_NAME,
    "name_matches": clean_name == AUTHORITY_NAME,
    "source": "recompiled_from_shape_and_prefill_v2_opts",
  }
  return linear, meta, [a, b]  # keep tensors alive so bufs stay resident


def build_candidates() -> tuple[list[tuple[str, UOp]], list[Tensor]]:
  rng = np.random.default_rng(13)
  a = Tensor(((rng.standard_normal((M, K)) * 0.1).astype(np.float16)), device="AMD")
  bt = Tensor(((rng.standard_normal((N, K)) * 0.1).astype(np.float16)), device="AMD")  # WMMA B layout: N x K
  c_lds = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
  c_gd = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
  Tensor.realize(a, bt, c_lds, c_gd)
  lds_insts, _ = build_converted_macro_insts()
  lds_linear, _ = _run_insts_lds(lds_insts, a, bt, c_lds, M, N, K, "ptm1_lds_macro_ds64", LDS_BYTES, BM, BN, THREADS)
  gd_linear, _ = _run_insts(build_gemm_pipe(M, N, K, 4, 2), a, bt, c_gd, M, N, K, 4, 2, "ptm1_global_direct_pipe_T4x2")
  return [("lds_macro_ds64", lds_linear), ("global_direct_pipe_T4x2", gd_linear)], [a, bt, c_lds, c_gd]


def time_interleaved(specs: list[tuple[str, UOp]], cnt: int) -> dict[str, list[float]]:
  times: dict[str, list[float]] = {name: [] for name, _ in specs}
  with Context(DEBUG=0):
    for _, lin in specs:  # warmup (compile + clock ramp), excluded from timing
      Device["AMD"].synchronize(); run_linear(lin); Device["AMD"].synchronize()
    for _ in range(cnt):  # round-robin: clock drift/boost hits every kernel equally
      for name, lin in specs:
        Device["AMD"].synchronize()
        t0 = time.perf_counter()
        run_linear(lin)
        Device["AMD"].synchronize()
        times[name].append(time.perf_counter() - t0)
  return times


def stats(ets: list[float]) -> dict[str, Any]:
  best = min(ets)
  median = sorted(ets)[len(ets) // 2]
  return {"best_s": best, "median_s": median, "best_tflops": FLOP / best * 1e-12,
          "median_tflops": FLOP / median * 1e-12, "times_s": ets}


def main() -> int:
  cnt = getenv("CNT", 30)
  sclk_start = sclk_reading()
  try:
    auth_linear, auth_meta, _keep_auth = build_authority()
    cand_specs, _keep_cand = build_candidates()
    specs = [("authority_tinygrad", auth_linear)] + cand_specs
    times = time_interleaved(specs, cnt)
    sclk_end = sclk_reading()
    per_kernel = {name: stats(times[name]) for name, _ in specs}
    error = None
  except Exception as e:
    auth_meta, per_kernel, sclk_end, error = {}, {}, None, repr(e)

  ran = bool(per_kernel) and all("best_tflops" in per_kernel.get(n, {}) for n in
                                 ("authority_tinygrad", "lds_macro_ds64", "global_direct_pipe_T4x2"))
  auth_tf = per_kernel.get("authority_tinygrad", {}).get("best_tflops")
  cand_tf = max((per_kernel.get(n, {}).get("best_tflops") or 0.0)
                for n in ("lds_macro_ds64", "global_direct_pipe_T4x2")) if ran else None
  ratio = (auth_tf / cand_tf) if (ran and cand_tf) else None
  if ratio is None: interpretation = "NOT_RUN"
  elif ratio >= 1.3: interpretation = "GAP_REAL_KERNEL_QUALITY"
  elif ratio <= 1.1: interpretation = "GAP_WAS_HARNESS_ARTIFACT"
  else: interpretation = "GAP_PARTIAL_AMBIGUOUS"

  # Contrast vs the PRIOR cross-harness numbers (authority 43.026 via _time_program; candidates 18.4/17.9
  # via a separate, lower-clock host-wall session). The cross-harness ratio overstates the gap; PTM-1's
  # one-clock ratio is the trustworthy one.
  lds_tf = per_kernel.get("lds_macro_ds64", {}).get("best_tflops")
  gd_tf = per_kernel.get("global_direct_pipe_T4x2", {}).get("best_tflops")
  prior_cross_harness = {
    "authority_time_program_tflops": 43.026, "lds_macro_prior_tflops": 18.383, "global_direct_prior_tflops": 17.881,
    "prior_cross_harness_ratio_authority_over_lds": 43.026 / 18.383,
    "note": "prior candidate numbers were a lower-clock session (separate process); not comparable to authority's clock. PTM-1 ratio supersedes.",
  }

  gate = {
    "all_three_ran_one_harness": ran,
    "single_process_single_clock": ran,
    "interleaved_round_robin": ran,
    "authority_kernel_identity_matches": bool(auth_meta.get("name_matches")),
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "phase": "PTM-1_same_harness_authority_bridge",
    "schema": "amd_bb5a10_ptm1_same_harness_bridge_v1",
    "verdict": "PASS_PTM1_SAME_HARNESS_AUTHORITY_BRIDGED" if gate_pass else "BLOCKED_PTM1_SAME_HARNESS_AUTHORITY_BRIDGE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": True,
    "error": error,
    "shape": [M, N, K],
    "flop": FLOP,
    "cnt": cnt,
    "harness": "single-process interleaved round-robin; per-launch Device['AMD'].synchronize()+perf_counter; warm cache; warmup excluded; best-of-N",
    "authority": auth_meta,
    "authority_prior_time_program_tflops": 43.026,
    "per_kernel": per_kernel,
    "authority_best_tflops": auth_tf,
    "best_candidate_tflops": cand_tf,
    "authority_over_best_candidate_ratio": ratio,
    "interpretation": interpretation,
    "prior_cross_harness": prior_cross_harness,
    "clock_provenance": {"sclk_start": sclk_start, "sclk_end": sclk_end,
                         "note": "interleaving is the primary clock control; sclk is provenance only"},
    "gate": gate,
    "next_action": "PTM-2 prefill primitive decision: GAP_REAL_KERNEL_QUALITY -> choose software_pipelined_k_loop; "
                   "GAP_WAS_HARNESS_ARTIFACT -> re-baseline (candidates already near true tinygrad authority).",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_global_direct_candidate_decision_result.json",
    ],
  }
  write_json("bb5a10_ptm1_same_harness_authority_bridge_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_ptm1_same_harness_authority_bridge_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "authority_tflops": auth_tf,
    "lds_macro_tflops": lds_tf,
    "global_direct_tflops": gd_tf,
    "ratio": ratio,
    "interpretation": interpretation,
    "prior_cross_harness_ratio": prior_cross_harness["prior_cross_harness_ratio_authority_over_lds"],
    "error": error,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
