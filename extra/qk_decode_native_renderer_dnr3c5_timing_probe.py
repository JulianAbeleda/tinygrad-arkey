#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from typing import Any, Callable

import numpy as np

from tinygrad import GlobalCounters, Tensor
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q8_BYTES, build_fullrow_reduce
from extra.q8_ffn_asm_gateup_full import stats_ms
from extra.q8_ffn_fast_artifact_probe import read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.q8_ffn_hcq_artifact import q8_dequant
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program
from extra.qk_decode_native_renderer_dnr3c4_semantic_reduction_probe import build_dnr3c4_candidate


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def build_inputs(gguf: pathlib.Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  q8_host = np.frombuffer(q8_blocks(x), dtype=np.uint8).copy()
  q8_x = q8_dequant(q8_host.tobytes(), 4096)
  q40, rows, k, _shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", HIDDEN)
  q41, rows1, k1, _shape1 = read_q4(gguf, "blk.0.ffn_up.weight", HIDDEN)
  if rows != HIDDEN or rows1 != HIDDEN or k != 4096 or k1 != 4096: raise ValueError((rows, rows1, k, k1))
  ref0, ref1 = q4_ref_rows(q40, rows, k, q8_x), q4_ref_rows(q41, rows, k, q8_x)
  return np.frombuffer(q40, dtype=np.uint32).copy(), np.frombuffer(q41, dtype=np.uint32).copy(), q8_host, ref0, ref1


def prepare_kernel(fxn: Callable, gate_words_host: np.ndarray, up_words_host: np.ndarray, q8_host: np.ndarray) -> tuple[Tensor, Tensor, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor(gate_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=fxn)[:2]
  return gate, up, gate.schedule_linear()


def time_linear(linear: Any, warmups: int, iters: int) -> dict[str, Any]:
  samples: list[float] = []
  for i in range(warmups + iters):
    GlobalCounters.reset()
    t0 = time.perf_counter()
    run_linear(linear)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    device_ms = GlobalCounters.time_sum_s * 1000.0
    if i >= warmups: samples.append(device_ms if device_ms > 0 else elapsed_ms)
  return stats_ms(samples)


def correctness(gate: Tensor, up: Tensor, ref0: np.ndarray, ref1: np.ndarray) -> dict[str, float]:
  got0, got1 = gate.numpy().astype(np.float32), up.numpy().astype(np.float32)
  err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
  return {
    "gate_max_abs": float(err0.max()),
    "gate_mean_abs": float(err0.mean()),
    "up_max_abs": float(err1.max()),
    "up_mean_abs": float(err1.mean()),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR-3C5 timing for native DNR-2 vs DNR-3C4 compound candidate")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=4)
  ap.add_argument("--iters", type=int, default=12)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c5_timing_result.json"))
  args = ap.parse_args()

  dnr3c4 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c4_semantic_reduction_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(args.gguf, args.seed)

  native_gate, native_up, native_linear = prepare_kernel(build_fullrow_reduce, gate_words_host, up_words_host, q8_host)
  c4_gate, c4_up, c4_linear = prepare_kernel(build_dnr3c4_candidate, gate_words_host, up_words_host, q8_host)

  # Compile/first-run both before measured loops.
  run_linear(native_linear)
  run_linear(c4_linear)
  native_timing = time_linear(native_linear, args.warmups, args.iters)
  c4_timing = time_linear(c4_linear, args.warmups, args.iters)
  native_correctness = correctness(native_gate, native_up, ref0, ref1)
  c4_correctness = correctness(c4_gate, c4_up, ref0, ref1)

  # Rebuild tiny programs for static counts without depending on measured tensors' internals.
  dummy_gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  dummy_up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  dummy_gw = Tensor.empty(gate_words_host.size, dtype=dtypes.uint32, device="AMD").contiguous()
  dummy_uw = Tensor.empty(up_words_host.size, dtype=dtypes.uint32, device="AMD").contiguous()
  dummy_q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  native_grouped = grouped(insts_from_program(build_fullrow_reduce(dummy_gate.uop, dummy_up.uop, dummy_gw.uop, dummy_uw.uop, dummy_q8.uop)))
  c4_grouped = grouped(insts_from_program(build_dnr3c4_candidate(dummy_gate.uop, dummy_up.uop, dummy_gw.uop, dummy_uw.uop, dummy_q8.uop)))

  oracle_us = float(oracle.get("known_timings_us", {}).get("hipcc_lld_gateup_current_loader", 0.0))
  native_us = native_timing["median_ms"] * 1000.0
  c4_us = c4_timing["median_ms"] * 1000.0
  gates = {
    "dnr3c4_passed": dnr3c4.get("gate_pass") is True,
    "native_correct": native_correctness["gate_max_abs"] <= 2e-3 and native_correctness["up_max_abs"] <= 2e-3,
    "c4_correct": c4_correctness["gate_max_abs"] <= 2e-3 and c4_correctness["up_max_abs"] <= 2e-3,
    "c4_faster_than_native": c4_us < native_us,
    "c4_lte_oracle_110pct": c4_us <= oracle_us * 1.10,
    "consumer_lte_60us": c4_us <= 60.0,
    "performance_measured": True,
  }
  if not gates["c4_correct"]:
    verdict = "BLOCKED_DNR3C5_C4_TIMING_CANDIDATE_INCORRECT"
  elif gates["c4_lte_oracle_110pct"] or gates["consumer_lte_60us"]:
    verdict = "PASS_DNR3C5_C4_TIMING_CLOSE_ENOUGH_FOR_PROMOTION_SCOPE"
  elif gates["c4_faster_than_native"]:
    verdict = "BLOCKED_DNR3C5_C4_IMPROVES_BUT_REMAINS_BEHIND_ORACLE"
  else:
    verdict = "BLOCKED_DNR3C5_C4_STATIC_SHAPE_DID_NOT_IMPROVE_TIMING"

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C5_DECODE_COMPOUND_CANDIDATE_TIMING",
    "schema": "decode_native_renderer_dnr3c5_timing_probe_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "timing": {
      "warmups": args.warmups,
      "iters": args.iters,
      "native": native_timing,
      "dnr3c4": c4_timing,
      "native_median_us": native_us,
      "dnr3c4_median_us": c4_us,
      "oracle_consumer_us": oracle_us,
      "dnr3c4_minus_native_us": c4_us - native_us,
      "dnr3c4_minus_oracle_us": c4_us - oracle_us,
    },
    "correctness": {"native": native_correctness, "dnr3c4": c4_correctness},
    "grouped": {"native": native_grouped, "dnr3c4": c4_grouped, "oracle": oracle["instruction_contract"]["oracle_grouped"]},
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C6 branch/wait attribution or promotion scope",
      "reason": "Timing decides whether the remaining branch/wait mismatch is worth implementing.",
      "minimum_unblock": [
        "if DNR-3C4 remains materially behind oracle, attribute whether branch/wait mismatch or marker placement is causal",
        "if DNR-3C4 is close enough, scope promotion hardening instead of more oracle count matching",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c4_semantic_reduction_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      str(args.gguf),
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "timing_us": {"native": native_us, "dnr3c4": c4_us, "oracle": oracle_us},
    "gates": gates,
    "out": str(args.out),
  }, indent=2))
  return 0 if gates["native_correct"] and gates["c4_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
