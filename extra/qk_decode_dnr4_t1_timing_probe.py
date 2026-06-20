#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from extra.q8_ffn_asm_fullrow_reduce import build_fullrow_reduce
from extra.qk_decode_dnr4_t1_reduction_reuse_probe import build_fullrow_reduce_dnr4_t1
from extra.qk_decode_native_renderer_dnr3c6_attribution_scope import build_b128_dsload_b128_no_markers
from extra.qk_decode_native_renderer_dnr3c7c_issue_interleaving_probe import (
  ROOT, build_inputs, build_unpack_all_then_dot_dsload_b128, correctness,
  prepare_kernel, static_grouped, time_interleaved,
)


OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dnr4_t1_timing_result.json"


def build_timing_rows(gguf: Path, seed: int, warmups: int, iters: int) -> list[dict[str, Any]]:
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(gguf, seed)
  variants: list[tuple[str, Callable]] = [
    ("native_dnr2", build_fullrow_reduce),
    ("best_static_dnr3c6", build_b128_dsload_b128_no_markers),
    ("c7c_best_unpack_dot_dsload_b128", build_unpack_all_then_dot_dsload_b128),
    ("dnr4_t1_reduction_reuse", build_fullrow_reduce_dnr4_t1),
  ]
  rows: list[dict[str, Any]] = []
  for name, fxn in variants:
    gate, up, linear = prepare_kernel(fxn, gate_words_host, up_words_host, q8_host)
    # Match the existing C7C harness: materialize once before reading correctness.
    from tinygrad.device import Device
    from tinygrad.engine.realize import run_linear
    run_linear(linear)
    Device["AMD"].synchronize()
    rows.append({
      "name": name,
      "linear": linear,
      "correctness": correctness(gate, up, ref0, ref1),
      "grouped": static_grouped(fxn, gate_words_host.size),
    })
  for row in rows:
    row["correct"] = row["correctness"]["gate_max_abs"] <= 2e-3 and row["correctness"]["up_max_abs"] <= 2e-3
  time_interleaved(rows, warmups, iters)
  for row in rows:
    del row["linear"]
  by_name = {row["name"]: row for row in rows}
  native_us = by_name["native_dnr2"]["median_us"]
  best_static_us = by_name["best_static_dnr3c6"]["median_us"]
  t1_us = by_name["dnr4_t1_reduction_reuse"]["median_us"]
  for row in rows:
    row["delta_vs_native_us"] = row["median_us"] - native_us
    row["delta_vs_best_static_us"] = row["median_us"] - best_static_us
    row["delta_vs_dnr4_t1_us"] = row["median_us"] - t1_us
  return rows


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR4-T1 same-harness timing")
  ap.add_argument("--gguf", type=Path, default=Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=4)
  ap.add_argument("--iters", type=int, default=12)
  ap.add_argument("--out", type=Path, default=OUT)
  args = ap.parse_args()

  rows = build_timing_rows(args.gguf, args.seed, args.warmups, args.iters)
  by_name = {row["name"]: row for row in rows}
  t1 = by_name["dnr4_t1_reduction_reuse"]
  native = by_name["native_dnr2"]
  best_static = by_name["best_static_dnr3c6"]
  c7c = by_name["c7c_best_unpack_dot_dsload_b128"]
  all_correct = all(row["correct"] for row in rows)
  t1_gain_vs_native = -t1["delta_vs_native_us"]
  t1_gain_vs_best = -t1["delta_vs_best_static_us"]
  t1_gain_vs_c7c = c7c["median_us"] - t1["median_us"]
  material = t1_gain_vs_native >= 30.0 or t1_gain_vs_best >= 15.0 or t1_gain_vs_c7c >= 10.0
  gates = {
    "all_variants_correct": all_correct,
    "t1_correct": t1["correct"],
    "t1_timed": "median_us" in t1,
    "t1_material_timing": material,
    "no_default_change": True,
  }
  if not all_correct:
    verdict = "BLOCKED_DNR4_T1_TIMING_INCORRECT"
  elif material:
    verdict = "PASS_DNR4_T1_TIMING_MATERIAL_SCOPE_PROMOTION"
  else:
    verdict = "BLOCKED_DNR4_T1_STRUCTURAL_ONLY_TIMING_NOT_MATERIAL"

  result = {
    "date": "2026-06-20",
    "phase": "DNR4_T1_REDUCTION_REUSE_TIMING",
    "schema": "decode_dnr4_t1_timing_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "timing_harness": {"warmups": args.warmups, "iters": args.iters, "method": "same-process interleaved timing"},
    "timing_context": {
      "native_us": native["median_us"],
      "best_static_us": best_static["median_us"],
      "c7c_best_us": c7c["median_us"],
      "dnr4_t1_us": t1["median_us"],
      "t1_gain_vs_native_us": t1_gain_vs_native,
      "t1_gain_vs_best_static_us": t1_gain_vs_best,
      "t1_gain_vs_c7c_us": t1_gain_vs_c7c,
    },
    "rows": rows,
    "gates": gates,
    "decision": {
      "if_blocked": "Treat DNR4-T1 as structural cleanup only and proceed to DNR4-T2 dot-body vector-band compression.",
      "if_pass": "Only then add PMC/resource confirmation before promotion.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "timing_context": result["timing_context"],
    "gates": gates,
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if all_correct else 1


if __name__ == "__main__":
  raise SystemExit(main())
