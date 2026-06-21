#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def main() -> int:
  bb5a6 = read_json("bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json", {})
  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  rows = [row for row in shape.get("rows", []) if row.get("role") == "ffn_gate_up"]
  authority = rows[0] if rows else {}
  tinygrad_tflops = authority.get("tinygrad_tflops")
  tensile_tflops = authority.get("median_tflops")
  threshold = 60.0
  gate = {
    "input_bb5a6_pass": bb5a6.get("verdict") == "PASS_BB5A6_CORRECTNESS" and bool(bb5a6.get("gate_pass")),
    "authority_prefill_timing_present": tinygrad_tflops is not None,
    "pure_tinygrad_reaches_60_tflops": tinygrad_tflops is not None and float(tinygrad_tflops) >= threshold,
    "tensile_oracle_reaches_60_tflops": tensile_tflops is not None and float(tensile_tflops) >= threshold,
    "default_behavior_changed": False,
  }
  pass_gate = gate["input_bb5a6_pass"] and gate["pure_tinygrad_reaches_60_tflops"] and not gate["default_behavior_changed"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.7_performance_gate",
    "schema": "amd_bb5a7_performance_gate_result_v1",
    "verdict": "PASS_BB5A7_PERFORMANCE_GATE" if pass_gate else "BLOCKED_BB5A7_PERFORMANCE_GATE_NOT_MET",
    "gate_pass": pass_gate,
    "default_behavior_changed": False,
    "performance_claim": bool(pass_gate),
    "threshold_tflops": threshold,
    "authority_prefill": {
      "role": authority.get("role"),
      "tinygrad_tflops": tinygrad_tflops,
      "tensile_oracle_tflops": tensile_tflops,
      "source_artifact": "bench/qk-tensile-extraction/shape_matrix.json",
    },
    "gate": gate,
    "decision": (
      "BB-5a.7 is blocked: the current pure tinygrad authority prefill row is below the 60 TFLOPS gate, even though "
      "the Tensile oracle remains above it. Do not transfer to q8 yet."
      if not pass_gate else
      "BB-5a.7 passes: pure tinygrad authority prefill reaches the 60 TFLOPS gate."
    ),
    "next_action": (
      "Reopen performance work only with a measured pure tinygrad pipelined prefill candidate; keep BB-6 blocked."
      if not pass_gate else "Proceed to BB-6 q8 transfer scope."
    ),
  }
  write_json("bb5a7_performance_gate_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json",
    "verdict": result["verdict"],
    "gate_pass": pass_gate,
    "tinygrad_tflops": tinygrad_tflops,
    "threshold": threshold,
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
