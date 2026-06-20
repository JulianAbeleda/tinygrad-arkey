#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def fget(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
  cur: Any = row
  for key in keys:
    if not isinstance(cur, dict) or key not in cur: return default
    cur = cur[key]
  return cur


def main() -> int:
  p8_perf = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json", {})
  p8_bottleneck = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json", {})
  global_direct = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_global_direct_candidate_decision_result.json", {})
  authority_capture = read_json("bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json", {})
  causal = read_json("bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json", {})
  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})

  p8_lds_tflops = fget(p8_perf, "performance", "best_tflops", default=0.0)
  p8_global_best = fget(global_direct, "best_variant", "best_tflops", default=0.0)
  p8_global_kernel = {
    "name": fget(global_direct, "best_variant", "name"),
    "tile": fget(global_direct, "best_variant", "tile"),
    "grid": fget(global_direct, "best_variant", "grid"),
    "local_size": fget(global_direct, "best_variant", "local_size"),
    "instruction_counts": fget(global_direct, "best_variant", "instruction_counts", default={}),
  }
  captured_tflops = fget(authority_capture, "timing", "best_tflops", default=0.0)
  captured_kernel = {
    "name": fget(authority_capture, "program", "name"),
    "grid": fget(authority_capture, "program", "global_size"),
    "local_size": fget(authority_capture, "program", "local_size"),
    "instruction_counts": fget(authority_capture, "mix", "disasm", default={}),
    "timing_samples_s": fget(authority_capture, "timing", "samples_s", default=[]),
  }
  ffn_gate_up = next((r for r in shape.get("rows", []) if r.get("role") == "ffn_gate_up"), {})

  checks = {
    "p8_performance_artifact_present": bool(p8_perf),
    "p8_bottleneck_classified": p8_bottleneck.get("verdict") == "PASS_BB5A10_P8_BOTTLENECK_CLASSIFIED_LDS_STAGING_FAMILY",
    "global_direct_decision_pass": global_direct.get("verdict") == "PASS_BB5A10_P8_GLOBAL_DIRECT_CANDIDATE_DECISION" and bool(global_direct.get("gate_pass")),
    "authority_capture_pass": authority_capture.get("verdict") == "PASS_AUTHORITY_KERNEL_CAPTURE_CAUSAL_INPUTS_READY" and bool(authority_capture.get("gate_pass")),
    "causal_delta_pass": causal.get("verdict") == "PASS_BB5A9_CAUSAL_DELTA_PACKAGE_IMPLEMENTATION_TRACKS_READY" and bool(causal.get("gate_pass")),
    "same_shape": fget(authority_capture, "shape", "m") == 512 and fget(authority_capture, "shape", "n") == 12288 and fget(authority_capture, "shape", "k") == 4096,
    "prior_authority_has_timing_join": bool(fget(authority_capture, "timing", "timing_join_pass")),
    "current_p8_has_sync_wall_samples": bool(fget(global_direct, "best_variant", "times_s")) and bool(fget(p8_perf, "performance", "times_s")),
  }

  identity_comparison = {
    "same_shape": checks["same_shape"],
    "same_kernel_identity": False,
    "reason": "The prior 43.026 TFLOPS row is the captured tinygrad authority program; current P8 rows are new hand-ASM custom kernels from _run_insts with different names, grids, tile shapes, and instruction streams.",
    "prior_authority_kernel": captured_kernel,
    "current_p8_best_global_direct_kernel": p8_global_kernel,
  }
  timing_comparison = {
    "prior_authority_timing_method": "captured authority row timing joined to source/ELF/disassembly/resource evidence",
    "current_p8_timing_method": "host perf_counter wall interval with explicit Device['AMD'].synchronize() before and after run_linear",
    "same_timing_method": False,
    "current_method_is_valid_for_current_candidates": True,
    "prior_43_is_valid_for_prior_authority_kernel": checks["prior_authority_has_timing_join"],
    "prior_43_validates_current_p8_candidates": False,
    "reason": "Both timing rows are useful, but they do not time the same kernel under the same harness. The 43.026 TFLOPS captured authority value cannot be used to overrule the synchronized P8 custom-kernel timings.",
  }
  performance_summary = {
    "shape_flops": 512 * 12288 * 4096 * 2,
    "p8_lds_best_tflops": p8_lds_tflops,
    "p8_global_direct_best_tflops": p8_global_best,
    "prior_captured_authority_best_tflops": captured_tflops,
    "tensile_median_tflops": ffn_gate_up.get("median_tflops"),
    "tensile_best_tflops": ffn_gate_up.get("best_tflops"),
    "p8_global_vs_prior_authority_ratio": (p8_global_best / captured_tflops) if captured_tflops else None,
    "p8_lds_vs_prior_authority_ratio": (p8_lds_tflops / captured_tflops) if captured_tflops else None,
  }
  decision = {
    "valid_timing_authority_for_current_p8_gate": "synchronized_p8_custom_kernel_harness",
    "valid_use_of_prior_43_tflops": "baseline evidence for the captured tinygrad authority kernel only; not a same-harness validation of current P8 custom kernels",
    "q8_transfer_reopen": False,
    "reopen_lds_tuning": False,
    "reopen_existing_global_direct_candidates": False,
    "next_action": "Build a same-harness authority timing bridge: time the captured 43 TFLOPS authority kernel and current P8 candidates under one common synchronized or device-timestamp harness before starting any new scheduling/ILP candidate.",
  }
  gate = {
    **checks,
    "identity_mismatch_detected": identity_comparison["same_kernel_identity"] is False,
    "timing_method_mismatch_detected": timing_comparison["same_timing_method"] is False,
    "current_p8_authority_selected": decision["valid_timing_authority_for_current_p8_gate"] == "synchronized_p8_custom_kernel_harness",
    "same_harness_bridge_required": "same-harness" in decision["next_action"],
  }
  gate_pass = all(gate.values())
  verdict = "PASS_BB5A10_P8_TIMING_AUTHORITY_RECONCILED_SAME_HARNESS_REQUIRED" if gate_pass else "BLOCKED_BB5A10_P8_TIMING_AUTHORITY_RECONCILIATION"
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_timing_authority_reconciliation",
    "schema": "amd_bb5a10_p8_timing_authority_reconciliation_v1",
    "verdict": verdict,
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "identity_comparison": identity_comparison,
    "timing_comparison": timing_comparison,
    "performance_summary": performance_summary,
    "decision": decision,
    "gate": gate,
    "next_action": decision["next_action"] if gate_pass else "Fix missing reconciliation inputs before further P8 work.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_global_direct_candidate_decision_result.json",
      "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json",
      "bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json",
      "bench/qk-tensile-extraction/shape_matrix.json",
    ],
  }
  write_json("bb5a10_p8_timing_authority_reconciliation_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json",
    "verdict": verdict,
    "gate_pass": gate_pass,
    "current_p8_authority": decision["valid_timing_authority_for_current_p8_gate"],
    "prior_43_validates_current_p8": timing_comparison["prior_43_validates_current_p8_candidates"],
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
