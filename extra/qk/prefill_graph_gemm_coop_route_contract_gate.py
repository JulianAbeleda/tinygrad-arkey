#!/usr/bin/env python3
"""Route-bound contract gate for cooperative fp16 WMMA staging.

The cooperative B-tile probe proves the lane map in a custom generated kernel.
This gate keeps that separate from the actual recovery criterion: the scheduler
must bind the cooperative mapping into the warmstart TC route and show a medium
GEMM win. Until then, the custom probe is only substrate evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "prefill-graph-gemm-coop-route-contract-gate.v1"
COOP_ARTIFACT = Path("bench/prefill-graph-gemm-coop-partition/latest.json")
MEDIUM_ARTIFACT = Path("bench/prefill-graph-gemm-medium-stage/latest.json")
ARTIFACT_DIR = Path("bench/prefill-graph-gemm-coop-route-contract")


def _read_json(path: Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except Exception as e:
    return {"missing_or_invalid": str(e), "path": str(path)}


def build_report(*, artifact: bool = True) -> dict[str, Any]:
  coop = _read_json(COOP_ARTIFACT)
  medium = _read_json(MEDIUM_ARTIFACT)
  coop_probe_pass = coop.get("verdict") == "PREFILL_GRAPH_GEMM_COOP_PARTITION_PROBE_PASS"
  medium_cases = medium.get("cases", {}) if isinstance(medium.get("cases"), dict) else {}
  baseline = medium_cases.get("baseline_table_local", {}) if isinstance(medium_cases.get("baseline_table_local"), dict) else {}
  post_tile_b = medium_cases.get("post_tile_b_stage", {}) if isinstance(medium_cases.get("post_tile_b_stage"), dict) else {}
  medium_has_b_tile = medium.get("evidence", {}).get("post_tile_b_stage_ok") is True
  medium_b_tile_beats = bool(post_tile_b.get("status") == "ok" and baseline.get("status") == "ok" and
                             float(post_tile_b.get("tflops", 0.0)) > float(baseline.get("tflops", 0.0)) * 1.05)

  route_bound_coop_case = medium_cases.get("post_coop_b_partition_stage", {})
  route_bound_coop_defined = isinstance(route_bound_coop_case, dict) and bool(route_bound_coop_case)
  route_bound_coop_executes = bool(route_bound_coop_defined and route_bound_coop_case.get("status") == "ok")
  route_bound_coop_beats = bool(route_bound_coop_executes and baseline.get("status") == "ok" and
                                float(route_bound_coop_case.get("tflops", 0.0)) > float(baseline.get("tflops", 0.0)) * 1.05)
  passed = bool(coop_probe_pass and route_bound_coop_beats)

  report = {
    "schema": SCHEMA,
    "route_id": "prefill_v2_scheduler_matmul_default",
    "target": "target_1_8b_fp16_graph_gemm_cooperative_partition_route_bound",
    "verdict": "PREFILL_GRAPH_GEMM_COOP_ROUTE_CONTRACT_PASS" if passed
      else "PREFILL_GRAPH_GEMM_COOP_ROUTE_CONTRACT_BLOCKED",
    "required_evidence": {
      "custom_coop_partition_probe_pass": coop_probe_pass,
      "medium_gate_has_b_tile_operand_stage": medium_has_b_tile,
      "medium_b_tile_operand_stage_beats_baseline": medium_b_tile_beats,
      "medium_gate_defines_route_bound_coop_partition_case": route_bound_coop_defined,
      "medium_gate_route_bound_coop_partition_executes": route_bound_coop_executes,
      "route_bound_coop_partition_beats_baseline": route_bound_coop_beats,
    },
    "artifacts": {
      "coop_partition": str(COOP_ARTIFACT),
      "medium_stage": str(MEDIUM_ARTIFACT),
      "baseline_tflops": baseline.get("tflops"),
      "post_tile_b_tflops": post_tile_b.get("tflops"),
      "route_bound_coop_status": route_bound_coop_case.get("status") if isinstance(route_bound_coop_case, dict) else None,
      "route_bound_coop_tflops": route_bound_coop_case.get("tflops") if isinstance(route_bound_coop_case, dict) else None,
    },
    "remaining_blocker": None if passed else (
      "custom cooperative B-tile partition is proven, but the warmstart TC route-bound cooperative case "
      + ("executes without beating baseline" if route_bound_coop_executes else "does not execute")
    ),
  }
  if artifact:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "latest.json").write_text(json.dumps(report, indent=2))
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--no-artifact", action="store_true")
  args = ap.parse_args(argv)
  report = build_report(artifact=not args.no_artifact)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
