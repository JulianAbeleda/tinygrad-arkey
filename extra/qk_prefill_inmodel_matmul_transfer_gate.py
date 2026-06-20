#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "bench/qk-inmodel-integration-penalty/inmodel_integration_penalty_audit_result.json"
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_inmodel_matmul_transfer_gate_result.json"


def required_component_speedup(share: float, target_full_speedup: float) -> float:
  denom = (1.0 / target_full_speedup) - (1.0 - share)
  return float("inf") if denom <= 0 else share / denom


def main() -> int:
  audit = json.loads(AUDIT.read_text())
  amdahl = audit["amdahl"]
  matmul_share = float(amdahl["matmul_share_of_span"])
  required_for_1p15 = required_component_speedup(matmul_share, 1.15)
  source = audit["source"]
  isolated_ratio = float(source["isolated_ours_gemm_tflops"]) / float(source["prefill_inmodel_effective_tflops"])
  theoretical_if_full_transfer = float(amdahl["if_matmul_gets_isolated_78p6_over_45"])

  # The isolated dependency-free GEMM is not currently a captured PREFILL_V2 graph node.
  # This gate decides whether more microkernel work is justified before graph transfer exists.
  graph_route_exists = False
  material_if_transferred = theoretical_if_full_transfer >= 1.15
  gates = {
    "audit_passed": audit.get("gate_pass") is True,
    "matmul_bucket_dominant": matmul_share >= 0.50,
    "material_if_full_transfer": material_if_transferred,
    "graph_route_exists": graph_route_exists,
  }
  if not gates["audit_passed"]:
    verdict = "BLOCKED_PREFILL_MATMUL_TRANSFER_AUDIT_MISSING"
  elif not gates["material_if_full_transfer"]:
    verdict = "KILL_PREFILL_GEMM_MICROKERNEL_WORK_LOW_AMDAHL"
  elif not gates["graph_route_exists"]:
    verdict = "BLOCKED_PREFILL_MATMUL_TRANSFER_NEEDS_GRAPH_ROUTE"
  else:
    verdict = "PASS_PREFILL_INMODEL_MATMUL_TRANSFER_MATERIAL"

  result = {
    "date": "2026-06-20",
    "phase": "PREFILL_INMODEL_MATMUL_TRANSFER_GATE",
    "schema": "prefill_inmodel_matmul_transfer_gate_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "inputs": {
      "matmul_share": matmul_share,
      "isolated_ours_gemm_tflops": source["isolated_ours_gemm_tflops"],
      "prefill_inmodel_effective_tflops": source["prefill_inmodel_effective_tflops"],
      "isolated_to_inmodel_ratio": round(isolated_ratio, 4),
    },
    "thresholds": {
      "target_full_prefill_speedup": 1.15,
      "required_matmul_speedup_for_1p15": round(required_for_1p15, 4),
      "theoretical_full_speedup_if_78p6_transfers": theoretical_if_full_transfer,
    },
    "gates": gates,
    "decision": {
      "next": "build a graph-captured matmul transfer row only if we choose to pursue prefill; do not start another isolated GEMM microkernel",
      "blocked_by": "no current PREFILL_V2 graph route for the dependency-free 78.6 TFLOPS GEMM",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": verdict, "thresholds": result["thresholds"], "gates": gates, "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
