#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_route_transfer_scope_result.json"


def main() -> int:
  result = {
    "date": "2026-06-20",
    "phase": "PREFILL_GRAPH_ROUTE_TRANSFER_SCOPE",
    "schema": "prefill_graph_route_transfer_scope_v1",
    "verdict": "PASS_PREFILL_GRAPH_ROUTE_TRANSFER_SCOPE_READY",
    "gate_pass": True,
    "default_behavior_changed": False,
    "performance_claim": False,
    "target": {
      "question": "Can the dependency-free GEMM become a captured PREFILL_V2 graph node?",
      "shape": {"M_tokens": 512, "N_out": 12288, "K_in": 4096},
      "layout": "A[T,K] x W[out,K]^T -> C[T,out]; W is already PREFILL_V2 natural fp16 [out,in]",
      "kernel": "extra.gemm.rdna3_wmma_matmul.build_gemm_lds2(BK=32,PAD=16,PLRA=1)",
    },
    "phases": [
      "graph-node feasibility at real gate/up shape",
      "one-role correctness vs PREFILL_V2 linear",
      "one-role graph timing",
      "full-bucket projection",
      "full PREFILL_V2 measurement",
    ],
    "next_executable_probe": "extra/qk_prefill_graph_node_feasibility_probe.py",
    "kill_conditions": [
      "if the node is not captured in HCQGraph, graph-route transfer is blocked on runtime integration",
      "if the node captures but cannot use PREFILL_V2 layout without transpose/copy, block on layout route",
      "if one-role in-graph timing fails to project >=1.15x full prefill, stop prefill GEMM transfer",
    ],
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result["verdict"], "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
