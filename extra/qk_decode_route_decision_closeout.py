#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer"

NORMALIZED = "bench/qk-decode-primitive-transfer/decode_role_contract_normalization_result.json"
READINESS = "bench/qk-decode-native-tooling/readiness.json"
P7D = "bench/qk-decode-mmvq-large-project/p7d_one_role_timing.json"
P7E = "bench/qk-decode-mmvq-large-project/p7e_gateup_amortization.json"
P8 = "bench/qk-decode-mmvq-large-project/p8_fused_lifecycle_decision.json"
Q8_PROMOTION = "bench/q8-ffn-artifact-promotion/promotion_result.json"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  normalized = read_json(NORMALIZED, {})
  readiness = read_json(READINESS, {})
  p7d = read_json(P7D, {})
  p7e = read_json(P7E, {})
  p8 = read_json(P8, {})
  q8 = read_json(Q8_PROMOTION, {})

  q4_attn_speedup = ((p7d.get("timing") or {}).get("speedup"))
  q4_gateup_speedup = ((p7e.get("timing") or {}).get("speedup"))
  p8_decision = p8.get("p8d_decision") or {}
  p8_gates = p8.get("gates") or {}
  q8_policy = ((q8.get("summary") or {}).get("policy_decision") or {})
  readiness_gate = readiness.get("start_gate") or {}

  routes = [
    {
      "route": "current_default_decode",
      "decision": "KEEP_PROMOTED_DEFAULT",
      "reason": "banked W==D decode stack remains the authority baseline",
      "default_on": True,
      "next": "none",
    },
    {
      "route": "imported_llama_q4_mmvq_graph_route",
      "decision": "CLOSED_AS_SPEED_ROUTE",
      "reason": "P7d attn_output and P7e gate/up both run and are stable, but both lose local timing",
      "evidence": {
        "p7d_verdict": p7d.get("verdict"),
        "attn_output_speedup": q4_attn_speedup,
        "p7e_verdict": p7e.get("verdict"),
        "gateup_speedup": q4_gateup_speedup,
        "p8_imported_llama_q4_route": p8_decision.get("imported_llama_q4_route"),
      },
      "default_on": False,
      "next": "do_not_expand_model_wide",
    },
    {
      "route": "fused_q8_mmvq_artifact",
      "decision": "KEEP_HARDENED_OPT_IN",
      "reason": "P8 artifact prototype clears local gate and Q8P promotion gates passed, but route is lossy and externally owned",
      "evidence": {
        "p8_artifact_route": p8_decision.get("fused_q8_mmvq_artifact_route"),
        "artifact_speedup_vs_p7e_baseline": ((p8.get("p8c_handwritten_prototype") or {}).get("artifact_speedup_vs_p7e_baseline")),
        "wd_rows": p8_decision.get("whole_decode_wd_rows"),
        "q8_promotion": q8.get("verdict"),
        "default_on": q8_policy.get("default_on"),
      },
      "default_on": False,
      "next": "use_as_measured_oracle_or_research_flag",
    },
    {
      "route": "native_tinygrad_mmvq_renderer",
      "decision": "PROJECT_LEVEL_BLOCKED",
      "reason": "current UOp/COMGR/ASM native routes miss the fused lifecycle gate and readiness has no >=30us attributed N2 feature",
      "evidence": {
        "p8_native_route": p8_decision.get("native_tinygrad_route"),
        "native_comgr_lifecycle_us": ((p8.get("p8b_current_uop_expressibility") or {}).get("native_comgr_lifecycle_us")),
        "native_amd_dsl_consumer_us": ((p8.get("p8b_current_uop_expressibility") or {}).get("native_amd_dsl_consumer_us")),
        "readiness_verdict": readiness.get("verdict"),
        "max_timing_grade_movement_us": readiness_gate.get("max_timing_grade_movement_us"),
        "required_movement_us": 30,
      },
      "default_on": False,
      "next": "start_only_if_broad_backend_work_is_accepted_or_new_attribution_clears_gate",
    },
  ]

  gates = {
    "normalization_passed": normalized.get("gate_pass") is True,
    "p7d_closed_attn_output": p7d.get("verdict") == "NO_LOCAL_TIMING_WIN" and q4_attn_speedup is not None and q4_attn_speedup < 1.0,
    "p7e_closed_gateup": p7e.get("verdict") == "NO_GATEUP_TIMING_WIN" and q4_gateup_speedup is not None and q4_gateup_speedup < 1.0,
    "p8_decision_complete": p8.get("verdict") == "P8_COMPLETE_ARTIFACT_YES_NATIVE_PROJECT_LEVEL" and all(p8_gates.values()),
    "q8_hardened_optin": q8.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN" and q8_policy.get("default_on") is False,
    "native_readiness_blocked": readiness.get("verdict") == "ROADMAP_ONLY" and readiness_gate.get("n2_candidate_count") == 0,
  }
  gate_pass = all(gates.values())
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ROUTE_DECISION_CLOSEOUT",
    "schema": "decode_route_decision_closeout_v1",
    "verdict": "PASS_DECODE_ROUTE_DECISION_CLOSEOUT" if gate_pass else "BLOCKED_DECODE_ROUTE_DECISION_CLOSEOUT",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "routes": routes,
    "gates": gates,
    "decode_next": {
      "recommended": "native_renderer_project_scope_against_q8_artifact_oracle",
      "why": "imported Q4 graph route is closed as a speed route; q8 artifact is already hardened opt-in; remaining decode parity work is native renderer/scheduler transfer or no-op",
      "do_not_do": ["continue imported Q4 route expansion", "BEAM/search without native primitive lowering", "default-on q8 artifact without policy acceptance"],
      "allowed": ["keep default decode", "use Q8_FFN_HANDWRITTEN=1 as opt-in oracle", "scope broad native MMVQ renderer if funded"],
    },
    "input_artifacts": [NORMALIZED, READINESS, P7D, P7E, P8, Q8_PROMOTION],
  }
  write_json("decode_route_decision_closeout_result.json", result)
  print(json.dumps({
    "out": str(OUT / "decode_route_decision_closeout_result.json"),
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "failed_gates": [k for k, v in gates.items() if not v],
    "recommended_next": result["decode_next"]["recommended"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
