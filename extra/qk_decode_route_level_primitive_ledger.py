#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_route_level_primitive_ledger_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  q8 = load("bench/q8-ffn-artifact-promotion/promotion_result.json", {})
  p8 = load("bench/qk-decode-mmvq-large-project/p8_fused_lifecycle_decision.json", {})
  p7d = load("bench/qk-decode-mmvq-large-project/p7d_one_role_timing.json", {})
  p7e = load("bench/qk-decode-mmvq-large-project/p7e_gateup_amortization.json", {})
  readiness = load("bench/qk-decode-native-tooling/readiness.json", {})
  route = load("bench/qk-decode-primitive-transfer/decode_route_decision_closeout_result.json", {})
  t3 = load("bench/qk-decode-primitive-transfer/decode_dnr4_t3_candidate_grid_result.json", {})

  q8_summary = q8.get("summary", {})
  q8_policy = q8_summary.get("policy_decision", {})
  q8_perf = q8_summary.get("performance", {})
  q8_quality = q8_summary.get("quality", {})
  p8_decision = p8.get("p8d_decision", {})
  wd_rows = p8_decision.get("whole_decode_wd_rows", [])
  min_wd_speedup = min((row.get("speedup", 0.0) for row in wd_rows), default=0.0)
  median_wd_speedup = p8_decision.get("median_wd_speedup", 0.0)

  routes = [
    {
      "route": "current_default_decode",
      "kind": "baseline",
      "ownership": "tinygrad",
      "quality_policy": "authority",
      "fallback": "n/a",
      "default_policy": "keep_default_on",
      "timing_evidence": "banked W==D authority baseline",
      "decision": "KEEP",
      "next": "none",
    },
    {
      "route": "q8_ffn_handwritten_artifact",
      "kind": "route_level_decode_primitive",
      "ownership": "external artifact, locally hashed and guarded",
      "quality_policy": q8_policy.get("quality_threshold"),
      "fallback": q8_policy.get("rollback"),
      "default_policy": "hardened_opt_in_default_off",
      "release_flag": q8_policy.get("release_flag"),
      "supported_model_set": q8_policy.get("supported_model_set"),
      "timing_evidence": {
        "min_wd_speedup": min_wd_speedup,
        "median_wd_speedup": median_wd_speedup,
        "promotion_min_speedup": q8_perf.get("min_speedup"),
        "artifact_speedup_vs_p7e_baseline": (p8.get("p8c_handwritten_prototype") or {}).get("artifact_speedup_vs_p7e_baseline"),
      },
      "quality_evidence": {
        "max_dnll": q8_quality.get("max_dnll"),
        "mean_dnll": q8_quality.get("mean_dnll"),
        "threshold": q8_quality.get("threshold"),
      },
      "decision": "KEEP_HARDENED_OPT_IN_DO_NOT_DEFAULT_ON",
      "next": "use_as_measured_oracle_and_owned-successor-target",
    },
    {
      "route": "imported_llama_q4_mmvq_graph",
      "kind": "source_import_oracle",
      "ownership": "llama source semantics imported into graph route",
      "quality_policy": "contract preserving",
      "fallback": "existing default tinygrad decode",
      "default_policy": "default_off",
      "timing_evidence": {
        "attn_output_speedup": (p7d.get("timing") or {}).get("speedup"),
        "gateup_speedup": (p7e.get("timing") or {}).get("speedup"),
      },
      "decision": "REJECT_AS_SPEED_ROUTE_KEEP_AS_SEMANTIC_ORACLE",
      "next": "do_not_expand_model_wide_for_speed",
    },
    {
      "route": "native_local_mmvq_schedule_edits",
      "kind": "native_schedule_rewrite",
      "ownership": "tinygrad",
      "quality_policy": "contract preserving",
      "fallback": "existing default tinygrad decode",
      "default_policy": "not_promoted",
      "timing_evidence": t3.get("timing_context", {}),
      "decision": "PARK_UNTIL_ATT_OR_NEW_ROUTE_OBJECTIVE",
      "next": "no_more_count_matching",
    },
    {
      "route": "owned_q8_lifecycle_successor",
      "kind": "future_owned_route_level_primitive",
      "ownership": "tinygrad-owned implementation required",
      "quality_policy": "must inherit q8 artifact dNLL gate and fallback policy",
      "fallback": "existing default tinygrad decode",
      "default_policy": "unknown_until_quality_and_coverage",
      "timing_evidence": "target q8 artifact W==D speedup and lifecycle timing",
      "decision": "SCOPE_NEXT_NOT_IMPLEMENTED",
      "next": "build metadata object for q8 producer/cache, gate/up consumers, fallback, model coverage, and timing gates",
    },
  ]

  gates = {
    "route_closeout_passed": route.get("gate_pass") is True,
    "q8_hardened_optin_passed": q8.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN",
    "q8_default_off": q8_policy.get("default_on") is False,
    "q8_quality_passed": q8_quality.get("max_dnll", 1.0) <= q8_quality.get("threshold", 0.0),
    "q8_wd_min_speedup_ge_1_05": min_wd_speedup >= 1.05,
    "imported_q4_rejected_by_timing": (
      p7d.get("verdict") == "NO_LOCAL_TIMING_WIN"
      and p7e.get("verdict") == "NO_GATEUP_TIMING_WIN"
      and (p7d.get("timing") or {}).get("speedup", 1.0) < 1.0
      and (p7e.get("timing") or {}).get("speedup", 1.0) < 1.0
    ),
    "native_local_rewrites_parked": t3.get("verdict") == "BLOCKED_DNR4_T3_NO_MATERIAL_NATIVE_LEVER_UNBLOCK_ATT",
    "native_readiness_no_n2": (readiness.get("start_gate") or {}).get("n2_candidate_count") == 0,
    "owned_successor_scoped": True,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ROUTE_LEVEL_PRIMITIVE_LEDGER",
    "schema": "decode_route_level_primitive_ledger_v1",
    "verdict": "PASS_DECODE_ROUTE_LEVEL_PRIMITIVE_LEDGER_READY" if all(gates.values()) else "BLOCKED_DECODE_ROUTE_LEVEL_PRIMITIVE_LEDGER_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "routes": routes,
    "decision": {
      "promotable_now": "q8_ffn_handwritten_artifact as hardened opt-in only",
      "not_promotable_now": [
        "q8 artifact default-on",
        "imported llama Q4 graph route as speed path",
        "native local MMVQ schedule edits",
      ],
      "next_local_work": "scope owned_q8_lifecycle_successor metadata object and gates",
      "parallel_tooling_work": "unblock ATT decoder library for PC timeline attribution",
    },
    "owned_successor_minimum_contract": [
      "q8 producer/cache lifecycle",
      "gate/up consumer reuse policy",
      "supported model/shape/device coverage",
      "quality dNLL gate and fallback behavior",
      "same-harness W==D timing gate",
      "artifact parity target before native lowering/search",
    ],
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "decision": result["decision"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
