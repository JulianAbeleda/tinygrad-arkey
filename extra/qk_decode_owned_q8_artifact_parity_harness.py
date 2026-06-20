#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_artifact_parity_harness_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  q8_promotion = load("bench/q8-ffn-artifact-promotion/promotion_result.json", {})
  q8_quality = load("bench/q8-ffn-artifact-promotion/quality_matrix.json", {})
  q8_perf = load("bench/q8-ffn-artifact-promotion/performance_matrix.json", {})
  p8 = load("bench/qk-decode-mmvq-large-project/p8_fused_lifecycle_decision.json", {})
  successor = load("bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_object_result.json", {})

  promotion_summary = q8_promotion.get("summary", {})
  policy = promotion_summary.get("policy_decision", {})
  quality_summary = promotion_summary.get("quality", {})
  perf_summary = promotion_summary.get("performance", {})
  p8_artifact = p8.get("p8c_handwritten_prototype", {})
  p8_decision = p8.get("p8d_decision", {})
  wd_rows = p8_decision.get("whole_decode_wd_rows", [])
  object_row = successor.get("object", {})
  parity = object_row.get("parity", {})

  baseline_rows = [
    {
      "route": "baseline_default_decode",
      "status": "authority_baseline",
      "ownership": "tinygrad",
      "default_on": True,
      "fallback": "n/a",
      "quality_policy": "existing behavior",
      "wd_rows": [
        {"ctx": row.get("ctx"), "tok_s": row.get("baseline_tok_s"), "relative_to_baseline": 1.0}
        for row in wd_rows
      ],
      "promotion_state": "keep",
    },
    {
      "route": "q8_ffn_handwritten_artifact",
      "status": "measured_hardened_opt_in",
      "ownership": "external_artifact_guarded_by_tinygrad",
      "default_on": False,
      "fallback": policy.get("rollback"),
      "quality_policy": policy.get("quality_threshold"),
      "wd_rows": [
        {"ctx": row.get("ctx"), "tok_s": row.get("q8_route_tok_s"), "speedup": row.get("speedup")}
        for row in wd_rows
      ],
      "quality": {
        "max_dnll": quality_summary.get("max_dnll"),
        "mean_dnll": quality_summary.get("mean_dnll"),
        "threshold": quality_summary.get("threshold"),
      },
      "lifecycle_us": p8_artifact.get("hipcc_lld_artifact_lifecycle_us"),
      "promotion_state": "hardened_opt_in_only",
    },
    {
      "route": "owned_q8_lifecycle_successor",
      "status": "target_row_unimplemented",
      "ownership": "tinygrad_owned_required",
      "default_on": False,
      "fallback": (object_row.get("policy") or {}).get("fallback"),
      "quality_policy": "must match q8 artifact quality gate",
      "wd_rows": [
        {"ctx": row.get("ctx"), "required_tok_s": row.get("q8_route_tok_s"), "required_speedup": row.get("speedup")}
        for row in wd_rows
      ],
      "quality": {
        "required_max_dnll_lte": parity.get("quality_threshold_dnll"),
        "artifact_max_dnll": parity.get("quality_max_dnll"),
      },
      "lifecycle_us_required_lte": parity.get("artifact_lifecycle_us"),
      "promotion_state": "blocked_until_owned_candidate_measured",
    },
  ]

  gates = {
    "successor_object_ready": successor.get("gate_pass") is True,
    "q8_promotion_passed": q8_promotion.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN",
    "quality_matrix_passed": q8_quality.get("gate_pass") is True,
    "performance_matrix_passed": q8_perf.get("gate_pass") is True,
    "artifact_default_off": policy.get("default_on") is False,
    "wd_rows_available": len(wd_rows) >= 4,
    "artifact_min_speedup_matches_target": perf_summary.get("min_speedup") == parity.get("wd_min_speedup"),
    "artifact_lifecycle_target_available": parity.get("artifact_lifecycle_us") == p8_artifact.get("hipcc_lld_artifact_lifecycle_us"),
    "owned_row_marked_unimplemented": baseline_rows[2]["status"] == "target_row_unimplemented",
    "no_default_change": True,
  }

  pass_gate = all(gates.values())
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_ARTIFACT_PARITY_HARNESS",
    "schema": "decode_owned_q8_artifact_parity_harness_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_ARTIFACT_PARITY_HARNESS_READY" if pass_gate else "BLOCKED_DECODE_OWNED_Q8_ARTIFACT_PARITY_HARNESS_INCOMPLETE",
    "gate_pass": pass_gate,
    "default_behavior_changed": False,
    "performance_claim": False,
    "parity_rows": baseline_rows,
    "required_owned_successor_evidence": [
      "owned producer/cache implementation row",
      "owned ffn_gate/ffn_up packed q4/q8 consumer row",
      "lifecycle_us <= artifact lifecycle target",
      "W==D speedup >= q8 artifact min speedup at ctx 512/1024/4096",
      "multi-window dNLL <= 0.01",
      "fallback and coverage policy equal or stricter than q8 artifact",
    ],
    "decision": {
      "artifact_status": "measured_hardened_opt_in_default_off",
      "successor_status": "object_and_parity_harness_ready_but_implementation_missing",
      "next": "build owned producer/cache candidate or owned consumer candidate; search remains blocked until one is lowerable",
    },
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
  return 0 if pass_gate else 1


if __name__ == "__main__":
  raise SystemExit(main())
