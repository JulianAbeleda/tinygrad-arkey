#!/usr/bin/env python3
"""TBF-0..TBF-2 read-only spec T-cheap batched-forward project audit.

Builds the IR contract and component ceiling ledger from existing artifacts.
Does not run hardware, route SPEC_DECODE, or modify model behavior.
"""
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-spec-tcheap-forward"


def read_json(rel:str) -> dict[str, Any]:
  return json.loads((ROOT / rel).read_text())


def ratio(num:float, den:float) -> float:
  return round(num / den, 3) if den else 0.0


def build_ir_contract() -> dict[str, Any]:
  return {
    "schema": "spec_tcheap_forward_ir_contract_v1",
    "phase": "TBF-1",
    "status": "PASS_CONTRACT_DEFINED",
    "route": "decode-only SPEC_DECODE research route; default off",
    "legal_K": [2, 3, 4],
    "legal_T": [3, 4, 5],
    "recommended_first_shape": {"K": 4, "T": 5, "draft": "Qwen3-0.6B-Q8_0"},
    "inputs": {
      "tokens": {
        "shape": "[1, T]",
        "semantics": "tokens[0] is the previous accepted token; tokens[1:T] are K draft proposals",
      },
      "base_pos": {
        "semantics": "position of tokens[0], equal to accepted_context_length - 1",
        "reason": "target prediction at output index i verifies draft token i for i<K",
      },
      "target_kv": "existing target KV prefix through base_pos; route must not corrupt committed prefix",
      "draft_kv": "draft KV prefix through accepted context; full-accept cache-hole case must be handled",
    },
    "outputs": {
      "target_token_predictions": {
        "shape": "[T]",
        "semantics": "argmax predictions for positions base_pos..base_pos+K",
      },
      "accept_prefix_len": {
        "range": "[0, K]",
        "semantics": "longest prefix where draft_tokens[i] == target_token_predictions[i]",
      },
      "emitted_tokens": "draft prefix plus target correction/bonus token at accept_prefix_len",
      "commit_metadata": "target/draft KV commit range and rollback/ignore range",
    },
    "kv_protocol": {
      "minimum_safe_design": "temporary target KV overlay for proposed block, then commit accepted prefix",
      "allowed_optimization": "idempotent rewrite of previous accepted token at base_pos if value-identical",
      "required_cases": ["zero_accept", "partial_accept", "full_accept"],
      "fallback": "if temp/rollback unavailable, fall back to target-only decode",
    },
    "shape_guards": {
      "T_must_be_concrete": True,
      "batch": 1,
      "decode_only": True,
      "normal_prefill_unchanged": True,
      "normal_decode_unchanged": True,
    },
    "quality_gate": "greedy byte-exact versus target-only decode",
    "speed_gate": "target verify <=1.3-1.5x one T==1 pass before end-to-end SPEC_DECODE route",
  }


def build_component_audit() -> dict[str, Any]:
  verify = read_json("bench/qk-spec-verify-component-breakdown/result.json")
  sdb = read_json("bench/qk-spec-decode-bandwidth-amortization/model.json")
  comp = {int(t): v for t, v in verify["eager_component_us"].items()}
  t1, t5 = comp[1], comp[5]
  groups = [
    ("q4k_gemm", ["q4k_gemm"], "Q4_K short-block weight reuse"),
    ("q6k_lm_head", ["q6k_gemm", "lm_head"], "Q6_K/lm_head short-block weight reuse"),
    ("attention_reduces", ["attention", "reduce_other"], "short-block causal attention + reductions"),
    ("elementwise_norm", ["elementwise_norm"], "elementwise/norm path"),
    ("linears_group", ["q4k_gemm", "q6k_gemm", "lm_head"], "all quantized linears together"),
  ]
  rows = []
  for name, keys, primitive in groups:
    us1 = sum(float(t1.get(k, 0.0)) for k in keys)
    us5 = sum(float(t5.get(k, 0.0)) for k in keys)
    r = ratio(us5, us1)
    rows.append({
      "component": name,
      "candidate_primitive": primitive,
      "T1_eager_us": round(us1, 3),
      "T5_eager_us": round(us5, 3),
      "T5_over_T1": r,
      "target_gate": "<=1.5x T1-equivalent for T=5",
      "passes_gate_now": r <= 1.5,
      "status": "FAIL_CURRENT_BASELINE" if r > 1.5 else "PASS_CURRENT_BASELINE",
    })

  return {
    "schema": "spec_tcheap_forward_component_audit_v1",
    "phase": "TBF-2",
    "status": "FAIL_CURRENT_BASELINE_NO_COMPONENT_CANDIDATE",
    "inputs": {
      "verify_breakdown": "bench/qk-spec-verify-component-breakdown/result.json",
      "sdb_model": "bench/qk-spec-decode-bandwidth-amortization/model.json",
    },
    "verify_gate": {
      "current_T5_x_one_pass": sdb["sdb2_verify_design_audit"]["current_T5_ms"] / sdb["sdb2_verify_design_audit"]["one_pass_ms"],
      "required_T5_x_one_pass": "1.3-1.5",
      "required_cut_fraction": sdb["sdb2_verify_design_audit"]["needed_cut_fraction"],
    },
    "rows": rows,
    "component_gate_passes": all(r["passes_gate_now"] for r in rows if r["component"] in {"q4k_gemm", "q6k_lm_head", "attention_reduces"}),
    "verdict": "TBF-2 does not earn TBF-3: current Q4_K, Q6_K/lm_head, and attention/reduces all scale far above the <=1.5x gate.",
  }


def build_result() -> dict[str, Any]:
  ir = build_ir_contract()
  audit = build_component_audit()
  return {
    "schema": "spec_tcheap_forward_tbf0_tbf2_result_v1",
    "scope": "docs/spec-decode-tcheap-batched-forward-project-scope-20260619.md",
    "TBF_0": {
      "status": "PASS_SCOPE_ACCEPTED_FOR_AUDIT_ONLY",
      "note": "User requested the phase; no SPEC_DECODE route or implementation is created.",
    },
    "TBF_1": ir,
    "TBF_2": audit,
    "final_verdict": {
      "status": "STOP_BEFORE_TBF_3",
      "reason": "IR contract is defined, but component ceilings do not pass; a concrete component candidate is required before any linears/attention implementation.",
      "next_allowed_work": "Bring a proposed component route for either grouped short-block linears or short-block attention, then rerun TBF-2 against it.",
    },
  }


def write_summary(result:dict[str, Any]) -> None:
  audit = result["TBF_2"]
  lines = [
    "# Spec T-cheap batched-forward TBF-0..2 result - 2026-06-19",
    "",
    "Read-only decode project audit. No SPEC_DECODE route, no prefill changes.",
    "",
    "## Verdict",
    "",
    f"- TBF-0: `{result['TBF_0']['status']}`",
    f"- TBF-1: `{result['TBF_1']['status']}`",
    f"- TBF-2: `{audit['status']}`",
    f"- Final: `{result['final_verdict']['status']}`",
    "",
    "## Component Gates",
    "",
    "| component | T5/T1 | gate | status |",
    "|---|---:|---|---|",
  ]
  for row in audit["rows"]:
    lines.append(f"| {row['component']} | {row['T5_over_T1']} | {row['target_gate']} | {row['status']} |")
  lines += [
    "",
    "## Next Allowed Work",
    "",
    result["final_verdict"]["next_allowed_work"],
    "",
  ]
  (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  result = build_result()
  (OUT / "ir_contract.json").write_text(json.dumps(result["TBF_1"], indent=2) + "\n")
  (OUT / "component_audit.json").write_text(json.dumps(result["TBF_2"], indent=2) + "\n")
  (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
  write_summary(result)
  print(json.dumps(result["final_verdict"], indent=2))


if __name__ == "__main__":
  main()
