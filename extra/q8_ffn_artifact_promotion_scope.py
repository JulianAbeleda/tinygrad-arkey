#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/q8-ffn-artifact-promotion"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  coverage = read_json("bench/qk-primitive-coverage/rows.json", {})
  baseline = read_json("bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json", {})
  q8_route = read_json("bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json", {})
  nll_base = read_json("bench/q8-ffn-handwritten-oracle/nll_baseline.json", {})
  nll_q8 = read_json("bench/q8-ffn-handwritten-oracle/nll_q8_route.json", {})
  policy = read_json("bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json", {})
  contract = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json", {})
  hardening = read_json("bench/qk-decode-path-split/small_q8_hardening.json", {})
  row = next((r for r in coverage.get("rows", []) if r.get("id") == "decode_q8_artifact_lifecycle"), {})

  baseline_by_ctx = {r.get("ctx"): r for r in baseline.get("rows", [])}
  q8_by_ctx = {r.get("ctx"): r for r in q8_route.get("rows", [])}
  speed_rows = []
  for ctx, b in sorted(baseline_by_ctx.items()):
    q = q8_by_ctx.get(ctx, {})
    if b and q:
      speed_rows.append({
        "ctx": ctx,
        "baseline_tok_s": b.get("tok_s_W"),
        "q8_tok_s": q.get("tok_s_W"),
        "speedup": (q.get("tok_s_W") / b.get("tok_s_W")) if b.get("tok_s_W") and q.get("tok_s_W") else None,
        "host_sync_pct": q.get("host_sync_pct_of_wall"),
      })
  min_speedup = min((r["speedup"] for r in speed_rows if r["speedup"]), default=None)
  dnll = nll_q8.get("nll", 0.0) - nll_base.get("nll", 0.0) if nll_q8 and nll_base else None

  phases = [
    {
      "phase": "Q8P-1",
      "name": "quality_promotion_gate",
      "status": "scoped",
      "current_evidence": {"single_window_tokens": nll_q8.get("tokens"), "single_window_dnll": dnll},
      "minimum_pass": "multi-window or task-quality eval passes dNLL <=0.01 on every accepted window, reports mean/max dNLL, and includes W==D greedy sanity.",
      "why_needed": "Current dNLL +0.002887 over 160 tokens is sufficient for research but too narrow for default promotion.",
      "outputs": [
        "bench/q8-ffn-artifact-promotion/quality_matrix.json",
        "docs/q8-ffn-artifact-promotion-quality-result-20260620.md",
      ],
    },
    {
      "phase": "Q8P-2",
      "name": "default_safety_gate",
      "status": "scoped",
      "current_evidence": {"default_changed": policy.get("default_changed"), "fallback": ((policy.get("requirements") or {}).get("fallback"))},
      "minimum_pass": "route remains flag-gated, fallback returns byte-identical existing decode, unsupported paths fall back, and default-off/default-on behavior is isolated in subprocess tests.",
      "why_needed": "Promotion requires boring failure modes and a kill switch, not just a speed flag.",
      "outputs": [
        "bench/q8-ffn-artifact-promotion/default_safety.json",
        "docs/q8-ffn-artifact-promotion-default-safety-result-20260620.md",
      ],
    },
    {
      "phase": "Q8P-3",
      "name": "coverage_gate",
      "status": "scoped",
      "current_evidence": policy.get("supported", {}),
      "minimum_pass": "all routed tensors/layers are enumerated; route is limited to Qwen3-8B dim=4096 hidden=12288 Q4_K gate/up on gfx1100; no accidental lm_head, attention, Q6, prefill, or unsupported model routing.",
      "why_needed": "The win is role-specific and lossy; accidental wider routing would change model semantics.",
      "outputs": [
        "bench/q8-ffn-artifact-promotion/coverage_matrix.json",
        "docs/q8-ffn-artifact-promotion-coverage-result-20260620.md",
      ],
    },
    {
      "phase": "Q8P-4",
      "name": "performance_gate",
      "status": "scoped",
      "current_evidence": {"rows": speed_rows, "min_speedup": min_speedup},
      "minimum_pass": "clean W==D decode reproduces >=3% speedup at ctx512/1024/4096 with host-sync <=5% and no contaminated per-step host Tensor path.",
      "why_needed": "Default promotion needs sustained model-path speed, not isolated artifact speed.",
      "outputs": [
        "bench/q8-ffn-artifact-promotion/performance_matrix.json",
        "docs/q8-ffn-artifact-promotion-performance-result-20260620.md",
      ],
    },
    {
      "phase": "Q8P-5",
      "name": "artifact_ownership_gate",
      "status": "scoped",
      "current_evidence": {
        "source_module": ((policy.get("requirements") or {}).get("source_module")),
        "runtime": ((policy.get("requirements") or {}).get("runtime")),
        "no_in_process_hip_runtime": ((policy.get("requirements") or {}).get("no_in_process_hip_runtime")),
        "kernarg_size": ((contract.get("launch_contract") or {}).get("kernarg_size")),
      },
      "minimum_pass": "manifest records source, build command, code hashes, kernargs, supported arch, no HIP runtime, fallback, and maintenance owner; or a native tinygrad-owned replacement is selected.",
      "why_needed": "The current win is an external hipcc/LLD artifact route, not a portable tinygrad backend feature.",
      "outputs": [
        "bench/q8-ffn-artifact-promotion/artifact_ownership.json",
        "docs/q8-ffn-artifact-promotion-artifact-ownership-result-20260620.md",
      ],
    },
    {
      "phase": "Q8P-6",
      "name": "model_policy_gate",
      "status": "scoped",
      "current_evidence": {"research_state": row.get("state"), "amdahl_or_potential": row.get("amdahl_or_potential")},
      "minimum_pass": "explicit decision accepts or rejects a lossy ~3-6% default route; names fallback, quality threshold, supported model set, release flag, and rollback criteria.",
      "why_needed": "Even if gates 1-5 pass, defaulting a lossy route for a small speedup is a policy call.",
      "outputs": [
        "bench/q8-ffn-artifact-promotion/model_policy_decision.json",
        "docs/q8-ffn-artifact-promotion-model-policy-result-20260620.md",
      ],
    },
  ]
  gate = {
    "q8_research_row_present": row.get("id") == "decode_q8_artifact_lifecycle",
    "wd_speed_rows_present": len(speed_rows) >= 4,
    "single_window_dnll_passes_research": dnll is not None and dnll <= 0.01,
    "policy_boundary_present": bool(policy),
    "oracle_contract_present": bool(contract),
    "six_phases_scoped": len(phases) == 6,
    "default_currently_unchanged": policy.get("default_changed") is False,
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "schema": "q8_ffn_artifact_promotion_scope_v1",
    "phase": "Q8P-scope",
    "verdict": "PASS_Q8_FFN_ARTIFACT_PROMOTION_SCOPE_READY" if gate_pass else "BLOCKED_Q8_FFN_ARTIFACT_PROMOTION_SCOPE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "current_research_authority": {
      "speed_rows": speed_rows,
      "min_speedup": min_speedup,
      "baseline_nll": nll_base.get("nll"),
      "q8_nll": nll_q8.get("nll"),
      "dnll": dnll,
      "coverage_row": row,
      "hardening": hardening,
    },
    "promotion_phases": phases,
    "input_artifacts": [
      "bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json",
      "bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json",
      "bench/q8-ffn-handwritten-oracle/nll_baseline.json",
      "bench/q8-ffn-handwritten-oracle/nll_q8_route.json",
      "bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "bench/qk-primitive-coverage/rows.json",
      "bench/qk-decode-path-split/small_q8_hardening.json",
    ],
    "gate": gate,
    "next_action": "Run Q8P-1 quality promotion gate first; do not default the q8 artifact route from the existing single-window research evidence.",
  }
  write_json("scope.json", result)
  print(json.dumps({
    "out": "bench/q8-ffn-artifact-promotion/scope.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "phases": len(phases),
    "min_speedup": min_speedup,
    "dnll": dnll,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
