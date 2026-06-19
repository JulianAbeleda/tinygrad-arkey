#!/usr/bin/env python3
"""SDB-1/SDB-2 read-only spec decode bandwidth-amortization model.

Consumes existing acceptance and verify artifacts. Does not run hardware or
route SPEC_DECODE.
"""
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-spec-decode-bandwidth-amortization"


def read_json(rel:str) -> dict[str, Any]:
  return json.loads((ROOT / rel).read_text())


def accepted_from_prefix_rates(rates:list[float], k:int) -> float:
  return round(1.0 + sum(float(x) for x in rates[:k]), 3)


def interp_verify_x(verify_x_by_t:dict[int, float], t:int) -> float:
  if t in verify_x_by_t:
    return verify_x_by_t[t]
  lo = max(x for x in verify_x_by_t if x < t)
  hi = min(x for x in verify_x_by_t if x > t)
  frac = (t - lo) / (hi - lo)
  return round(verify_x_by_t[lo] + frac * (verify_x_by_t[hi] - verify_x_by_t[lo]), 3)


def speedup(accepted:float, draft_cost:float, verify_x:float, runtime_x:float) -> float:
  return round(accepted / (draft_cost + verify_x + runtime_x), 3)


def verify_budget(accepted:float, draft_cost:float, runtime_x:float, target_speedup:float) -> float:
  return round(accepted / target_speedup - draft_cost - runtime_x, 3)


def build_model() -> dict[str, Any]:
  acc = read_json("bench/qk-spec-decode-acceptance/result.json")
  verify = read_json("bench/qk-spec-verify-component-breakdown/result.json")
  prod = read_json("bench/qk-spec-decode-production/baseline.json")

  one_ms = float(verify["jit_verify_ms"]["1"])
  verify_x_by_t = {int(t): round(float(ms) / one_ms, 3) for t, ms in verify["jit_verify_ms"].items()}
  target_tok_s = float(acc["draft_0p6B"]["speed_model"]["target_ms"])
  # target_ms is stored in ms in this nested artifact; use tok/s fields for cost ratio.
  target_tok_s = float(acc["target_decode_tok_s"])
  draft_06_tok_s = float(acc["draft_0p6B"]["draft_decode_tok_s"])
  draft_17_tok_s = float(acc["draft_decode_tok_s"])

  k4_06_rates = [float(x) for x in acc["draft_0p6B"]["K4"]["per_position_accept"]]
  k4_17_rates = [float(x) for x in acc["acceptance"]["K4"]["per_position_accept"]]
  k8_17_rates = [float(x) for x in acc["acceptance"]["K8"]["per_position_accept"]]

  rows = []
  for draft_name, draft_tok_s, rates_by_k in [
    ("Qwen3-0.6B-Q8_0", draft_06_tok_s, {2: k4_06_rates, 3: k4_06_rates, 4: k4_06_rates}),
    ("Qwen3-1.7B-Q8_0", draft_17_tok_s, {2: k4_17_rates, 3: k4_17_rates, 4: k4_17_rates, 8: k8_17_rates}),
  ]:
    for k, rates in rates_by_k.items():
      t = k + 1
      accepted = accepted_from_prefix_rates(rates, k)
      draft_cost = round(k * target_tok_s / draft_tok_s, 3)
      current_v = interp_verify_x(verify_x_by_t, t)
      row = {
        "draft": draft_name,
        "K": k,
        "T_verify": t,
        "accepted_per_pass": accepted,
        "draft_cost_target_passes": draft_cost,
        "current_verify_x_one_pass": current_v,
        "current_speedup_R0": speedup(accepted, draft_cost, current_v, 0.0),
        "current_speedup_R0p2": speedup(accepted, draft_cost, current_v, 0.2),
        "verify_budget_for_1p2x_R0": verify_budget(accepted, draft_cost, 0.0, 1.2),
        "verify_budget_for_1p2x_R0p2": verify_budget(accepted, draft_cost, 0.2, 1.2),
        "verify_budget_for_1p5x_R0": verify_budget(accepted, draft_cost, 0.0, 1.5),
        "verify_budget_for_1p5x_R0p2": verify_budget(accepted, draft_cost, 0.2, 1.5),
      }
      row["meets_1p2_if_verify_1p5_R0p2"] = speedup(accepted, draft_cost, 1.5, 0.2) >= 1.2
      row["meets_1p5_if_verify_1p0_R0p2"] = speedup(accepted, draft_cost, 1.0, 0.2) >= 1.5
      rows.append(row)

  comp = verify["eager_component_us"]["5"]
  comp_total = sum(float(v) for v in comp.values())
  current_t5_ms = float(verify["jit_verify_ms"]["5"])
  target_1p5_ms = one_ms * 1.5
  needed_cut_ms = current_t5_ms - target_1p5_ms
  components = []
  groups = [
    ("q4k_gemm", ["q4k_gemm"], "Q4_K batched weight-read reuse"),
    ("q6k_gemm_lm_head", ["q6k_gemm", "lm_head"], "Q6_K/lm_head batched weight-read reuse"),
    ("attention_reduces", ["attention", "reduce_other"], "short-block causal verify attention + reductions"),
    ("elementwise_norm", ["elementwise_norm"], "norm/RoPE/SwiGLU/residual"),
  ]
  for name, keys, primitive in groups:
    eager_us = sum(float(comp.get(k, 0.0)) for k in keys)
    share = eager_us / comp_total if comp_total else 0.0
    real_ms = share * current_t5_ms
    components.append({
      "component": name,
      "share_T5_directional": round(share, 3),
      "real_ms_at_T5_directional": round(real_ms, 3),
      "candidate_primitive": primitive,
      "single_component_T_independent_would_meet_1p5x": (current_t5_ms - real_ms) <= target_1p5_ms,
      "status": "not_single_sufficient" if name != "elementwise_norm" else "too_small",
    })

  audit = {
    "current_T5_ms": round(current_t5_ms, 3),
    "one_pass_ms": round(one_ms, 3),
    "target_T5_ms_for_1p5x_one_pass": round(target_1p5_ms, 3),
    "needed_cut_ms": round(needed_cut_ms, 3),
    "needed_cut_fraction": round(needed_cut_ms / current_t5_ms, 3),
    "components": components,
    "shared_primitive_found": False,
    "classification": "project_level_batched_forward",
    "reason": "Q4_K, Q6_K/lm_head, and attention/reduces are co-dominant; no single existing primitive can cut the required ~68% of verify time.",
  }

  return {
    "schema": "spec_decode_bandwidth_amortization_sdb1_sdb2_v1",
    "inputs": {
      "acceptance": "bench/qk-spec-decode-acceptance/result.json",
      "verify_breakdown": "bench/qk-spec-verify-component-breakdown/result.json",
      "production_baseline": "bench/qk-spec-decode-production/baseline.json",
      "target_tok_s": target_tok_s,
      "production_spec_speedup": prod["spec_production"],
      "verify_x_by_T": verify_x_by_t,
    },
    "sdb1_speed_model": rows,
    "sdb2_verify_design_audit": audit,
    "verdict": {
      "SDB_1": "PASS_MODEL_BUILT",
      "SDB_2": "NO_BOUNDED_SHARED_PRIMITIVE",
      "next": "Do not build SDB-3 minimal proof unless a project-level T-cheap batched-forward route is funded.",
    },
  }


def write_summary(model:dict[str, Any]) -> None:
  lines = [
    "# Spec decode bandwidth amortization SDB-1/SDB-2 - 2026-06-19",
    "",
    "Read-only analysis. No hardware execution or SPEC_DECODE route.",
    "",
    "## SDB-1",
    "",
    "| draft | K | accepted/pass | draft cost | current verify | current speedup R=0 | verify budget for 1.2x (R=0.2) |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for r in model["sdb1_speed_model"]:
    lines.append(f"| {r['draft']} | {r['K']} | {r['accepted_per_pass']} | {r['draft_cost_target_passes']} | "
                 f"{r['current_verify_x_one_pass']} | {r['current_speedup_R0']} | {r['verify_budget_for_1p2x_R0p2']} |")
  audit = model["sdb2_verify_design_audit"]
  lines += [
    "",
    "## SDB-2",
    "",
    f"- current T=5 verify: `{audit['current_T5_ms']}ms`",
    f"- one pass: `{audit['one_pass_ms']}ms`",
    f"- target for `<=1.5x`: `{audit['target_T5_ms_for_1p5x_one_pass']}ms`",
    f"- required cut: `{audit['needed_cut_ms']}ms` (`{audit['needed_cut_fraction']}` of verify)",
    f"- classification: `{audit['classification']}`",
    "",
    "| component | share | candidate primitive | single sufficient? |",
    "|---|---:|---|:--:|",
  ]
  for c in audit["components"]:
    lines.append(f"| {c['component']} | {c['share_T5_directional']} | {c['candidate_primitive']} | "
                 f"{c['single_component_T_independent_would_meet_1p5x']} |")
  lines += [
    "",
    "## Verdict",
    "",
    "- Current spec remains non-viable because verify is too expensive.",
    "- The PMU bandwidth framing is correct, but the missing target verify is project-level batched-forward work.",
    "- SDB-3 should not start as a bounded kernel proof unless a credible T-cheap full-verify route is introduced.",
    "",
  ]
  (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  model = build_model()
  (OUT / "model.json").write_text(json.dumps(model, indent=2) + "\n")
  write_summary(model)
  print(json.dumps(model["verdict"], indent=2))


if __name__ == "__main__":
  main()
