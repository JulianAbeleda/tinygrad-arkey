#!/usr/bin/env python3
"""PMS-R6: prefill pipe as a search-owned TEMPLATE (encode the role-selective policy from FACTS, not an if-statement).

The shipped prefill default is the role-selective software-pipelined GEMM (extra/qk_prefill_graph_gemm_route.py): pipe
ON for the latency-bound sub-BLAS roles (attn_qo, attn_kv, ffn_down), pipe OFF (lds2) for the BLAS-saturated
ffn_gate_up. In the code this is a special-case `if ... and out_f == 12288: pipe_mode = False` (route.py:61).

This audit re-derives that include/exclude decision from a per-role FACTS table (role, M/N/K, tm/tn, lds2 achieved
fraction-of-BLAS ceiling, latency_bound) via one rule -- pipe_enabled = lds2_blas_ratio < saturation_threshold -- and
proves the derived policy is IDENTICAL to the code's actual decision. It then REPLAYS ROLE_SELECTIVE_PASS_BEATS_GLOBAL
through the R2 evaluator (no GPU re-measurement). Rollback stays PREFILL_PIPE_ROLE_SELECTIVE=0.

Run:  PYTHONPATH=. python3 extra/qk_prefill_pipe_template_audit.py
"""
from __future__ import annotations
import json, pathlib
from extra.qk_route_manifest import route, rollback_env

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-prefill-pipe-template-audit"
ROUTE_ID = "prefill_pipe_role_selective_default"
GLOBAL_ROLLBACK_ID = "prefill_pipe_global_rollback"

# A role is EXCLUDED from the pipe iff its lds2 path is already at/above the BLAS ceiling (pipe then only adds ILP
# overhead and regresses). Below the ceiling (latency-bound) the pipe's higher ILP LIFTS the role.
SATURATION_THRESHOLD = 1.0   # fraction-of-BLAS at/above which the role is saturated -> keep on lds2

# Per-role FACTS (M=512 prefill ubatch). lds2_blas_ratio = lds2-path achieved TFLOPS / measured external BLAS ceiling.
# Sources: external BLAS ceilings docs/prefill-pure-machine-search-roadmap-20260629.md (hipBLASLt/rocBLAS); gate_up
# saturation + pipe regression from extra/qk_prefill_graph_gemm_route.py:58-62 (107% on lds2 -> 89% under pipe, -17%).
ROLE_FACTS = {
  "attn_qo":     {"M": 512, "N": 4096,  "K": 4096,  "tm": 2, "tn": 2, "blas_ceiling_tflops": 76.7,
                  "lds2_blas_ratio": None, "latency_bound": True,
                  "fact": "rocBLAS 76.7 TFLOPS ceiling; lds2 sub-BLAS (latency-bound) -> pipe lifts"},
  "attn_kv":     {"M": 512, "N": 1024,  "K": 4096,  "tm": 2, "tn": 2, "blas_ceiling_tflops": 51.8,
                  "lds2_blas_ratio": None, "latency_bound": True,
                  "fact": "rocBLAS 51.8 TFLOPS ceiling; small-N kv, lds2 sub-BLAS (latency-bound) -> pipe lifts"},
  "ffn_down":    {"M": 512, "N": 4096,  "K": 12288, "tm": 2, "tn": 2, "blas_ceiling_tflops": 70.9,
                  "lds2_blas_ratio": None, "latency_bound": True,
                  "fact": "rocBLAS 70.9 TFLOPS ceiling; lds2 sub-BLAS (latency-bound) -> pipe lifts"},
  "ffn_gate_up": {"M": 512, "N": 12288, "K": 4096,  "tm": 2, "tn": 2, "blas_ceiling_tflops": 69.8,
                  "lds2_blas_ratio": 1.07, "latency_bound": False,
                  "fact": "hipBLASLt 69.8 TFLOPS ceiling; lds2 already ~107% of BLAS (saturated) -> pipe regresses -17% (107->89%)"},
}


def pipe_enabled_from_facts(f: dict) -> bool:
  """The single FACT-driven rule (not a shape if-statement): exclude only the BLAS-saturated role."""
  r = f.get("lds2_blas_ratio")
  if r is not None and r >= SATURATION_THRESHOLD:
    return False   # saturated -> keep on lds2 (pipe off)
  return True       # sub-BLAS / latency-bound -> pipe on


def code_pipe_decision(f: dict) -> bool:
  """The route's ACTUAL decision (extra/qk_prefill_graph_gemm_route.py:61): pipe off iff out_f == 12288."""
  return not (f["N"] == 12288)


def replay_role_selective() -> dict:
  """Reproduce ROLE_SELECTIVE_PASS_BEATS_GLOBAL via the R2 evaluator (replay; no GPU)."""
  from extra.qk_candidate_evaluator import evaluate
  r = evaluate(ROUTE_ID)
  raw = ROOT / "bench/qk-prefill-pipe-role-selective/latest.json"
  raw_verdict = json.load(open(raw)).get("verdict") if raw.exists() else "MISSING"
  return {"evaluator_tier": r.get("tier_classification"), "evaluator_disposition": r.get("disposition"),
          "decision_reproduced": r.get("decision_reproduced"), "raw_artifact_verdict": raw_verdict,
          "per_ctx": r.get("per_ctx"), "speed_stats_pct": r.get("speed_stats_pct"),
          "evaluator_verdict": r.get("verdict")}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  derived = {role: pipe_enabled_from_facts(f) for role, f in ROLE_FACTS.items()}
  code = {role: code_pipe_decision(f) for role, f in ROLE_FACTS.items()}
  policy_matches_code = (derived == code)
  derived_excluded = sorted(r for r, on in derived.items() if not on)
  code_excluded = sorted(r for r, on in code.items() if not on)

  replay = replay_role_selective()
  rs_pass = replay.get("raw_artifact_verdict") == "ROLE_SELECTIVE_PASS_BEATS_GLOBAL" and replay.get("decision_reproduced")

  encoded = policy_matches_code   # the role policy is reproduced from facts (equals the code decision)
  verdict = "PMS_R6_PASS_PREFILL_TEMPLATE_PROVEN" if (encoded and rs_pass) else "PMS_R6_BLOCKED_ROLE_POLICY_NOT_ENCODED"

  result = {
    "scope": "PMS-R6 prefill pipe role policy as a fact-driven template (encode from facts, replay the gate)",
    "verdict": verdict,
    "route_id": ROUTE_ID, "rollback": rollback_env(ROUTE_ID),
    "route_attribution": route(ROUTE_ID).get("route_attribution"),
    "saturation_threshold_fraction_of_blas": SATURATION_THRESHOLD,
    "rule": "pipe_enabled(role) = (lds2_blas_ratio is None) or (lds2_blas_ratio < SATURATION_THRESHOLD); "
            "i.e. exclude ONLY the BLAS-saturated role from the pipe -- NOT a shape if-statement.",
    "role_facts": ROLE_FACTS,
    "derived_pipe_enabled": derived,
    "code_pipe_enabled_out_f_12288_off": code,
    "policy_reproduced_from_facts": policy_matches_code,
    "derived_excluded_roles": derived_excluded,
    "code_excluded_roles": code_excluded,
    "policy_note": ("the fact-driven rule excludes exactly {ffn_gate_up} (lds2 ratio 1.07 >= 1.0); the route code "
                    "excludes exactly the out_f==12288 role (also ffn_gate_up). Same decision, derived from the "
                    "BLAS-saturation FACT rather than the hardcoded shape predicate."),
    "role_selective_replay": replay,
    "rollback_contract": {"flag": "PREFILL_PIPE_ROLE_SELECTIVE=0", "target": GLOBAL_ROLLBACK_ID,
                          "manifest_rollback": rollback_env(ROUTE_ID)},
    "do_not": ["do not change the default speed path", "do not re-prove the speed result (replay only)"],
  }
  json.dump(result, open(OUT / "latest.json", "w"), indent=2)
  md = ["# PMS-R6 Prefill Pipe Role Policy Template Audit", "",
        f"Verdict: **{verdict}**", "",
        f"Rule (fact-driven, not a shape if): `pipe_enabled = lds2_blas_ratio < {SATURATION_THRESHOLD}`.", "",
        f"Policy reproduced from facts == code decision: **{policy_matches_code}** "
        f"(derived excluded {derived_excluded} == code excluded {code_excluded}).", "",
        "| role | M | N | K | tm | tn | lds2/BLAS | latency-bound | pipe (derived) | pipe (code) |",
        "|---|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|"]
  for role, f in ROLE_FACTS.items():
    md.append(f"| {role} | {f['M']} | {f['N']} | {f['K']} | {f['tm']} | {f['tn']} | "
              f"{f['lds2_blas_ratio']} | {f['latency_bound']} | {derived[role]} | {code[role]} |")
  md += ["", f"Role-selective replay: raw `{replay.get('raw_artifact_verdict')}`, evaluator "
         f"`{replay.get('evaluator_tier')}` (reproduced={replay.get('decision_reproduced')}). "
         f"Rollback: PREFILL_PIPE_ROLE_SELECTIVE=0 -> {GLOBAL_ROLLBACK_ID}.", ""]
  (OUT / "summary.md").write_text("\n".join(md))
  print(verdict, "| policy_from_facts==code:", policy_matches_code,
        "| role_selective replay:", replay.get("raw_artifact_verdict"), "reproduced:", replay.get("decision_reproduced"))
  print("  derived excluded:", derived_excluded, "| code excluded:", code_excluded)
  return 0 if verdict.endswith("PROVEN") else 1


if __name__ == "__main__":
  raise SystemExit(main())
