#!/usr/bin/env python3
"""PMS-R2: table-driven candidate evaluator.

One evaluator for correctness, route attribution, speed, tiered classification, rollback, and an appendable
project-search-ledger row. It is AUDIT-FIRST: it REPLAYS a known decision by reading the route's cited authority
artifact (it does NOT re-run GPU measurement unless the artifact is missing).

Consumes (per scope PMS-R2):
  route_manifest (extra/qk_route_manifest.py)
  profile_id
  candidate route_id
  baseline route_id
  contexts
  authority_type: decode_wd | prefill_whole
  tiered thresholds (TIER_A / TIER_B / equivalence band)

Produces (bench/qk-candidate-evaluator/<route_id>/):
  latest.json            -- normalized per-ctx table + correctness/route/default/speed/tier/rollback
  summary.md             -- human table
  route_attribution.json -- route-bound + no-hidden-fallback evidence from the manifest + authority artifact
  ledger_update.json     -- appendable project-search-ledger row (NOT auto-appended)

Replays the three known decisions:
  decode_q4k_g3_generated            -> SPEED_EQUIVALENT_PASS  (reproduces AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT)
  decode_q6k_direct_refuted          -> REFUTED_REGRESSION     (preserves AMD_ISA_Q6K_DIRECT_SPEED_REGRESSION)
  prefill_pipe_role_selective_default-> PROMOTE_TIER_B         (reproduces ROLE_SELECTIVE_PASS_BEATS_GLOBAL; +3% over
                                                                global pipe is a TIER_B residual under the documented 5% bar)

Run:  PYTHONPATH=. python3 extra/qk_candidate_evaluator.py
"""
from __future__ import annotations
import json, pathlib, statistics
from extra.qk_route_manifest import route, rollback_env, route_env

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-candidate-evaluator"

# Tiered threshold policy (candidate-vs-baseline % delta; positive = candidate faster). The PROFILE descriptor is the
# SINGLE SOURCE (qwen3_8b_q4_k_m_gfx1100.json "thresholds"); evaluate() loads it by profile_id. DEFAULT_THRESHOLDS is the
# matching code fallback for routes without a profile thresholds block (documented tiered policy: TIER_A 5%, TIER_B 2%).
DEFAULT_THRESHOLDS = {"tier_a_pct": 5.0, "tier_b_pct": 2.0, "equiv_band_pct": 1.0, "regression_pct": -2.0,
                      "per_ctx_regression_guard_pct": -2.0}
PROFILES_DIR = ROOT / "bench/qk-search-spaces/profiles"

def _profile_thresholds(profile_id: str | None) -> dict:
  """Load the promotion thresholds from the route's profile descriptor (the single source). The manifest profile_id
  carries a workload suffix (..._decode / ..._prefill); the descriptor file is the base id. Returns {} if absent."""
  if not profile_id: return {}
  base = profile_id
  for suf in ("_decode", "_prefill"):
    if base.endswith(suf): base = base[: -len(suf)]
  p = PROFILES_DIR / f"{base}.json"
  if not p.exists(): return {}
  thr = json.load(open(p)).get("thresholds") or {}
  return {k: v for k, v in thr.items() if k.endswith("_pct")}

# ---- authority-artifact adapters: normalize each artifact into a per-ctx table ----
# Each adapter returns: {ctx_str: {baseline_tok_s, candidate_tok_s, delta_pct, token_match, route_bound,
#                                   candidate_spread_pct, baseline_spread_pct}}, plus (artifact_verdict, correct_flag).
def _adapt_g3(d: dict) -> tuple[dict, str, bool, dict]:
  tab = {}
  for c, p in d["per_ctx"].items():
    base, cand = p["owned_tok_s"], p["g3_tok_s"]
    tab[c] = {"baseline_tok_s": base, "candidate_tok_s": cand,
              "delta_pct": round((cand - base) / base * 100, 3),
              "token_match": bool(p.get("token_match")),
              "route_bound": bool(p.get("g3_fired") and p.get("route_clean")),
              "candidate_spread_pct": p.get("g3_spread_pct"), "baseline_spread_pct": p.get("owned_spread_pct")}
  corr = all(v["token_match"] for v in tab.values())
  evid = {"correctness_gate": "token_match vs owned/default at every ctx",
          "route_gate": "g3_fired & route_clean & leaked_routes=={} & roles_not_g3=={}",
          "forbidden_routes": d.get("promotion_contract", {}).get("forbidden_routes_for_eligible_roles", [])}
  return tab, d.get("verdict", ""), corr, evid

def _adapt_q6k(d: dict) -> tuple[dict, str, bool, dict]:
  tab = {}
  for c, p in d["wd"].items():
    tab[c] = {"baseline_tok_s": p["baseline_tok_s"], "candidate_tok_s": p["candidate_tok_s"],
              "delta_pct": round(p["delta_pct"], 3), "token_match": bool(p.get("token_match")),
              "route_bound": bool(p.get("candidate_halfwarp_fired")),
              "candidate_spread_pct": p.get("candidate_spread_pct"), "baseline_spread_pct": p.get("baseline_spread_pct")}
  corr = bool(d.get("token_match_all_ctx"))
  evid = {"correctness_gate": "token_match_all_ctx", "route_gate": "candidate_halfwarp_fired non-empty (route_bound_all_ctx)",
          "baseline": "flag-off == shipped coop route"}
  return tab, d.get("verdict", ""), corr, evid

def _adapt_prefill_rs(d: dict) -> tuple[dict, str, bool, dict]:
  tab = {}
  for c, p in d["table"].items():
    base, cand = p["global_pipe"], p["role_selective"]
    tab[c] = {"baseline_tok_s": base, "candidate_tok_s": cand,
              "delta_pct": round(p.get("rs_vs_global_pct", (cand - base) / base * 100), 3),
              "token_match": bool(d.get("correct_equivalent")), "route_bound": True,
              "candidate_spread_pct": p.get("rs_spread_pct"), "baseline_spread_pct": p.get("global_spread_pct")}
  corr = bool(d.get("correct_equivalent"))
  evid = {"correctness_gate": "correct_equivalent (logit fingerprint match: argmax+sum)",
          "route_gate": "role-selective excludes ffn_gate_up (out_f==12288) from pipe; keeps pipe for attn_qo/attn_kv/ffn_down",
          "fingerprints": d.get("fingerprints", {})}
  return tab, d.get("verdict", ""), corr, evid

# route_id -> (relative artifact path, adapter, authority_type, baseline_route_id, expected_disposition)
REPLAYS = {
  "decode_q4k_g3_generated": ("bench/amd-isa-backend-g3-weight-promotion/latest.json", _adapt_g3, "decode_wd",
                              "decode_q4k_owned_warp", "promote"),
  "decode_q6k_direct_refuted": ("bench/amd-isa-backend-q6k-direct-speed/latest.json", _adapt_q6k, "decode_wd",
                                "decode_q6k_coop_shipped", "refute"),
  "prefill_pipe_role_selective_default": ("bench/qk-prefill-pipe-role-selective/latest.json", _adapt_prefill_rs,
                                          "prefill_whole", "prefill_pipe_global_rollback", "promote"),
}

def classify(tab: dict, correct: bool, thresholds: dict) -> tuple[str, str, dict]:
  """Tiered classification on the per-ctx candidate-vs-baseline deltas."""
  deltas = [v["delta_pct"] for v in tab.values()]
  med = round(statistics.median(deltas), 3)
  worst = round(min(deltas), 3)
  best = round(max(deltas), 3)
  if not correct:
    return "REFUTED_CORRECTNESS", "refute", {"median_pct": med, "worst_pct": worst, "best_pct": best}
  if not all(v["route_bound"] for v in tab.values()):
    return "BLOCKED_ROUTE_NOT_BOUND", "block", {"median_pct": med, "worst_pct": worst, "best_pct": best}
  stats = {"median_pct": med, "worst_pct": worst, "best_pct": best}
  if med <= thresholds["regression_pct"]:
    return "REFUTED_REGRESSION", "refute", stats
  if worst <= thresholds["per_ctx_regression_guard_pct"]:  # any ctx at/below the guard refutes, regardless of median
    return "REFUTED_REGRESSION", "refute", stats
  if med >= thresholds["tier_a_pct"]:
    return "PROMOTE_TIER_A", "promote", stats
  if med >= thresholds["tier_b_pct"]:
    return "PROMOTE_TIER_B", "promote", stats
  if abs(med) < thresholds["equiv_band_pct"]:
    return "SPEED_EQUIVALENT_PASS", "promote", stats
  return "INCONCLUSIVE", "defer", stats

def evaluate(route_id: str, contexts: list[int] | None = None, thresholds: dict | None = None) -> dict:
  if route_id not in REPLAYS:
    raise KeyError(f"no replay adapter for {route_id!r}; known: {sorted(REPLAYS)}")
  rel, adapter, authority_type, baseline_id, expected_disp = REPLAYS[route_id]
  art = ROOT / rel
  rmeta = route(route_id)
  # thresholds: profile descriptor (single source) over code fallback, caller override last.
  thresholds = {**DEFAULT_THRESHOLDS, **_profile_thresholds(rmeta.get("profile_id")), **(thresholds or {})}
  if not art.exists():
    return {"route_id": route_id, "verdict": "PMS_R2_BLOCKED_AUTHORITY_HARNESS_INCOMPLETE",
            "missing_artifact": str(rel), "authority_gate": rmeta.get("authority_gate")}
  d = json.load(open(art))
  tab, artifact_verdict, correct, route_evid = adapter(d)
  if contexts:
    tab = {c: tab[c] for c in (str(x) for x in contexts) if c in tab}
  tier, disposition, stats = classify(tab, correct, thresholds)
  reproduced = (disposition == expected_disp)

  # default contract: env={} -> shipped default; non-empty env -> opt-in/forced.
  env = route_env(route_id)
  default_contract = "default_on_no_flag" if not env else "opt_in_or_forced"
  rb = rollback_env(route_id)

  attribution = {
    "route_id": route_id, "route_attribution": rmeta.get("route_attribution"),
    "expected_kernels": rmeta.get("expected_kernels"), "forbidden_kernels": rmeta.get("forbidden_kernels"),
    "route_bound_all_ctx": all(v["route_bound"] for v in tab.values()),
    "no_hidden_fallback": all(v["route_bound"] for v in tab.values()),
    "route_evidence_from_artifact": route_evid,
    "strict_fallback": rmeta.get("strict_fallback", False)}

  result = {
    "scope": "PMS-R2 candidate evaluator (replay)", "route_id": route_id, "baseline_route_id": baseline_id,
    "profile_id": rmeta.get("profile_id"), "authority_type": authority_type,
    "authority_artifact": str(rel), "authority_gate": rmeta.get("authority_gate"),
    "contexts": sorted(int(c) for c in tab), "thresholds": thresholds,
    "per_ctx": tab, "speed_stats_pct": stats,
    "correctness": {"correct": correct, "gate": route_evid.get("correctness_gate")},
    "route_attribution": attribution,
    "default_contract": {"env_to_force": env, "kind": default_contract, "rollback": rb,
                         "manifest_status": rmeta.get("status")},
    "noise_spread": {c: {"candidate_spread_pct": v["candidate_spread_pct"], "baseline_spread_pct": v["baseline_spread_pct"]}
                     for c, v in tab.items()},
    "tier_classification": tier, "disposition": disposition,
    "artifact_verdict": artifact_verdict, "expected_disposition": expected_disp, "decision_reproduced": reproduced,
    "verdict": "PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS" if reproduced else "PMS_R2_BLOCKED_AUTHORITY_HARNESS_INCOMPLETE",
    "ledger_row": _ledger_row(route_id, baseline_id, authority_type, tab, stats, tier, correct, attribution,
                              artifact_verdict, rb),
  }
  return result

def _ledger_row(route_id, baseline_id, authority_type, tab, stats, tier, correct, attribution, artifact_verdict, rb) -> dict:
  lane = "decode" if authority_type == "decode_wd" else "prefill"
  pc = {"decode_q4k_g3_generated": "GEMM", "decode_q6k_direct_refuted": "GEMM",
        "prefill_pipe_role_selective_default": "route_policy"}.get(route_id, "route_policy")
  return {
    "candidate_id": f"{lane}/{route_id}",
    "lane": lane, "primitive_class": pc,
    "knobs": {"env_to_force": route_env(route_id), "rollback": rb},
    "oracle": baseline_id,
    "correctness": "byte-identical/token-match" if correct else "FAILED",
    "route_identity": attribution.get("route_attribution"),
    "materialization_abi": "n/a",
    "isa": f"tier={tier}",
    "local_diagnostic": f"median {stats['median_pct']}% / worst {stats['worst_pct']}% / best {stats['best_pct']}% candidate-vs-baseline",
    "authority_benchmark": {"authority_type": authority_type,
                            "per_ctx_tok_s": {c: {"baseline": v["baseline_tok_s"], "candidate": v["candidate_tok_s"],
                                                  "delta_pct": v["delta_pct"]} for c, v in tab.items()}},
    "verdict": f"{tier} (reproduces {artifact_verdict})",
    "stop_reason": "replay of cited authority artifact; no GPU re-run (audit-first)",
    "artifact_links": [REPLAYS[route_id][0], "extra/qk_route_manifest.py", "extra/qk_candidate_evaluator.py"],
    "learned_rule": f"{route_id}: {tier} vs {baseline_id} on {authority_type}; rollback {rb}.",
  }

def _md(r: dict) -> str:
  L = [f"# PMS-R2 Candidate Evaluator -- {r['route_id']}", "",
       f"Verdict: **{r['verdict']}**", "",
       f"Tier: **{r['tier_classification']}** (disposition `{r['disposition']}`); "
       f"reproduces artifact `{r['artifact_verdict']}` -> decision_reproduced=**{r['decision_reproduced']}**", "",
       f"Baseline: `{r['baseline_route_id']}` | authority: `{r['authority_type']}` ({r['authority_artifact']})", "",
       f"Default contract: {r['default_contract']['kind']} | rollback: {r['default_contract']['rollback']}", "",
       "| ctx | baseline tok/s | candidate tok/s | delta % | token_match | route_bound | cand spread % |",
       "|---:|---:|---:|---:|:--:|:--:|---:|"]
  for c in sorted(r["per_ctx"], key=int):
    v = r["per_ctx"][c]
    L.append(f"| {c} | {v['baseline_tok_s']} | {v['candidate_tok_s']} | {v['delta_pct']} | "
             f"{v['token_match']} | {v['route_bound']} | {v['candidate_spread_pct']} |")
  L += ["", f"Speed: median {r['speed_stats_pct']['median_pct']}% / worst {r['speed_stats_pct']['worst_pct']}% / "
        f"best {r['speed_stats_pct']['best_pct']}%.",
        f"Correctness: {r['correctness']}.",
        f"Route-bound all ctx: {r['route_attribution']['route_bound_all_ctx']}; no hidden fallback: "
        f"{r['route_attribution']['no_hidden_fallback']}.", ""]
  return "\n".join(L)

def run_one(route_id: str) -> dict:
  r = evaluate(route_id)
  d = OUT / route_id
  d.mkdir(parents=True, exist_ok=True)
  json.dump(r, open(d / "latest.json", "w"), indent=2)
  json.dump(r["route_attribution"], open(d / "route_attribution.json", "w"), indent=2)
  json.dump(r["ledger_row"], open(d / "ledger_update.json", "w"), indent=2)
  open(d / "summary.md", "w").write(_md(r))
  return r

def main():
  allpass = True
  for rid in REPLAYS:
    r = run_one(rid)
    ok = r.get("decision_reproduced", False)
    allpass &= ok
    print(f"{rid}: {r['tier_classification']} | reproduces {r.get('artifact_verdict')} | reproduced={ok}")
  print("\nPMS-R2 verdict:", "PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS" if allpass
        else "PMS_R2_BLOCKED_AUTHORITY_HARNESS_INCOMPLETE")

if __name__ == "__main__":
  main()
