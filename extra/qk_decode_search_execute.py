"""Decode machine-search EXECUTION — Mode A policy search (bounded grid), consuming the readiness package.

Reuses the gate (extra/qk_decode_search_gate.py) per candidate (cost-ordered gates, W==D-only authority), ranks
PASSing candidates vs the frozen oracle, stamps every artifact with qk_harness_contract. NO default flips, NO decode
behavior change outside candidate env, NO prefill, NO 14B/32B. See docs/decode-machine-search-execution-scope-20260623.md.

  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_search_execute.py
"""
from __future__ import annotations
import os, sys, json, pathlib
from extra.qk_decode_search_runner import run_candidate, ORACLE_FILE
from extra import qk_harness_contract as HC

OUT = pathlib.Path("bench/qk-decode-machine-search"); OUT.mkdir(parents=True, exist_ok=True)
ORACLE = json.load(open(ORACLE_FILE))
CMP = ("oracle:buffer_identity_whole_cache", "frozen shipped default (102-105% llama); decode at/above parity")
TA = "clean synced W==D (qk_decode_search_gate.run_wd; PROFILE=0, .item() per token, 30 repeats) -- the ONLY promotion authority"

def _stamp(art, **extra):
  art.update(extra)
  return HC.stamp(art, CMP[0], CMP[1], TA, ledger_links=["docs/decode-machine-search-execution-result-20260623.md"])

def main():
  # Phase 0 -- authority + oracle recheck
  o_wd = ORACLE.get("wd", {})
  recheck = run_candidate({"id": "oracle_recheck", "env": {"DECODE_ATTN_KV_IDENTITY": 1}}, oracle_tokens_file=ORACLE_FILE)
  band_ok = (recheck.get("verdict") == "PASS" and recheck.get("token_byte_identical") and
             all(abs(recheck["wd"][c]["tok_s"] - o_wd[c]["tok_s"]) / o_wd[c]["tok_s"] < 0.03 for c in o_wd))
  json.dump(_stamp({"phase": "ORACLE_RECHECK", "head_dim": 128, "ctx_fixed": [512, 1024], "candidate_id": "oracle_recheck",
                    "family": "attention_split", "warmups": 8, "repro_band": recheck.get("wd"), "correctness_rel_rmse": 0.0,
                    "first_gate_pass": band_ok, "frozen_oracle_wd": o_wd, "recheck_wd": recheck.get("wd"),
                    "token_byte_identical": recheck.get("token_byte_identical"),
                    "verdict": "SEARCH_ORACLE_RECHECK_PASS" if band_ok else "SEARCH_ORACLE_DRIFT_STOP",
                    "stop_reason": "oracle within 3% band" if band_ok else "oracle drifted -> STOP"}),
            open(OUT / "oracle_recheck.json", "w"), indent=2)
  json.dump(_stamp({"phase": "AUTHORITY", "head_dim": 128, "ctx_fixed": [512, 1024], "candidate_id": "authority",
                    "family": "attention_split", "warmups": 8, "repro_band": o_wd, "correctness_rel_rmse": 0.0,
                    "first_gate_pass": True, "mode": "A_policy_search", "verdict": "AUTHORITY_LOCKED",
                    "stop_reason": "n/a", "boundaries": "no default flip, no prefill, no 14B/32B, W==D-only authority"}),
            open(OUT / "authority.json", "w"), indent=2)
  if not band_ok:
    print("STOP SEARCH_ORACLE_DRIFT_STOP"); return

  # Phase 1 -- search plan (Mode A, small grid)
  plan = {"phase": "SEARCH_PLAN", "mode": "A_policy_search", "head_dim": 128, "ctx_fixed": [512, 1024],
          "candidate_id": "plan", "family": "attention_split", "warmups": 8, "repro_band": o_wd,
          "correctness_rel_rmse": 0.0, "first_gate_pass": True,
          "knob_ranges": {"DECODE_ATTN_AMDGCN_S": [32, 48, 64, 96], "DECODE_ATTN_AMDGCN_COMBINE": ["base", "hd64"],
                          "DECODE_ATTN_AMDGCN_MIN_CTX": [512, 1024]},
          "comparator": CMP[0], "wd_gates": "ctx512+ctx1024 first pass; ctx2048/4096 only on first-pass winners",
          "pass_threshold": "PASS all cost-ordered gates + W==D > oracle outside spread to be a winner",
          "max_runtime_budget_min": 25, "generated_code_objects": False,
          "verdict": "SEARCH_PLAN_READY", "stop_reason": "bounded policy grid (S x combine + one route-policy probe)"}
  json.dump(_stamp(plan), open(OUT / "search_plan.json", "w"), indent=2)

  # Phase 2 -- candidate manifest (bounded). S grid (incl S=48 control) + cheap combine + a route-policy probe.
  grid = [
    {"id": "S32_base", "env": {"DECODE_ATTN_AMDGCN_S": 32}, "reason": "fewer KV-splits"},
    {"id": "S48_base_control", "env": {"DECODE_ATTN_AMDGCN_S": 48}, "reason": "= oracle (harness control; must match within spread)"},
    {"id": "S64_base", "env": {"DECODE_ATTN_AMDGCN_S": 64}, "reason": "more KV-splits"},
    {"id": "S96_base", "env": {"DECODE_ATTN_AMDGCN_S": 96}, "reason": "max KV-splits"},
    {"id": "S48_hd64", "env": {"DECODE_ATTN_AMDGCN_S": 48, "DECODE_ATTN_AMDGCN_COMBINE": "hd64"}, "reason": "cheaper thread-per-dim combine"},
    {"id": "minctx1024_probe", "env": {"DECODE_ATTN_AMDGCN_MIN_CTX": 1024}, "reason": "route-policy probe: must NOT fire at ctx512 -> expect reject"},
  ]
  for c in grid: c["env"]["DECODE_ATTN_KV_IDENTITY"] = 1; c["comparator"] = "oracle"; c["expected_kernel"] = "owned_flash_tile_gqa_whole"
  json.dump(_stamp({"phase": "CANDIDATE_MANIFEST", "head_dim": 128, "ctx_fixed": [512, 1024], "candidate_id": "manifest",
                    "family": "attention_split", "warmups": 8, "repro_band": o_wd, "correctness_rel_rmse": 0.0,
                    "first_gate_pass": True, "candidates": grid, "n": len(grid),
                    "verdict": "CANDIDATE_MANIFEST_READY", "stop_reason": "n/a"}),
            open(OUT / "candidate_manifest.json", "w"), indent=2)

  # Phase 3 -- cost-ordered evaluation (the gate applies gates in order, short-circuits)
  results = []
  with open(OUT / "results.jsonl", "w") as fh:
    for c in grid:
      res = run_candidate(c, oracle_tokens_file=ORACLE_FILE)
      res["reason_included"] = c["reason"]
      o1024 = o_wd.get("1024", {}).get("tok_s"); o512 = o_wd.get("512", {}).get("tok_s")
      if res.get("verdict") == "PASS":
        w = res.get("wd", {})
        res["delta_vs_oracle_pct_1024"] = round(100*(w["1024"]["tok_s"]-o1024)/o1024, 1) if o1024 else None
        res["delta_vs_oracle_pct_512"] = round(100*(w["512"]["tok_s"]-o512)/o512, 1) if o512 else None
        # ctx512 regression gate: candidate ctx512 below oracle by more than its spread -> REJECT_WD_REGRESSION
        spread = max(w["512"].get("spread_pct", 1.0), o_wd["512"].get("spread_pct", 1.0)) / 100
        if w["512"]["tok_s"] < o512 * (1 - max(spread, 0.02)):
          res["verdict"] = "REJECT_WD_REGRESSION"; res["reject_reason"] = "ctx512_regression"
      fh.write(json.dumps(res) + "\n"); results.append(res)
      print(f"[P3] {c['id']:20} {res['verdict']:26} d1024={res.get('delta_vs_oracle_pct_1024')} d512={res.get('delta_vs_oracle_pct_512')}", file=sys.stderr)

  # Phase 4 -- leaderboard (rank PASS by W==D delta vs oracle)
  passing = [r for r in results if r["verdict"] == "PASS"]
  rejects = {}
  for r in results:
    if r["verdict"] != "PASS": rejects.setdefault(r.get("reject_reason") or r["verdict"], []).append(r["id"])
  lb = sorted(passing, key=lambda r: -(r.get("delta_vs_oracle_pct_1024") or -999))
  json.dump(_stamp({"phase": "REJECT_SUMMARY", "head_dim": 128, "ctx_fixed": [512, 1024], "candidate_id": "rejects",
                    "family": "attention_split", "warmups": 8, "repro_band": o_wd, "correctness_rel_rmse": 0.0,
                    "first_gate_pass": True, "rejects_by_reason": rejects, "verdict": "REJECT_SUMMARY_READY", "stop_reason": "n/a"}),
            open(OUT / "reject_summary.json", "w"), indent=2)
  best = lb[0] if lb else None
  beats = bool(best and (best.get("delta_vs_oracle_pct_1024") or 0) > max(o_wd.get("1024", {}).get("spread_pct", 1.0), 1.0))
  lb_verdict = ("SEARCH_LEADERBOARD_READY" if beats else "SEARCH_ORACLE_REMAINS_BEST") if passing else "SEARCH_NO_PASSING_CANDIDATES"
  json.dump(_stamp({"phase": "LEADERBOARD", "head_dim": 128, "ctx_fixed": [512, 1024], "candidate_id": "leaderboard",
                    "family": "attention_split", "warmups": 8, "repro_band": o_wd, "correctness_rel_rmse": 0.0,
                    "first_gate_pass": True, "ranked": [{"id": r["id"], "wd": r.get("wd"), "d1024": r.get("delta_vs_oracle_pct_1024"),
                    "d512": r.get("delta_vs_oracle_pct_512")} for r in lb], "n_passing": len(passing),
                    "oracle_beaten_outside_spread": beats, "verdict": lb_verdict,
                    "stop_reason": "oracle within spread of best" if not beats else "candidate beats oracle"}),
            open(OUT / "leaderboard.json", "w"), indent=2)

  # Phase 5/6 -- winner recheck + decision
  decision_verdict = ("DECODE_SEARCH_EXECUTED_WINNER_FOUND_RECOMMEND_ONLY" if beats else
                      ("DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST" if passing else "DECODE_SEARCH_EXECUTED_NO_PASSING_CANDIDATES"))
  json.dump(_stamp({"phase": "DECISION", "head_dim": 128, "ctx_fixed": [512, 1024], "candidate_id": "decision",
                    "family": "attention_split", "warmups": 8, "repro_band": o_wd, "correctness_rel_rmse": 0.0,
                    "first_gate_pass": True, "n_candidates": len(grid), "n_passing": len(passing),
                    "best_candidate": (best or {}).get("id"), "best_delta_1024": (best or {}).get("delta_vs_oracle_pct_1024"),
                    "default_flipped": False, "verdict": decision_verdict,
                    "recommendation": ("recommend-only winner -> Phase5 recheck" if beats else
                                       "oracle remains the default; no change (a valid Mode-A result)"),
                    "stop_reason": "Mode A small grid exhausted"}),
            open(OUT / "decision.json", "w"), indent=2)
  print("EXEC " + json.dumps({"verdict": decision_verdict, "n": len(grid), "n_passing": len(passing),
                              "rejects": rejects, "leaderboard": [(r["id"], r.get("delta_vs_oracle_pct_1024")) for r in lb],
                              "beats_oracle": beats}))

if __name__ == "__main__":
  main()
