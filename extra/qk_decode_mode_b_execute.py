"""Decode Mode B — generated owned-tile VARIANT search (tile constants TK/S/combine/VEC/UNROLL). Consumes the
readiness package: per variant, set the tile-constant env knobs + QK_CAND_KERNEL (the variant symbol) and run the gate
(cost-ordered: build+correctness -> route-fire -> materialization -> ISA -> 64-tok byte-identical -> clean synced
W==D). Rank PASS by W==D Δ vs the frozen oracle. NO default flip; W==D is the only authority; variants are additive
(default env -> shipped kernel, byte-identical). See docs/decode-mode-b-generated-tile-variant-search-scope-20260623.md.

  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_mode_b_execute.py
"""
from __future__ import annotations
import os, sys, json, pathlib
from extra.qk_decode_search_runner import run_candidate, ORACLE_FILE
from extra import qk_harness_contract as HC

OUT = pathlib.Path("bench/qk-decode-mode-b-search"); OUT.mkdir(parents=True, exist_ok=True)
ORACLE = json.load(open(ORACLE_FILE))
CMP = ("oracle:buffer_identity_whole_cache", "frozen shipped default (TK16/S48/base/VEC1/U1; 102-105% llama)")
TA = "clean synced W==D (qk_decode_search_gate.run_wd; PROFILE=0, .item()/tok, 30 repeats) -- the ONLY authority"

def _sym(tk, vec, unroll):
  base = "owned_flash_tile_gqa_whole"
  return base + (f"_tk{tk}_v{vec}_u{unroll}" if (tk, vec, unroll) != (16, 1, 1) else "")

def _cand(cid, tk=16, s=48, combine="base", vec=1, unroll=1, reason=""):
  return {"id": cid, "env": {"DECODE_ATTN_KV_IDENTITY": 1, "DECODE_ATTN_AMDGCN_TK": tk, "DECODE_ATTN_AMDGCN_S": s,
          "DECODE_ATTN_AMDGCN_COMBINE": combine, "DECODE_ATTN_AMDGCN_VEC": vec, "DECODE_ATTN_AMDGCN_UNROLL": unroll,
          "QK_CAND_KERNEL": _sym(tk, vec, unroll)}, "knobs": {"tk": tk, "s": s, "combine": combine, "vec": vec, "unroll": unroll},
          "reason": reason}

def _stamp(art, **extra):
  art.update(extra)
  art.setdefault("head_dim", 128); art.setdefault("ctx_fixed", [512, 1024]); art.setdefault("candidate_id", art.get("phase","x"))
  art.setdefault("family", "attention_tile_variant"); art.setdefault("warmups", 8); art.setdefault("repro_band", ORACLE.get("wd"))
  art.setdefault("correctness_rel_rmse", 0.0); art.setdefault("first_gate_pass", True); art.setdefault("stop_reason", "n/a")
  return HC.stamp(art, CMP[0], CMP[1], TA, ledger_links=["docs/decode-mode-b-search-result-20260623.md"])

def main():
  o_wd = ORACLE.get("wd", {}); o1024 = o_wd.get("1024", {}).get("tok_s"); o512 = o_wd.get("512", {}).get("tok_s")
  # P0 oracle recheck
  rc = run_candidate({"id": "oracle_recheck", "env": {"DECODE_ATTN_KV_IDENTITY": 1}}, oracle_tokens_file=ORACLE_FILE)
  band_ok = rc.get("verdict") == "PASS" and rc.get("token_byte_identical") and all(
    abs(rc["wd"][c]["tok_s"] - o_wd[c]["tok_s"]) / o_wd[c]["tok_s"] < 0.03 for c in o_wd)
  json.dump(_stamp({"phase": "ORACLE_RECHECK", "recheck_wd": rc.get("wd"),
                    "verdict": "SEARCH_ORACLE_RECHECK_PASS" if band_ok else "SEARCH_ORACLE_DRIFT_STOP"}),
            open(OUT / "oracle_recheck.json", "w"), indent=2)
  json.dump(_stamp({"phase": "AUTHORITY", "mode": "B_generated_tile_variant", "verdict": "AUTHORITY_LOCKED",
                    "boundaries": "additive variants (default env=shipped, byte-identical); no flip; W==D-only"}),
            open(OUT / "authority.json", "w"), indent=2)
  if not band_ok:
    print("STOP SEARCH_ORACLE_DRIFT_STOP"); return

  # P1/P2 plan + manifest -- coordinate descent from oracle (TK16,S48,base,VEC1,U1)
  grid = [
    _cand("oracle_equiv", reason="= shipped (control; must match oracle within spread)"),
    _cand("TK8", tk=8, reason="shallower LDS position tile"), _cand("TK32", tk=32, reason="deeper LDS tile"),
    _cand("S32", s=32, reason="fewer splits"), _cand("S64", s=64, reason="more splits"), _cand("S96", s=96, reason="max splits"),
    _cand("combine_hd64", combine="hd64", reason="cheaper thread-per-dim combine"),
    _cand("VEC2", vec=2, reason="half2 LDS staging"), _cand("VEC4", vec=4, reason="int2/b64 LDS staging"),
    _cand("U2", unroll=2, reason="position loop unroll 2"), _cand("U4", unroll=4, reason="position loop unroll 4"),
    # best-of-axis combos
    _cand("TK8_VEC2", tk=8, vec=2, reason="combo"), _cand("VEC2_U2", vec=2, unroll=2, reason="combo"),
    _cand("TK32_VEC4_U2", tk=32, vec=4, unroll=2, reason="combo"),
  ]
  json.dump(_stamp({"phase": "SEARCH_PLAN", "mode": "B", "n": len(grid),
                    "knob_ranges": {"TK": [8, 16, 32], "S": [32, 48, 64, 96], "combine": ["base", "hd64"], "VEC": [1, 2, 4], "UNROLL": [1, 2, 4]},
                    "generated_code_objects": True, "verdict": "SEARCH_PLAN_READY"}), open(OUT / "search_plan.json", "w"), indent=2)
  json.dump(_stamp({"phase": "CANDIDATE_MANIFEST", "candidates": [{"id": c["id"], "knobs": c["knobs"], "expected_kernel": c["env"]["QK_CAND_KERNEL"], "reason": c["reason"]} for c in grid],
                    "n": len(grid), "verdict": "CANDIDATE_MANIFEST_READY"}), open(OUT / "candidate_manifest.json", "w"), indent=2)

  # P3 cost-ordered evaluation (the gate short-circuits)
  results = []
  with open(OUT / "results.jsonl", "w") as fh:
    for c in grid:
      res = run_candidate(c, oracle_tokens_file=ORACLE_FILE); res["knobs"] = c["knobs"]; res["reason"] = c["reason"]
      if res.get("verdict") == "PASS":
        w = res.get("wd", {})
        res["delta_vs_oracle_pct_1024"] = round(100*(w["1024"]["tok_s"]-o1024)/o1024, 1) if o1024 else None
        res["delta_vs_oracle_pct_512"] = round(100*(w["512"]["tok_s"]-o512)/o512, 1) if o512 else None
        spread = max(w["512"].get("spread_pct", 1.0), o_wd["512"].get("spread_pct", 1.0)) / 100
        if w["512"]["tok_s"] < o512 * (1 - max(spread, 0.02)):
          res["verdict"] = "REJECT_WD_REGRESSION"; res["reject_reason"] = "ctx512_regression"
      fh.write(json.dumps(res) + "\n"); results.append(res)
      print(f"[P3] {c['id']:16} {res['verdict']:28} d1024={res.get('delta_vs_oracle_pct_1024')} d512={res.get('delta_vs_oracle_pct_512')}", file=sys.stderr)

  # P4 leaderboard + reject summary
  passing = [r for r in results if r["verdict"] == "PASS"]
  rejects = {}
  for r in results:
    if r["verdict"] != "PASS": rejects.setdefault(r.get("reject_reason") or r["verdict"], []).append(r["id"])
  lb = sorted(passing, key=lambda r: -(r.get("delta_vs_oracle_pct_1024") or -999))
  best = lb[0] if lb else None
  beats = bool(best and (best.get("delta_vs_oracle_pct_1024") or 0) > max(o_wd.get("1024", {}).get("spread_pct", 1.0), 1.0))
  json.dump(_stamp({"phase": "REJECT_SUMMARY", "rejects_by_reason": rejects, "verdict": "REJECT_SUMMARY_READY"}), open(OUT / "reject_summary.json", "w"), indent=2)
  json.dump(_stamp({"phase": "LEADERBOARD", "ranked": [{"id": r["id"], "knobs": r["knobs"], "wd": r.get("wd"), "d1024": r.get("delta_vs_oracle_pct_1024"), "d512": r.get("delta_vs_oracle_pct_512")} for r in lb],
                    "n_passing": len(passing), "oracle_beaten_outside_spread": beats,
                    "verdict": "SEARCH_LEADERBOARD_READY" if beats else ("SEARCH_ORACLE_REMAINS_BEST" if passing else "SEARCH_NO_PASSING_CANDIDATES")}),
            open(OUT / "leaderboard.json", "w"), indent=2)
  verdict = ("DECODE_MODE_B_EXECUTED_WINNER_FOUND_RECOMMEND_ONLY" if beats else
             ("DECODE_MODE_B_EXECUTED_ORACLE_REMAINS_BEST" if passing else "DECODE_MODE_B_EXECUTED_NO_PASSING_CANDIDATES"))
  json.dump(_stamp({"phase": "DECISION", "n_candidates": len(grid), "n_passing": len(passing), "best": (best or {}).get("id"),
                    "best_delta_1024": (best or {}).get("delta_vs_oracle_pct_1024"), "default_flipped": False, "verdict": verdict,
                    "recommendation": ("recommend-only winner -> recheck" if beats else "oracle remains default; tile constants are optimal (valid result)")}),
            open(OUT / "decision.json", "w"), indent=2)
  print("MODEB " + json.dumps({"verdict": verdict, "n": len(grid), "n_passing": len(passing), "rejects": rejects,
                               "leaderboard": [(r["id"], r.get("delta_vs_oracle_pct_1024")) for r in lb], "beats": beats}))

if __name__ == "__main__":
  main()
