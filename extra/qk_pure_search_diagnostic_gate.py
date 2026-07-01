#!/usr/bin/env python3
"""TG-P6 gate: exercise the PURE_MACHINE_SEARCH_ONLY diagnostic (extra/qk_pure_search_guard.py).

Pure logic (no GPU): checks the guard's verdict under representative environments.
  fail-current   : normal fast default -> attention is the owned HIP tile (external_handwritten) -> IMPURE (must fail)
  pass-candidate : force the generated attention route (DECODE_FLASH_BLOCK_TILE_G5_8B=1) + all generated defaults on
                   -> every hot family machine_authored_generated -> PURE (must pass)
  rollback-named : roll a generated default back to its handwritten oracle -> IMPURE, and the violation names the route

Writes bench/tg-p6-pure-search-diagnostic/{latest.json,summary.md,fail_current.json,pass_candidate.json}.
Verdict TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE or a precise blocker.
"""
from __future__ import annotations
import json, pathlib

from extra.qk_pure_search_guard import effective_routes, pure_search_violations

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p6-pure-search-diagnostic"

# effective-route env scenarios. Generated defaults are on unless a rollback flag flips them.
GEN_ON = {"BUBBLEBEAM_FUTURESIGHT": "1", "DECODE_Q6K_GENERATED": "1", "PREFILL_GENERATED_SCHEDULE": "1"}
FAIL_CURRENT = dict(GEN_ON)                                   # attention owned (default) -> impure
PASS_CANDIDATE = {**GEN_ON, "DECODE_FLASH_BLOCK_TILE_G5_8B": "1"}   # attention generated -> pure
ROLLBACK_Q6K = {**PASS_CANDIDATE, "DECODE_Q6K_GENERATED": "0"}      # a named rollback -> impure


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  fail_v = pure_search_violations(FAIL_CURRENT)
  pass_v = pure_search_violations(PASS_CANDIDATE)
  rb_v = pure_search_violations(ROLLBACK_Q6K)

  fail_current = {"env": FAIL_CURRENT, "effective_routes": effective_routes(FAIL_CURRENT), "violations": fail_v,
                  "expected": "impure (attention owned HIP is external_handwritten)"}
  pass_candidate = {"env": PASS_CANDIDATE, "effective_routes": effective_routes(PASS_CANDIDATE), "violations": pass_v,
                    "expected": "pure (all hot families machine_authored_generated)"}
  json.dump(fail_current, open(OUT / "fail_current.json", "w"), indent=2)
  json.dump(pass_candidate, open(OUT / "pass_candidate.json", "w"), indent=2)

  # gates
  fail_current_ok = len(fail_v) >= 1 and any(v["family"] == "decode_attention" for v in fail_v)
  pass_candidate_ok = len(pass_v) == 0
  rollback_named_ok = any(v["family"] == "decode_q6k_gemv" and v["rolled_back_to_oracle"] and v["route_id"] for v in rb_v)
  route_report_ok = all("effective_route" in r and "provenance" in r for r in effective_routes(FAIL_CURRENT))

  all_ok = fail_current_ok and pass_candidate_ok and rollback_named_ok and route_report_ok
  verdict = "TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE" if all_ok else "TG_P6_BLOCKED_MANIFEST_RUNTIME_BINDING"
  latest = {"scope": "TG-P6 PURE_MACHINE_SEARCH_ONLY diagnostic mode", "verdict": verdict,
            "gates": {
              "fail_current": f"{'PASS' if fail_current_ok else 'FAIL'} (normal default is impure: attention owned HIP; "
                              f"violations={[v['family'] for v in fail_v]})",
              "pass_after": f"{'PASS' if pass_candidate_ok else 'FAIL'} (forcing generated attention makes all hot families pure)",
              "explicit_rollback": f"{'PASS' if rollback_named_ok else 'FAIL'} (a named rollback flag surfaces a violation "
                                   f"naming the route + scope)",
              "route_report": f"{'PASS' if route_report_ok else 'FAIL'} (guard prints per-family route + provenance)"},
            "note": "TG-P5 refuted the generated 8B attention on speed, so on a NORMAL fast run PURE_MACHINE_SEARCH_ONLY=1 "
                    "correctly FAILS (attention owned HIP is external_handwritten). pass_candidate demonstrates the mode "
                    "would pass if the generated attention were selected (at a speed cost).",
            "fail_current_violations": fail_v, "pass_candidate_violations": pass_v, "rollback_named_violations": rb_v}
  json.dump(latest, open(OUT / "latest.json", "w"), indent=2)
  md = [f"# TG-P6 Pure-Search Diagnostic Mode\n", f"Verdict: **{verdict}**\n",
        "| gate | result |", "|---|---|"]
  for k, v in latest["gates"].items(): md.append(f"| {k} | {v} |")
  md += ["", "Effective routes on a normal fast default run:", ""]
  for r in effective_routes(FAIL_CURRENT):
    md.append(f"- **{r['family']}**: `{r['effective_route']}` ({r['provenance']}) — {'pure' if r['pure'] else 'IMPURE'}")
  open(OUT / "summary.md", "w").write("\n".join(md) + "\n")
  print(verdict, "fail_current_viol=", [v["family"] for v in fail_v], "pass_candidate_viol=", pass_v)
  return 0 if all_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
