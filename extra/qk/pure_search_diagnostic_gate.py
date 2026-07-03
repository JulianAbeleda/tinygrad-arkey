#!/usr/bin/env python3
"""TG-P6 gate: exercise the PURE_MACHINE_SEARCH_ONLY diagnostic (extra/qk/pure_search_guard.py).

Pure logic (no GPU): checks the guard's verdict under representative environments.
  pass-current   : normal default -> every hot family is generated/spec-driven -> PURE (must pass)
  rollback-named : roll a generated default back to its rollback oracle -> IMPURE, and the violation names the route

Writes bench/tg-p6-pure-search-diagnostic/{latest.json,summary.md,pass_current.json,rollback_q6k.json}.
Verdict TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE or a precise blocker.
"""
from __future__ import annotations
import json, pathlib

from extra.qk.pure_search_guard import effective_routes, pure_search_violations

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/tg-p6-pure-search-diagnostic"

# effective-route env scenarios. Generated defaults are on unless a rollback flag flips them.
GEN_ON = {"BUBBLEBEAM_FUTURESIGHT": "1", "DECODE_Q6K_GENERATED": "1", "PREFILL_GENERATED_SCHEDULE": "1"}
PASS_CURRENT = {**GEN_ON, "DECODE_FLASH_BLOCK_TILE_G5_8B": "1"}
ROLLBACK_Q6K = {**PASS_CURRENT, "DECODE_Q6K_GENERATED": "0"}      # a named rollback -> impure


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  pass_v = pure_search_violations(PASS_CURRENT)
  rb_v = pure_search_violations(ROLLBACK_Q6K)

  pass_current = {"env": PASS_CURRENT, "effective_routes": effective_routes(PASS_CURRENT), "violations": pass_v,
                  "expected": "pure (all hot families machine_authored_generated)"}
  rollback_q6k = {"env": ROLLBACK_Q6K, "effective_routes": effective_routes(ROLLBACK_Q6K), "violations": rb_v,
                  "expected": "impure (explicit rollback to Q6_K shipped oracle)"}
  json.dump(pass_current, open(OUT / "pass_current.json", "w"), indent=2)
  json.dump(rollback_q6k, open(OUT / "rollback_q6k.json", "w"), indent=2)

  # gates
  pass_current_ok = len(pass_v) == 0
  rollback_named_ok = any(v["family"] == "decode_q6k_gemv" and v["rolled_back_to_oracle"] and v["route_id"] for v in rb_v)
  route_report_ok = all("effective_route" in r and "provenance" in r for r in effective_routes(PASS_CURRENT))

  all_ok = pass_current_ok and rollback_named_ok and route_report_ok
  verdict = "TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE" if all_ok else "TG_P6_BLOCKED_MANIFEST_RUNTIME_BINDING"
  latest = {"scope": "TG-P6 PURE_MACHINE_SEARCH_ONLY diagnostic mode", "verdict": verdict,
            "gates": {
              "pass_current": f"{'PASS' if pass_current_ok else 'FAIL'} (normal generated default has no violations)",
              "explicit_rollback": f"{'PASS' if rollback_named_ok else 'FAIL'} (a named rollback flag surfaces a violation "
                                   f"naming the route + scope)",
              "route_report": f"{'PASS' if route_report_ok else 'FAIL'} (guard prints per-family route + provenance)"},
            "note": "The hot default path is generated/spec-driven. Explicit rollback flags still surface as impurity.",
            "pass_current_violations": pass_v, "rollback_named_violations": rb_v}
  json.dump(latest, open(OUT / "latest.json", "w"), indent=2)
  md = [f"# TG-P6 Pure-Search Diagnostic Mode\n", f"Verdict: **{verdict}**\n",
        "| gate | result |", "|---|---|"]
  for k, v in latest["gates"].items(): md.append(f"| {k} | {v} |")
  md += ["", "Effective routes on the current generated default run:", ""]
  for r in effective_routes(PASS_CURRENT):
    md.append(f"- **{r['family']}**: `{r['effective_route']}` ({r['provenance']}) — {'pure' if r['pure'] else 'IMPURE'}")
  open(OUT / "summary.md", "w").write("\n".join(md) + "\n")
  print(verdict, "pass_current_viol=", pass_v, "rollback_named_viol=", [v["family"] for v in rb_v])
  return 0 if all_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
