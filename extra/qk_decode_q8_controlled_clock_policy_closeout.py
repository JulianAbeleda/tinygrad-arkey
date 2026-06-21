#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_controlled_clock_policy_closeout_result.json"


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def read_json(relpath: str) -> dict[str, Any]:
  p = ROOT / relpath
  return json.loads(p.read_text()) if p.exists() else {}


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def lane_summary(artifact: dict[str, Any], lane: str) -> dict[str, Any]:
  return ((artifact.get("summary") or {}).get(lane) or {})


def metric(row: dict[str, Any], name: str, stat: str = "median") -> float | None:
  v = ((row.get(name) or {}).get(stat))
  return float(v) if v is not None else None


def main() -> int:
  ap = argparse.ArgumentParser(description="Close out q8 controlled-clock policy after clock authority audit")
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  clock = read_json("bench/qk-decode-primitive-transfer/decode_q8_clock_authority_result.json")
  confirm = read_json("bench/qk-decode-primitive-transfer/decode_q8_clock_authority_manual_peak_confirm_result.json")
  promotion = read_json("bench/q8-ffn-artifact-promotion/promotion_result.json")
  auto = lane_summary(clock, "auto")
  manual = lane_summary(confirm, "manual_peak") or lane_summary(clock, "manual_peak")

  auto_total = metric(auto, "total_us")
  manual_total = metric(manual, "total_us")
  manual_pass_sessions = int(manual.get("target_pass_sessions") or 0)
  manual_sessions = int(manual.get("sessions") or 0)
  gates = {
    "clock_audit_present": bool(clock),
    "manual_confirm_present": bool(confirm),
    "artifact_promotion_hardened_opt_in": promotion.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN",
    "auto_user_realistic_blocked": auto_total is not None and auto_total > args.target_us,
    "manual_peak_controlled_fast": manual_total is not None and manual_total <= args.target_us,
    "manual_peak_majority_sessions_pass": manual_sessions > 0 and manual_pass_sessions >= (manual_sessions // 2 + 1),
    "no_default_change": all(x.get("default_behavior_changed") is False for x in (clock, confirm, promotion) if x),
  }
  if not gates["auto_user_realistic_blocked"]:
    verdict = "BLOCKED_DECODE_Q8_POLICY_AUTO_NOT_BLOCKED_RECHECK"
  elif all(gates.values()):
    verdict = "PASS_DECODE_Q8_CONTROLLED_CLOCK_RESEARCH_ROUTE_POLICY"
  else:
    verdict = "BLOCKED_DECODE_Q8_CONTROLLED_CLOCK_POLICY_INCOMPLETE"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_CONTROLLED_CLOCK_POLICY_CLOSEOUT",
    "schema": "decode_q8_controlled_clock_policy_closeout_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "commit": git_sha(),
    "target_us": args.target_us,
    "inputs": {
      "clock_authority": "bench/qk-decode-primitive-transfer/decode_q8_clock_authority_result.json",
      "manual_peak_confirmation": "bench/qk-decode-primitive-transfer/decode_q8_clock_authority_manual_peak_confirm_result.json",
      "q8_artifact_promotion": "bench/q8-ffn-artifact-promotion/promotion_result.json",
    },
    "summary": {
      "auto": {
        "median_lifecycle_us": auto_total,
        "target_pass_sessions": auto.get("target_pass_sessions"),
        "sessions": auto.get("sessions"),
        "decision": "blocked_user_realistic_authority" if gates["auto_user_realistic_blocked"] else "recheck",
      },
      "manual_peak": {
        "median_lifecycle_us": manual_total,
        "best_lifecycle_us": metric(manual, "total_us", "min"),
        "worst_lifecycle_us": metric(manual, "total_us", "max"),
        "target_pass_sessions": manual_pass_sessions,
        "sessions": manual_sessions,
        "median_consumer_us": metric(manual, "consumer_us"),
        "median_producer_us": metric(manual, "producer_us"),
        "decision": "controlled_fast_research_authority" if gates["manual_peak_controlled_fast"] else "blocked",
      },
    },
    "policy": {
      "default_on": False,
      "route_status": "hardened_opt_in_controlled_clock_research_route",
      "enable_flag": "Q8_FFN_HANDWRITTEN=1",
      "clock_authority": "manual_peak only",
      "user_realistic_auto_authority": "blocked",
      "reporting_rule": "Report auto and manual_peak separately. Do not mix controlled-clock speed with auto-session claims.",
      "rollback": "unset Q8_FFN_HANDWRITTEN and restore GPU perf level to auto",
      "promotion_boundary": "No default-on promotion unless owner accepts controlled-clock authority or the route passes under auto.",
    },
    "gates": gates,
    "next": {
      "if_controlled_clock_accepted": "Document/run Q8_FFN_HANDWRITTEN=1 under manual_peak as a controlled research route.",
      "if_auto_required": "q8 remains blocked; do not spend next work on primitive rewrites until auto target/policy is changed.",
      "engineering_followup": "Wrap route benchmark command with lane set/restore guard if controlled-clock runs become routine.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "auto_median_us": auto_total,
    "manual_peak_median_us": manual_total,
    "manual_peak_pass_sessions": f"{manual_pass_sessions}/{manual_sessions}",
    "policy": result["policy"],
    "out": rel(args.out),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
