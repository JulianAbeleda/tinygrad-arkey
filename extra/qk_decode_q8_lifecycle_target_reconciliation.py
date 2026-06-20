#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_lifecycle_target_reconciliation_result.json"
CHILD = ROOT / "extra/qk_decode_owned_q8_interleaved_lifecycle_gate.py"


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def median(xs: list[float]) -> float:
  return float(statistics.median(xs))


def summarize(xs: list[float]) -> dict[str, Any]:
  if not xs: return {"n": 0}
  return {
    "n": len(xs),
    "min_us": min(xs),
    "p10_us": sorted(xs)[max(0, int(len(xs) * 0.10) - 1)],
    "median_us": median(xs),
    "mean_us": float(statistics.fmean(xs)),
    "max_us": max(xs),
  }


def lifecycle_totals(result: dict[str, Any]) -> list[float]:
  return [float(row["total_us"]) for row in result.get("rows", []) if row.get("label") == "lifecycle"]


def run_child(args: argparse.Namespace, session: int) -> tuple[dict[str, Any], dict[str, Any]]:
  child_out = args.out.parent / f"decode_q8_lifecycle_target_reconciliation_session_{session:02d}.json"
  cmd = [
    sys.executable, rel(CHILD),
    "--rounds", str(args.rounds),
    "--warmups", str(args.warmups),
    "--seed", str(args.seed + session),
    "--target-us", str(args.target_us),
    "--out", rel(child_out),
  ]
  if args.rows is not None:
    cmd += ["--rows", str(args.rows)]
  if args.gguf is not None:
    cmd += ["--gguf", str(args.gguf)]
  p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if not child_out.exists():
    return {"session": session, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(child_out)}, {}
  return {"session": session, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(child_out)}, json.loads(child_out.read_text())


def main() -> int:
  ap = argparse.ArgumentParser(description="Repeat decode q8 interleaved lifecycle gate and reconcile target margin")
  ap.add_argument("--sessions", type=int, default=5)
  ap.add_argument("--rounds", type=int, default=24)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--steady-drop", type=int, default=4)
  ap.add_argument("--seed", type=int, default=31)
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--threshold-variance-us", type=float, default=1.0)
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  children: list[dict[str, Any]] = []
  sessions: list[dict[str, Any]] = []
  for session in range(args.sessions):
    child_meta, child_result = run_child(args, session)
    children.append(child_meta)
    totals = lifecycle_totals(child_result)
    steady = totals[args.steady_drop:]
    summaries = child_result.get("summaries", {})
    session_row = {
      "session": session,
      "artifact": child_meta["artifact"],
      "returncode": child_meta["returncode"],
      "verdict": child_result.get("verdict"),
      "gates": child_result.get("gates", {}),
      "full_lifecycle": summarize(totals),
      "steady_lifecycle": summarize(steady),
      "producer_only_median_us": (summaries.get("producer_only") or {}).get("median_us"),
      "lifecycle_producer_median_us": (summaries.get("lifecycle_producer") or {}).get("median_us"),
      "lifecycle_consumer_median_us": (summaries.get("lifecycle_consumer") or {}).get("median_us"),
      "clock": child_result.get("clock", {}),
    }
    sessions.append(session_row)

  full_medians = [s["full_lifecycle"]["median_us"] for s in sessions if s["full_lifecycle"].get("n")]
  steady_medians = [s["steady_lifecycle"]["median_us"] for s in sessions if s["steady_lifecycle"].get("n")]
  best_rows = [s["full_lifecycle"]["min_us"] for s in sessions if s["full_lifecycle"].get("n")]
  full_reconciled_us = median(full_medians) if full_medians else float("inf")
  steady_reconciled_us = median(steady_medians) if steady_medians else float("inf")
  best_observed_us = min(best_rows) if best_rows else float("inf")
  best_policy_us = min(full_reconciled_us, steady_reconciled_us)
  delta_us = best_policy_us - args.target_us

  all_artifacts_present = len(full_medians) == args.sessions
  all_producer_correct = all((s.get("gates") or {}).get("producer_correct") for s in sessions)
  all_consumer_correct = all((s.get("gates") or {}).get("consumer_correct") for s in sessions)
  full_clears = full_reconciled_us <= args.target_us
  steady_clears = steady_reconciled_us <= args.target_us

  if not all_artifacts_present or not all_producer_correct or not all_consumer_correct:
    verdict = "BLOCKED_DECODE_Q8_LIFECYCLE_INCORRECT"
  elif full_clears or steady_clears:
    verdict = "PASS_DECODE_Q8_LIFECYCLE_TARGET_RECONCILED"
  elif delta_us <= args.threshold_variance_us:
    verdict = "BLOCKED_DECODE_Q8_LIFECYCLE_THRESHOLD_VARIANCE"
  else:
    verdict = "BLOCKED_DECODE_Q8_LIFECYCLE_SCHEDULE_DEBT"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_LIFECYCLE_TARGET_RECONCILIATION",
    "schema": "decode_q8_lifecycle_target_reconciliation_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "target_us": args.target_us,
    "threshold_variance_us": args.threshold_variance_us,
    "sessions_requested": args.sessions,
    "rounds_per_session": args.rounds,
    "steady_drop": args.steady_drop,
    "summary": {
      "full_median_of_session_medians_us": full_reconciled_us,
      "steady_median_of_session_medians_us": steady_reconciled_us,
      "best_policy_delta_us": delta_us,
      "best_observed_row_us": best_observed_us,
      "best_observed_delta_us": best_observed_us - args.target_us,
    },
    "gates": {
      "all_artifacts_present": all_artifacts_present,
      "all_producer_correct": all_producer_correct,
      "all_consumer_correct": all_consumer_correct,
      "full_reconciled_lte_target": full_clears,
      "steady_reconciled_lte_target": steady_clears,
      "delta_lte_threshold_variance": delta_us <= args.threshold_variance_us,
    },
    "sessions": sessions,
    "children": children,
    "decision": {
      "if_pass": "target clears under repeated full or steady paired policy; reopen promotion discussion",
      "if_threshold_variance": "route remains within 1us; decide target/steady policy before schedule work",
      "if_schedule_debt": "reopen consumer/native schedule work with repeated steady miss as objective",
      "if_incorrect": "fix correctness or artifact generation before timing",
    },
  }
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "full_median_of_session_medians_us": full_reconciled_us,
    "steady_median_of_session_medians_us": steady_reconciled_us,
    "target_us": args.target_us,
    "best_policy_delta_us": delta_us,
    "best_observed_row_us": best_observed_us,
    "gates": result["gates"],
    "out": rel(args.out),
  }, indent=2))
  return 0 if all_artifacts_present and all_producer_correct and all_consumer_correct else 1


if __name__ == "__main__":
  raise SystemExit(main())
