#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.qk_descriptor_policy import load_json, write_json


def _model_score(scorecard:dict[str, Any], model_size:str) -> dict[str, Any] | None:
  for row in scorecard.get("rows", []):
    if row.get("model_size") == model_size: return row
  return None


def _gap_model(gap_profile:dict[str, Any] | None, model_size:str) -> dict[str, Any] | None:
  if gap_profile is None: return None
  for row in gap_profile.get("models", []):
    if row.get("model") == model_size: return row
  return None


def _format_priority(gap_row:dict[str, Any] | None) -> dict[str, int]:
  if not gap_row or gap_row.get("status") != "profiled":
    return {"Q4_K": 0, "Q6_K": 1}
  attr = gap_row.get("attribution_named") or {}
  q4 = float(attr.get("q4k_primitive_gemv_ms_tok") or 0.0)
  q6 = float(attr.get("q6k_primitive_gemv_ms_tok") or 0.0)
  return {"Q4_K": 0, "Q6_K": 1} if q4 >= q6 else {"Q6_K": 0, "Q4_K": 1}


def _candidate_priority(candidate:dict[str, Any], priorities:dict[str, int]) -> tuple[int, int, str]:
  if candidate.get("id") == "current": return (-1, 0, "current")
  changes = candidate.get("changes") or []
  fmt = changes[0].get("format") if changes else ""
  storage = int(candidate.get("expected_storage_bytes") or 0)
  return (priorities.get(str(fmt), 9), -storage, str(candidate.get("id")))


def build_loop(candidate_set:dict[str, Any], static_gate:dict[str, Any], *,
               scorecard:dict[str, Any] | None=None, gap_profile:dict[str, Any] | None=None,
               max_to_benchmark:int=6) -> dict[str, Any]:
  if candidate_set.get("kind") != "qk_candidate_set": raise ValueError("expected kind=qk_candidate_set")
  if static_gate.get("kind") != "qk_candidate_static_gate": raise ValueError("expected kind=qk_candidate_static_gate")
  model_size = candidate_set.get("model_size")
  gates = {row["id"]: row for row in static_gate.get("rows", [])}
  candidates = {cand["id"]: cand for cand in candidate_set.get("candidates", [])}
  priorities = _format_priority(_gap_model(gap_profile, model_size))
  score = None if scorecard is None else _model_score(scorecard, model_size)

  rows = []
  for cand in sorted(candidates.values(), key=lambda c: _candidate_priority(c, priorities)):
    gate = gates.get(cand["id"])
    if gate is None:
      rows.append({"id": cand["id"], "decision": "reject_missing_static_gate", "reasons": ["candidate missing from static gate"]})
      continue
    if gate["status"] != "pass":
      rows.append({"id": cand["id"], "decision": "reject_static", "reasons": gate.get("reasons", [])})
      continue
    if cand["id"] == "current":
      rows.append({
        "id": cand["id"],
        "decision": "baseline",
        "changes": cand.get("changes", []),
        "generated_tok_s": None if score is None else score.get("generated_tok_s"),
        "generated_pct_llama": None if score is None else score.get("generated_pct_llama"),
        "policy_path": None,
        "reasons": ["current accepted policy anchors the search loop"],
      })
      continue
    pending_count = sum(1 for row in rows if row.get("decision") == "benchmark_next")
    decision = "benchmark_next" if pending_count < max_to_benchmark else "defer"
    rows.append({
      "id": cand["id"],
      "decision": decision,
      "changes": cand.get("changes", []),
      "expected_storage_bytes": cand.get("expected_storage_bytes"),
      "policy_path": None,
      "reasons": [] if decision == "benchmark_next" else [f"deferred by max_to_benchmark={max_to_benchmark}"],
    })
  summary = {
    "candidates": len(rows),
    "static_rejects": sum(1 for row in rows if row["decision"] == "reject_static"),
    "benchmark_next": sum(1 for row in rows if row["decision"] == "benchmark_next"),
    "deferred": sum(1 for row in rows if row["decision"] == "defer"),
    "baseline_tok_s": None if score is None else score.get("generated_tok_s"),
    "baseline_pct_llama": None if score is None else score.get("generated_pct_llama"),
  }
  return {
    "kind": "qk_ansor_transition_loop_v0",
    "mode": "static_candidate_planning",
    "model": candidate_set.get("model"),
    "model_size": model_size,
    "source_candidates": candidate_set.get("source_descriptor"),
    "max_to_benchmark": max_to_benchmark,
    "priority": {
      "basis": "named QK GEMV bucket ordering from gap profile when available",
      "format_priority": priorities,
    },
    "rows": rows,
    "summary": summary,
    "next_step": "benchmark rows with decision=benchmark_next through the QK policy pipeline; promote only if correctness and stability gates pass",
  }


def write_candidate_policies(loop:dict[str, Any], candidate_set:dict[str, Any], policies_dir:pathlib.Path) -> dict[str, str]:
  policies_dir.mkdir(parents=True, exist_ok=True)
  candidates = {cand["id"]: cand for cand in candidate_set.get("candidates", [])}
  written = {}
  for row in loop["rows"]:
    if row["decision"] not in ("baseline", "benchmark_next"): continue
    cand = candidates[row["id"]]
    path = policies_dir / f"{row['id']}.policy.json"
    write_json(path, cand["policy"])
    row["policy_path"] = str(path)
    written[row["id"]] = str(path)
  return written


def loop_markdown(loop:dict[str, Any]) -> str:
  lines = [
    f"# QK Ansor Transition Loop v0: {loop['model_size']}",
    "",
    "Static candidate-planning loop. This is the first reproducible machine",
    "surface after descriptors: generate policy candidates, fail-closed gate",
    "them, and emit the bounded set that should be benchmarked next.",
    "",
    "## Summary",
    "",
    f"- mode: `{loop['mode']}`",
    f"- baseline tok/s: `{loop['summary']['baseline_tok_s']}`",
    f"- baseline % llama.cpp: `{loop['summary']['baseline_pct_llama']}`",
    f"- benchmark next: `{loop['summary']['benchmark_next']}`",
    f"- deferred: `{loop['summary']['deferred']}`",
    f"- static rejects: `{loop['summary']['static_rejects']}`",
    "",
    "| id | decision | changes | policy | reasons |",
    "|---|---|---:|---|---|",
  ]
  for row in loop["rows"]:
    reasons = "; ".join(row.get("reasons") or []) or "none"
    policy = row.get("policy_path") or "n/a"
    lines.append(f"| `{row['id']}` | `{row['decision']}` | {len(row.get('changes') or [])} | `{policy}` | {reasons} |")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Build QK Ansor-transition search-loop v0 artifacts")
  parser.add_argument("--candidates", type=pathlib.Path, required=True)
  parser.add_argument("--static-gate", type=pathlib.Path, required=True)
  parser.add_argument("--scorecard", type=pathlib.Path)
  parser.add_argument("--gap-profile", type=pathlib.Path)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path)
  parser.add_argument("--policies-dir", type=pathlib.Path)
  parser.add_argument("--max-to-benchmark", type=int, default=6)
  args = parser.parse_args()
  candidate_set = load_json(args.candidates.expanduser())
  static_gate = load_json(args.static_gate.expanduser())
  scorecard = load_json(args.scorecard.expanduser()) if args.scorecard else None
  gap_profile = load_json(args.gap_profile.expanduser()) if args.gap_profile else None
  loop = build_loop(candidate_set, static_gate, scorecard=scorecard, gap_profile=gap_profile,
                    max_to_benchmark=args.max_to_benchmark)
  if args.policies_dir:
    write_candidate_policies(loop, candidate_set, args.policies_dir)
  write_json(args.json, loop)
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(loop_markdown(loop))
  else:
    print(loop_markdown(loop))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
