#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.qk_semantic_candidate import is_raw_accept_status

DEFAULT_MODELS = ("8b", "14b")


def _read(path:pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def _write(path:pathlib.Path, data:dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _portable(path:pathlib.Path, repo:pathlib.Path) -> str:
  resolved = str(repo.resolve())
  text = str(path)
  if text == resolved: return "."
  return text.replace(resolved + "/", "")


def _model_report(base:pathlib.Path, model:str) -> dict[str, Any]:
  root = base / model
  candidates = _read(root / "candidates.json")
  static_gate = _read(root / "static-gate.json")
  microbench = _read(root / "microbench.json")
  rows = microbench.get("rows") or []
  raw_accepts = [row for row in rows if is_raw_accept_status(row.get("status"))]
  strong_raw_accepts = [row for row in raw_accepts if row.get("gain") is not None and row["gain"] >= 0.10]
  return {
    "model": model.upper(),
    "candidate_summary": candidates.get("summary") or {},
    "static_gate_summary": static_gate.get("summary") or {},
    "microbench_summary": microbench.get("summary") or {},
    "rows": [
      {
        "id": row["id"],
        "status": row.get("status"),
        "gain": row.get("gain"),
        "current_gbs": (row.get("current") or {}).get("quant_gbs"),
        "candidate_gbs": (row.get("candidate") or {}).get("quant_gbs"),
        "full_decode_supported": row.get("full_decode_supported"),
        "reasons": row.get("reasons") or [],
      }
      for row in rows
    ],
    "raw_accepts": [row["id"] for row in raw_accepts],
    "strong_raw_accepts": [row["id"] for row in strong_raw_accepts],
    "status": "raw_accept" if raw_accepts else "reject",
  }


def build_verdict(base:pathlib.Path, *, repo:pathlib.Path=pathlib.Path.cwd(), models:tuple[str, ...]=DEFAULT_MODELS) -> dict[str, Any]:
  reports = [_model_report(base, model) for model in models]
  all_rows = [row for report in reports for row in report["rows"]]
  raw_accepts = [row for row in all_rows if is_raw_accept_status(row.get("status"))]
  strong_raw_accepts = [row for row in raw_accepts if row.get("gain") is not None and row["gain"] >= 0.10]
  invalid = [row for row in all_rows if row.get("status") == "invalid"]
  if strong_raw_accepts:
    overall = "semantic_codegen_v4_strong_raw_accept_unconfirmed"
  elif raw_accepts:
    overall = "semantic_codegen_v4_weak_raw_accept_unconfirmed"
  else:
    overall = "semantic_codegen_v4_rejected"
  reasons = []
  for report in reports:
    if report["strong_raw_accepts"]:
      reasons.append(f"{report['model']} strong raw accepts: {', '.join(report['strong_raw_accepts'])}")
    elif report["raw_accepts"]:
      reasons.append(f"{report['model']} weak raw accepts: {', '.join(report['raw_accepts'])}; below full-decode promise threshold")
    else:
      rejected = [row for row in report["rows"] if row.get("status") == "reject"]
      tied = [row for row in report["rows"] if row.get("status") == "tie"]
      worst_invalid = [row for row in report["rows"] if row.get("status") == "invalid"]
      bits = []
      if tied: bits.append(f"{len(tied)} ties")
      if rejected: bits.append(f"{len(rejected)} rejects")
      if worst_invalid: bits.append(f"{len(worst_invalid)} invalid")
      reasons.append(f"{report['model']} no raw accepts ({', '.join(bits) or 'no rows'})")
  if strong_raw_accepts:
    reasons.append("full decode still requires a matching confirmation rerun before promotion")
  else:
    reasons.append("full decode and 32B skipped because the 8B/14B microbench gate produced no strong raw accepts")
  reasons.append("Family C v1 is an aligned uint32x4 vector-load memory-access probe, not a schedule-only knob")
  return {
    "kind": "qk_semantic_codegen_v4_verdict",
    "schema_version": 1,
    "base": _portable(base, repo),
    "gate_models": [model.upper() for model in models],
    "models": reports,
    "summary": {
      "overall_decision": overall,
      "models": len(reports),
      "microbench_rows": len(all_rows),
      "raw_microbench_accepts": len(raw_accepts),
      "strong_raw_microbench_accepts": len(strong_raw_accepts),
      "microbench_invalid": len(invalid),
      "full_decode_candidates": len(strong_raw_accepts),
      "full_decode_confirmed_accepts": 0,
      "run_32b": False,
      "reasons": reasons,
    },
  }


def verdict_markdown(verdict:dict[str, Any]) -> str:
  lines = [
    "# QK Semantic Codegen v4 Verdict",
    "",
    "This is the 8B/14B gate for Family C v1: exact-tensor Q4_K ffn_gate",
    "aligned uint32x4 vector-load partial GEMV. 32B is intentionally excluded",
    "unless both target models show promise.",
    "",
    "## Summary",
    "",
    f"- overall decision: `{verdict['summary']['overall_decision']}`",
    f"- microbench rows: `{verdict['summary']['microbench_rows']}`",
    f"- raw microbench accepts: `{verdict['summary']['raw_microbench_accepts']}`",
    f"- strong raw microbench accepts: `{verdict['summary']['strong_raw_microbench_accepts']}`",
    f"- microbench invalid: `{verdict['summary']['microbench_invalid']}`",
    f"- full-decode candidates: `{verdict['summary']['full_decode_candidates']}`",
    f"- full-decode confirmed accepts: `{verdict['summary']['full_decode_confirmed_accepts']}`",
    f"- run 32B: `{verdict['summary']['run_32b']}`",
    "",
    "Reasons:",
    "",
  ]
  lines += [f"- {reason}" for reason in verdict["summary"]["reasons"]]
  lines += [
    "",
    "## Models",
    "",
    "| model | row | status | gain % | current GB/s | candidate GB/s | reasons |",
    "|---|---|---|---:|---:|---:|---|",
  ]
  for report in verdict["models"]:
    for row in report["rows"]:
      gain = row.get("gain")
      current = row.get("current_gbs")
      candidate = row.get("candidate_gbs")
      reasons = "; ".join(row.get("reasons") or []) or "none"
      lines.append(
        f"| {report['model']} | `{row['id']}` | `{row.get('status')}` | "
        f"{'n/a' if gain is None else f'{gain * 100:.2f}'} | "
        f"{'n/a' if current is None else f'{current:.2f}'} | "
        f"{'n/a' if candidate is None else f'{candidate:.2f}'} | {reasons} |"
      )
  lines += [
    "",
    "## Interpretation",
    "",
    "Family C v1 is accepted only if the aligned vector-load rewrite produces a",
    "strong raw microbench gain. A weak raw accept is not enough for full decode",
    "because single-tensor gains dilute at model scope. No accept means the next",
    "step is hardware-counter profiling or a deeper memory-layout/codegen",
    "capability, not another schedule-only variant.",
    "",
  ]
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Build semantic QK codegen v4 verdict")
  parser.add_argument("--base", type=pathlib.Path, required=True)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  args = parser.parse_args()
  verdict = build_verdict(args.base, repo=pathlib.Path.cwd())
  _write(args.json, verdict)
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(verdict_markdown(verdict))
  print(verdict_markdown(verdict))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
