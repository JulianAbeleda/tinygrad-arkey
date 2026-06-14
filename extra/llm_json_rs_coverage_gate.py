#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

def load_summary(path:pathlib.Path) -> dict[str, Any]:
  data = json.loads(path.read_text())
  if data.get("kind") != "llm_json_rejection_sample_summary":
    raise ValueError(f"{path}: expected kind=llm_json_rejection_sample_summary")
  if not isinstance(data.get("categories"), dict) or not data["categories"]:
    raise ValueError(f"{path}: expected non-empty categories")
  return data

def build_report(summary:dict[str, Any], *, categories:list[str], min_selected:int) -> dict[str, Any]:
  if min_selected < 1: raise ValueError("--min-selected must be >= 1")
  if not categories: categories = sorted(summary["categories"])
  rows: dict[str, dict[str, Any]] = {}
  failures: list[str] = []
  for category in categories:
    src = summary["categories"].get(category)
    if src is None:
      rows[category] = {"status": "fail", "reason": "missing category", "selected_train_rows": 0, "accepted_attempts": 0, "attempts": 0, "near_miss": 0}
      failures.append(f"{category}: missing category")
      continue
    selected = int(src.get("selected_train_rows", 0))
    status = "pass" if selected >= min_selected else "fail"
    reason = "" if status == "pass" else f"selected_train_rows {selected} < {min_selected}"
    rows[category] = {
      "status": status,
      "reason": reason,
      "attempts": int(src.get("attempts", 0)),
      "accepted_attempts": int(src.get("accepted_attempts", 0)),
      "near_miss": int(src.get("near_miss", 0)),
      "selected_train_rows": selected,
    }
    if reason: failures.append(f"{category}: {reason}")
  return {
    "kind": "llm_json_rs_coverage_gate",
    "status": "pass" if not failures else "fail",
    "min_selected": min_selected,
    "categories": rows,
    "failures": failures,
    "summary": {
      "artifact": summary.get("artifact"),
      "attempts": summary.get("attempts"),
      "accepted_attempts": summary.get("accepted_attempts"),
      "selected_train_rows": summary.get("selected_train_rows"),
      "sft_rows": summary.get("sft_rows"),
    },
  }

def markdown(report:dict[str, Any]) -> str:
  lines = [
    "# JSON RS Coverage Gate",
    "",
    f"- status: `{report['status']}`",
    f"- min selected train rows per category: `{report['min_selected']}`",
    "",
    "| category | status | selected | accepted attempts | attempts | near miss | reason |",
    "|---|---|---:|---:|---:|---:|---|",
  ]
  for category, row in report["categories"].items():
    lines.append(
      f"| `{category}` | `{row['status']}` | {row['selected_train_rows']} | "
      f"{row['accepted_attempts']} | {row['attempts']} | {row['near_miss']} | {row['reason']} |"
    )
  if report["failures"]:
    lines += ["", "## Failures", ""]
    lines += [f"- {failure}" for failure in report["failures"]]
  lines.append("")
  return "\n".join(lines)

def main() -> int:
  parser = argparse.ArgumentParser(description="Gate strict-JSON rejection-sampling category coverage")
  parser.add_argument("summary", type=pathlib.Path)
  parser.add_argument("--categories", nargs="*", default=[])
  parser.add_argument("--min-selected", type=int, default=20)
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()
  report = build_report(load_summary(args.summary), categories=args.categories, min_selected=args.min_selected)
  if args.json is not None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  if args.md is not None:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(markdown(report))
  print(markdown(report))
  return 0 if report["status"] == "pass" else 1

if __name__ == "__main__":
  raise SystemExit(main())
