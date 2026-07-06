#!/usr/bin/env python3
"""Compact CLI report for prefill performance-lowering phase tracking (scaffold).

This module intentionally avoids tinygrad/runtime imports and model loading.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from extra.qk import prefill_performance_lowering_registry as registry

SCHEMA = "prefill-performance-lowering-report.v1"


def _target_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  targets: dict[str, dict[str, Any]] = {}
  for row in rows:
    target = row["target"]
    target_data = targets.setdefault(target, {
      "target_label": row.get("target_label", target),
      "rows": [],
      "max_phase": -1,
      "blocking_count": 0,
    })
    target_data["rows"].append((row["phase_order"], row["id"]))
    target_data["max_phase"] = max(target_data["max_phase"], row["phase"])
    if row["status"] == "blocked":
      target_data["blocking_count"] += 1
  for target_id, target_data in targets.items():
    target_data["rows"] = [row_id for _, row_id in sorted(target_data["rows"])]
    target_data["done"] = all(row["status"] == "done" for row in rows if row["target"] == target_id)
  return targets


def build_prefill_performance_lowering_report(target: str | None = None) -> dict[str, Any]:
  all_rows = registry.rows()
  rows = all_rows
  if target is not None:
    rows = [r for r in all_rows if r["target"] == target]
  target_summary = _target_summary(rows)

  by_status: dict[str, int] = {}
  blockers: list[str] = []
  reusable_files: set[str] = set()
  for row in rows:
    by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    reusable_files.update(row["reuse_files"])
    if row["status"] != "done" and row["blockers"]:
      for blocker in row["blockers"]:
        blockers.append(f"{row['id']}: {blocker}")

  return {
    "schema": SCHEMA,
    "scope_doc": registry.DOC_PATH,
    "rows": rows,
    "row_count": len(rows),
    "target_count": len(target_summary),
    "targets": target_summary,
    "by_status": by_status,
    "scope_files": sorted(reusable_files),
    "blocker_list": blockers,
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description="Build prefill performance lowering registry report.")
  ap.add_argument("--compact", action="store_true", help="print compact JSON")
  ap.add_argument("--target", help="filter rows by target id (target_1 or target_2)")
  args = ap.parse_args(argv)

  report = build_prefill_performance_lowering_report(args.target)
  indent = None if args.compact else 2
  print(json.dumps(report, indent=indent))
  return report


if __name__ == "__main__":
  main()
