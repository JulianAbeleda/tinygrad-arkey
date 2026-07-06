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
_PROMOTION_PHASE_NAME = "promotion"
_SIDECAR_NOTE_MARKERS = ("optional", "only needed if", "only needed")
_EVIDENCE_NOTE_MARKERS = ("passes", "can emit", "supports numerically correct", "probe exists", "contract exists")
_ACTIVE_BLOCKER_MARKERS = (
  "blocked", "does not", "fails", "failure", "missing", "must", "needs", "not ", "no ", "wrong", "without",
)


def _is_promotion_row(row: dict[str, Any]) -> bool:
  return row["phase_name"] == _PROMOTION_PHASE_NAME or row["id"].endswith("_promotion")


def _row_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
  return (row["target"], row["phase_order"], row["id"])


def _classify_note(note: str, status: str) -> str:
  lower = note.lower()
  if any(marker in lower for marker in _SIDECAR_NOTE_MARKERS):
    return "sidecar"
  if any(marker in lower for marker in _EVIDENCE_NOTE_MARKERS) and not any(marker in lower for marker in _ACTIVE_BLOCKER_MARKERS):
    return "evidence"
  if status == "done":
    return "evidence"
  return "active_blocker"


def _notes_by_type(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
  out = {"active_blocker": [], "evidence": [], "sidecar": []}
  for row in rows:
    for note in row["blockers"]:
      out[_classify_note(note, row["status"])].append(f"{row['id']}: {note}")
  return out


def _by_owner_area(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  by_owner: dict[str, dict[str, Any]] = {}
  for row in rows:
    owner_area = row["owner_area"]
    owner_data = by_owner.setdefault(owner_area, {
      "rows": [],
      "row_count": 0,
      "gates": set(),
      "status_counts": {},
      "active_blocker_count": 0,
      "evidence_note_count": 0,
      "sidecar_blocker_count": 0,
      "parallel_ready_rows": [],
      "blocked_rows": [],
    })
    owner_data["rows"].append((row["target"], row["phase_order"], row["id"]))
    owner_data["row_count"] += 1
    owner_data["gates"].update(row["gates"])
    owner_data["status_counts"][row["status"]] = owner_data["status_counts"].get(row["status"], 0) + 1

    row_note_types = [_classify_note(note, row["status"]) for note in row["blockers"]]
    if "active_blocker" in row_note_types:
      owner_data["active_blocker_count"] += row_note_types.count("active_blocker")
      owner_data["blocked_rows"].append((row["target"], row["phase_order"], row["id"]))
    owner_data["evidence_note_count"] += row_note_types.count("evidence")
    owner_data["sidecar_blocker_count"] += row_note_types.count("sidecar")

    if row["status"] in {"pending", "not_started"} and "active_blocker" not in row_note_types:
      owner_data["parallel_ready_rows"].append((row["target"], row["phase_order"], row["id"]))

  for owner_data in by_owner.values():
    owner_data["rows"] = [row_id for _, _, row_id in sorted(owner_data["rows"])]
    owner_data["parallel_ready_rows"] = [row_id for _, _, row_id in sorted(owner_data["parallel_ready_rows"])]
    owner_data["blocked_rows"] = [row_id for _, _, row_id in sorted(owner_data["blocked_rows"])]
    owner_data["gates"] = sorted(owner_data["gates"])

  return by_owner


def _gates_to_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
  gate_rows: dict[str, list[str]] = {}
  for row in rows:
    for gate in row["gates"]:
      gate_rows.setdefault(gate, []).append((row["target"], row["phase_order"], row["id"]))
  return {
    gate: [row_id for _, _, row_id in sorted(entries)]
    for gate, entries in sorted(gate_rows.items(), key=lambda kv: kv[0])
  }


def _orchestration(rows: list[dict[str, Any]]) -> dict[str, Any]:
  notes_by_type = _notes_by_type(rows)
  by_owner = _by_owner_area(rows)
  gate_rows = _gates_to_rows(rows)
  parallel_rows: list[tuple[str, int, str]] = []
  active_blocker_rows: list[tuple[str, int, str]] = []
  status_blocked_rows: list[tuple[str, int, str]] = []

  for row in rows:
    has_active_blocker = any(_classify_note(note, row["status"]) == "active_blocker" for note in row["blockers"])
    if has_active_blocker:
      active_blocker_rows.append((row["target"], row["phase_order"], row["id"]))
    if row["status"] == "blocked":
      status_blocked_rows.append((row["target"], row["phase_order"], row["id"]))
    if row["status"] in {"pending", "not_started"} and not has_active_blocker:
      parallel_rows.append((row["target"], row["phase_order"], row["id"]))

  parallel_rows = sorted(parallel_rows)
  active_blocker_rows = sorted(active_blocker_rows)
  status_blocked_rows = sorted(status_blocked_rows)

  return {
    "by_owner_area": by_owner,
    "gates": gate_rows,
    "notes": notes_by_type,
    "parallel_ready_rows": [row_id for _, _, row_id in parallel_rows],
    "active_blocker_rows": [row_id for _, _, row_id in active_blocker_rows],
    "status_blocked_rows": [row_id for _, _, row_id in status_blocked_rows],
    "summary": {
      "active_blocker_count": len(notes_by_type["active_blocker"]),
      "evidence_note_count": len(notes_by_type["evidence"]),
      "sidecar_blocker_count": len(notes_by_type["sidecar"]),
      "parallel_ready_count": len(parallel_rows),
      "owner_areas": sorted(by_owner),
    },
  }


def _target_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  targets: dict[str, dict[str, Any]] = {}
  for row in rows:
    target = row["target"]
    target_data = targets.setdefault(target, {
      "target_label": row.get("target_label", target),
      "rows": [],
      "max_phase": -1,
      "blocking_count": 0,
      "completion_percent_sum": 0,
    })
    target_data["rows"].append((row["phase_order"], row["id"]))
    target_data["max_phase"] = max(target_data["max_phase"], row["phase"])
    target_data["completion_percent_sum"] += row["completion_percent"]
    if row["status"] == "blocked":
      target_data["blocking_count"] += 1
  for target_id, target_data in targets.items():
    target_data["rows"] = [row_id for _, row_id in sorted(target_data["rows"])]
    row_count = len(target_data["rows"])
    target_data["average_completion_percent"] = round(target_data.pop("completion_percent_sum") / row_count, 1) if row_count else 0.0
    target_data["done"] = all(row["status"] == "done" for row in rows if row["target"] == target_id)
  return targets


def build_prefill_performance_lowering_report(
  target: str | None = None,
  pre_promotion_only: bool = False,
) -> dict[str, Any]:
  all_rows = registry.rows()
  rows = all_rows
  if target is not None:
    valid_targets = sorted({row["target"] for row in all_rows})
    if target not in valid_targets:
      raise ValueError(f"unknown prefill performance target {target!r}; expected one of {valid_targets}")
    rows = [r for r in all_rows if r["target"] == target]
  rows = sorted(rows, key=_row_sort_key)
  promotion_rows = [r for r in rows if _is_promotion_row(r)]
  if pre_promotion_only:
    rows = [r for r in rows if not _is_promotion_row(r)]
  target_summary = _target_summary(rows)

  by_status: dict[str, int] = {}
  blockers: list[str] = []
  reusable_files: set[str] = set()
  average_completion_percent = round(sum(row["completion_percent"] for row in rows) / len(rows), 1) if rows else 0.0
  for row in rows:
    by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    reusable_files.update(row["reuse_files"])
    if row["status"] != "done" and row["blockers"]:
      for blocker in row["blockers"]:
        blockers.append(f"{row['id']}: {blocker}")

  return {
    "schema": SCHEMA,
    "scope_doc": registry.DOC_PATH,
    "pre_promotion_only": pre_promotion_only,
    "row_scope": "pre_promotion" if pre_promotion_only else "full",
    "promotion_rows": [row["id"] for row in promotion_rows],
    "rows": rows,
    "row_count": len(rows),
    "average_completion_percent": average_completion_percent,
    "target_count": len(target_summary),
    "targets": target_summary,
    "by_status": by_status,
    "scope_files": sorted(reusable_files),
    "blocker_list": blockers,
    "orchestration": _orchestration(rows),
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description="Build prefill performance lowering registry report.")
  ap.add_argument("--compact", action="store_true", help="print compact JSON")
  ap.add_argument("--target", help="filter rows by target id (target_1 or target_2)")
  ap.add_argument(
    "--pre-promotion",
    action="store_true",
    help="exclude promotion rows for pre-promotion completion tracking",
  )
  ap.add_argument("--orchestration", action="store_true", help="print orchestration slice only")
  args = ap.parse_args(argv)

  report = build_prefill_performance_lowering_report(
    args.target,
    pre_promotion_only=args.pre_promotion,
  )
  indent = None if args.compact else 2
  payload = report["orchestration"] if args.orchestration else report
  print(json.dumps(payload, indent=indent))
  return report


if __name__ == "__main__":
  main()
