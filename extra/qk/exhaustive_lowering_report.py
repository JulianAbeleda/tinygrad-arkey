#!/usr/bin/env python3
"""Exhaustive lowering report for the QK strict lowering debt queue.

This scaffold wraps ``pure_kernel_surface_audit.build()`` and exports a single JSON
artifact.  No route facts are duplicated here; all route facts come from the existing
audit/registry data.
"""
from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from extra.qk import pure_kernel_surface_audit as audit


SCHEMA = "exhaustive-lowering-report.v1"


def _load_phase_lookup() -> dict[str, dict[str, Any]]:
  try:
    phase_registry = importlib.import_module("extra.qk.lowering_phase_registry")
  except ModuleNotFoundError:
    return {}

  rows_fn = getattr(phase_registry, "rows", None)
  if not callable(rows_fn):
    return {}

  phase_rows = rows_fn()
  if not isinstance(phase_rows, (list, tuple)):
    return {}

  by_id: dict[str, dict[str, Any]] = {}
  for row in phase_rows:
    if not isinstance(row, dict):
      continue
    row_id = row.get("id")
    surface_id = row.get("surface_id")
    route_id = row.get("route_id")
    ident = (
      row_id if isinstance(row_id, str)
      else route_id if isinstance(route_id, str)
      else surface_id if isinstance(surface_id, str)
      else None
    )
    if ident is None:
      continue
    by_id[ident] = row
  return by_id


_PHASE_METADATA_KEYS = ("phase", "phase_name", "target_lowering_level", "next_action")


def _enrich_with_phase(row: dict[str, Any], phase_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
  if not row:
    return row
  item_id = row.get("route_id") or row.get("surface_id") or row.get("work_item_id")
  if not isinstance(item_id, str):
    return row
  phase_row = phase_lookup.get(item_id)
  if not phase_row:
    return row

  out = dict(row)
  for key in _PHASE_METADATA_KEYS:
    if key in phase_row:
      out[key] = phase_row[key]
  return out


def _strict_default_blocker_rows(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
  return [r for r in audit_report.get("strict_default_purity", {}).get("blockers", []) if isinstance(r, dict)]


def _unmanifested_runtime_rows(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
  return [r for r in audit_report.get("unmanifested_runtime_surfaces", []) if isinstance(r, dict)]


def _work_queue(
  strict_rows: list[dict[str, Any]],
  runtime_rows: list[dict[str, Any]],
  phase_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
  work_queue: list[dict[str, Any]] = []
  queued_ids: set[str] = set()
  for row in strict_rows:
    entry = dict(row)
    entry["work_item_type"] = "strict_default_route_blocker"
    entry["work_item_id"] = entry.get("route_id", "")
    if entry["work_item_id"]:
      queued_ids.add(str(entry["work_item_id"]))
      work_queue.append(_enrich_with_phase(entry, phase_lookup))

  for row in runtime_rows:
    entry = dict(row)
    entry["work_item_type"] = "unmanifested_runtime_surface"
    entry["work_item_id"] = entry.get("surface_id", "")
    if entry["work_item_id"]:
      queued_ids.add(str(entry["work_item_id"]))
      work_queue.append(_enrich_with_phase(entry, phase_lookup))

  for item_id, phase_row in phase_lookup.items():
    if item_id in queued_ids:
      continue
    entry = {"work_item_type": "phase_registry_item", "work_item_id": item_id}
    work_queue.append(_enrich_with_phase(entry, phase_lookup))

  return work_queue


def build_exhaustive_lowering_report() -> dict[str, Any]:
  audit_report = audit.build()
  strict_rows = _strict_default_blocker_rows(audit_report)
  runtime_rows = _unmanifested_runtime_rows(audit_report)
  strict_blocker_ids = [row["route_id"] for row in strict_rows if isinstance(row.get("route_id"), str)]
  runtime_blocker_ids = [row["surface_id"] for row in runtime_rows if isinstance(row.get("surface_id"), str)]

  phase_lookup = _load_phase_lookup()
  blockers = dict(audit_report.get("audit_blockers", {}))
  blockers.setdefault("strict_default_route_blockers", sorted(strict_blocker_ids))
  blockers.setdefault("unmanifested_runtime_surfaces", sorted(runtime_blocker_ids))

  return {
    "schema": SCHEMA,
    "audit_verdict": audit_report.get("verdict", "UNKNOWN"),
    "blockers": blockers,
    "work_queue": _work_queue(strict_rows, runtime_rows, phase_lookup),
    "audit_report": audit_report,
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description="Build an exhaustive lowering debt report.")
  ap.add_argument("--compact", action="store_true", help="print compact JSON")
  args = ap.parse_args(argv)

  report = build_exhaustive_lowering_report()
  indent = 2 if not args.compact else None
  print(json.dumps(report, indent=indent))
  return report


if __name__ == "__main__":
  main()
