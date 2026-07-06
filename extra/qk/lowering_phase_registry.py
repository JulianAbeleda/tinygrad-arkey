"""Scaffold registry for handwritten-kernel lowering phases.

This registry tracks a roadmap for lowering blockers and does not duplicate route or runtime
surface metadata; it only stores the shared phase metadata needed by downstream tooling.
"""

from __future__ import annotations

from typing import Any, TypedDict

from extra.qk import route_manifest, runtime_surface_registry


class LoweringPhaseRegistryRow(TypedDict):
  id: str
  phase: int
  phase_name: str
  target_lowering_level: str
  next_action: str


_PHASE_ROWS: tuple[LoweringPhaseRegistryRow, ...] = ()


_KNOWN_IDS = set(route_manifest.ROUTES) | set(runtime_surface_registry.surface_ids())
_VALID_LEVELS = {"L3", "L4", "L5"}
_VALID_PHASES = tuple(range(1, 6))


def _validate_rows(rows: tuple[LoweringPhaseRegistryRow, ...]) -> None:
  seen = set[str]()
  for row in rows:
    if row["id"] in seen:
      raise ValueError(f"duplicate lowering phase row id {row['id']!r}")
    if row["target_lowering_level"] not in _VALID_LEVELS:
      raise ValueError(f"invalid target lowering level for {row['id']!r}: {row['target_lowering_level']!r}")
    if row["phase"] not in _VALID_PHASES:
      raise ValueError(f"invalid phase for {row['id']!r}: {row['phase']!r}")
    if row["id"] not in _KNOWN_IDS:
      raise ValueError(f"unknown lowering phase id {row['id']!r}")
    seen.add(row["id"])


def _sanitize_row(row: LoweringPhaseRegistryRow) -> dict[str, Any]:
  return {
    "id": row["id"],
    "phase": row["phase"],
    "phase_name": row["phase_name"],
    "target_lowering_level": row["target_lowering_level"],
    "next_action": row["next_action"],
  }


_validate_rows(_PHASE_ROWS)


def ids() -> tuple[str, ...]:
  return tuple(row["id"] for row in _PHASE_ROWS)


def rows() -> list[dict[str, Any]]:
  return [_sanitize_row(r) for r in _PHASE_ROWS]


def row(row_id: str) -> dict[str, Any]:
  for r in _PHASE_ROWS:
    if r["id"] == row_id:
      return _sanitize_row(r)
  raise KeyError(f"unknown lowering phase row id {row_id!r}")


def build() -> dict[str, Any]:
  all_rows = rows()
  by_level = {}
  for r in all_rows:
    by_level.setdefault(r["target_lowering_level"], 0)
    by_level[r["target_lowering_level"]] += 1
  by_phase = {}
  for r in all_rows:
    by_phase.setdefault(r["phase"], 0)
    by_phase[r["phase"]] += 1
  return {
    "schema": "lowering-phase-registry.v1",
    "total_rows": len(all_rows),
    "rows": all_rows,
    "by_level": by_level,
    "by_phase": by_phase,
  }
