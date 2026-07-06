"""Centralized completion criteria for exhaustive lowering.

This module tracks *how* we decide a route is "done" at each target
lowering level without repeating per-route rows. Route IDs remain in
`lowering_phase_registry.py` / `pure_kernel_surface_audit.py`.
"""

from __future__ import annotations

from typing import Any, TypedDict


class LoweringDoneCriteriaRow(TypedDict):
  target_lowering_level: str
  required_criteria: list[str]


VALID_LOWERING_LEVELS = ("L3", "L4", "L5")


GENERIC_COMPLETION_CRITERIA = (
  "semantic_correctness",
  "generated_only_surface",
  "route_binding",
  "no_hidden_fallback",
  "expected_kernel_binding",
  "strict_audit_status",
  "rollback_quarantine",
  "perf_gate_policy",
)


_LEVEL_CRITERIA: dict[str, tuple[str, ...]] = {
  "L3": (
    "semantic_correctness",
    "generated_only_surface",
    "route_binding",
    "no_hidden_fallback",
    "expected_kernel_binding",
    "strict_audit_status",
    "rollback_quarantine",
  ),
  "L4": (
    "semantic_correctness",
    "route_binding",
    "no_hidden_fallback",
    "expected_kernel_binding",
    "strict_audit_status",
    "rollback_quarantine",
    "perf_gate_policy",
  ),
  "L5": (
    "semantic_correctness",
    "route_binding",
    "no_hidden_fallback",
    "expected_kernel_binding",
    "strict_audit_status",
    "rollback_quarantine",
    "perf_gate_policy",
  ),
}


def _validate_level(level: str) -> None:
  if level not in VALID_LOWERING_LEVELS:
    raise ValueError(f"invalid lowering level {level!r}; expected one of {VALID_LOWERING_LEVELS}")


def _validate_rows(level_rows: dict[str, tuple[str, ...]]) -> None:
  if set(level_rows) != set(VALID_LOWERING_LEVELS):
    raise ValueError(
      f"done criteria levels must be exactly {VALID_LOWERING_LEVELS}; got {tuple(level_rows)}"
    )
  known_levels = set(VALID_LOWERING_LEVELS)
  required_criteria = set(GENERIC_COMPLETION_CRITERIA)
  for level, criteria in level_rows.items():
    if level not in known_levels:
      raise ValueError(f"unknown lowering level for done criteria {level!r}")
    if not isinstance(criteria, tuple):
      raise ValueError(f"criteria list for level {level!r} must be a tuple")
    seen = set[str]()
    for criterion in criteria:
      if criterion not in required_criteria:
        raise ValueError(f"invalid completion criterion {criterion!r} for level {level!r}")
      if criterion in seen:
        raise ValueError(f"duplicate criterion {criterion!r} for level {level!r}")
      seen.add(criterion)


def _sanitize_row(level: str) -> dict[str, Any]:
  if level not in _LEVEL_CRITERIA:
    raise ValueError(f"unknown lowering level for done criteria {level!r}")
  return {
    "target_lowering_level": level,
    "required_criteria": list(_LEVEL_CRITERIA[level]),
  }


def criteria_for_level(level: str) -> dict[str, Any]:
  _validate_level(level)
  return _sanitize_row(level)


def rows() -> list[dict[str, Any]]:
  return [_sanitize_row(level) for level in VALID_LOWERING_LEVELS]


def build() -> dict[str, Any]:
  all_rows = rows()
  by_level = {}
  criteria_coverage = {}
  for criterion in GENERIC_COMPLETION_CRITERIA:
    criteria_coverage[criterion] = sum(1 for row in all_rows if criterion in row["required_criteria"])

  for row in all_rows:
    by_level[row["target_lowering_level"]] = len(row["required_criteria"])

  return {
    "schema": "lowering-done-criteria.v1",
    "rows": all_rows,
    "by_level": by_level,
    "by_criterion": criteria_coverage,
  }


_validate_rows(_LEVEL_CRITERIA)
