#!/usr/bin/env python3
"""Lifecycle counter schema/report helpers for future hybrid MMQ atoms.

This module is intentionally disconnected from route selection and GPU code. It
only validates and aggregates lifecycle counters that a Q4_K/Q8_1 or Q6_K/Q8_1
hybrid MMQ atom exporter can later emit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA = "hybrid-mmq-lifecycle-report.v1"

COUNTER_NAMES: tuple[str, ...] = (
  "activation_quant_epochs",
  "activation_q8_1_global_writes",
  "activation_q8_1_reads",
  "packed_weight_global_loads",
  "scale_min_metadata_loads",
  "dot_accumulation_epochs",
  "dot_ops_or_packed_dot_insts",
  "barriers",
  "intermediate_global_writes",
  "output_store_epochs",
  "output_stores",
  "duplicate_quant_work",
  "duplicate_dequant_or_scale_work",
  "split_k_reductions",
)

EPOCH_COUNTERS: tuple[str, ...] = (
  "activation_quant_epochs",
  "packed_weight_global_loads",
  "scale_min_metadata_loads",
  "dot_accumulation_epochs",
  "output_store_epochs",
)

DEFAULT_ROUTE_ID = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"


@dataclass(frozen=True)
class MMQLifecycleRow:
  role: str
  tile_id: str
  counters: Mapping[str, int]
  quant: str = "Q4_K"
  activation: str = "Q8_1"

  def to_json(self) -> dict[str, Any]:
    return {
      "role": self.role,
      "tile_id": self.tile_id,
      "quant": self.quant,
      "activation": self.activation,
      "counters": {name: self.counters[name] for name in COUNTER_NAMES if name in self.counters},
    }


def zero_counters(**overrides: int) -> dict[str, int]:
  counters = {name: 0 for name in COUNTER_NAMES}
  for name, value in overrides.items():
    if name not in counters:
      raise KeyError(f"unknown MMQ lifecycle counter {name!r}")
    counters[name] = value
  return counters


def _coerce_row(row: MMQLifecycleRow | Mapping[str, Any]) -> dict[str, Any]:
  return row.to_json() if isinstance(row, MMQLifecycleRow) else dict(row)


def validate_lifecycle_rows(rows: list[MMQLifecycleRow | Mapping[str, Any]]) -> list[str]:
  errors: list[str] = []
  for idx, input_row in enumerate(rows):
    row = _coerce_row(input_row)
    path = f"$.tiles[{idx}]"
    if not isinstance(row.get("role"), str) or not row.get("role"):
      errors.append(f"{path}.role must be a non-empty string")
    if not isinstance(row.get("tile_id"), str) or not row.get("tile_id"):
      errors.append(f"{path}.tile_id must be a non-empty string")
    counters = row.get("counters")
    if not isinstance(counters, Mapping):
      errors.append(f"{path}.counters must be an object")
      continue
    for name in COUNTER_NAMES:
      if name not in counters:
        errors.append(f"{path}.counters.{name} missing")
        continue
      value = counters[name]
      if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{path}.counters.{name} must be an integer")
      elif value < 0:
        errors.append(f"{path}.counters.{name} must be non-negative")
  return errors


def _sum_counters(rows: list[dict[str, Any]]) -> dict[str, int]:
  totals = {name: 0 for name in COUNTER_NAMES}
  for row in rows:
    counters = row["counters"]
    for name in COUNTER_NAMES:
      totals[name] += int(counters[name])
  return totals


def aggregate_lifecycle_rows(rows: list[MMQLifecycleRow | Mapping[str, Any]]) -> dict[str, Any]:
  errors = validate_lifecycle_rows(rows)
  if errors:
    raise ValueError("; ".join(errors))
  normalized = [_coerce_row(row) for row in rows]
  by_role: dict[str, list[dict[str, Any]]] = {}
  by_tile: dict[str, list[dict[str, Any]]] = {}
  for row in normalized:
    by_role.setdefault(str(row["role"]), []).append(row)
    by_tile.setdefault(str(row["tile_id"]), []).append(row)
  return {
    "total": _sum_counters(normalized),
    "by_role": {role: _sum_counters(role_rows) for role, role_rows in sorted(by_role.items())},
    "by_tile": {tile: _sum_counters(tile_rows) for tile, tile_rows in sorted(by_tile.items())},
  }


def build_lifecycle_report(rows: list[MMQLifecycleRow | Mapping[str, Any]], *, route_id: str = DEFAULT_ROUTE_ID,
                           atom: str = "q4k_q8_1_hybrid_mmq", validate: bool = True) -> dict[str, Any]:
  normalized = [_coerce_row(row) for row in rows]
  errors = validate_lifecycle_rows(normalized)
  report = {
    "schema": SCHEMA,
    "route_id": route_id,
    "atom": atom,
    "counter_names": list(COUNTER_NAMES),
    "epoch_counters": list(EPOCH_COUNTERS),
    "ok": not errors,
    "tiles": normalized,
    "aggregation": aggregate_lifecycle_rows(normalized) if not errors else None,
    "errors": errors,
    "notes": (
      "Lifecycle counter report only; it does not select routes, launch kernels, "
      "or prove GPU correctness."
    ),
  }
  if validate and errors:
    raise ValueError("; ".join(errors))
  return report
