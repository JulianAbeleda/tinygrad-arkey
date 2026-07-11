from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "tinygrad.mmq_timing_result.v1"
TIMING_STATUSES = ("measured", "blocked", "not_run", "oracle_only")
TOP_LEVEL_FIELDS = (
  "schema", "candidate_id", "backend", "shape", "comparator_id", "production_dispatch_changed",
  "timing_status", "timings_ms", "speedup_vs_comparator", "blockers",
)
SHAPE_FIELDS = ("M", "N", "K")


def _validate_non_empty_string(value: Any, path: str) -> None:
  if not isinstance(value, str) or value == "":
    raise ValueError(f"{path} must be a non-empty string")


def _validate_positive_int(value: Any, path: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{path} must be a positive integer")
  return value


def _validate_non_negative_number(value: Any, path: str) -> int | float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
    raise ValueError(f"{path} must be a non-negative number")
  return value


def _validate_positive_number(value: Any, path: str) -> int | float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{path} must be a positive number")
  return value


def _validate_shape(value: Any, path: str) -> dict[str, int]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{path} must be a dict")
  unknown = set(value) - set(SHAPE_FIELDS)
  if unknown:
    raise ValueError(f"{path} contains unknown fields: {sorted(unknown)}")
  missing = set(SHAPE_FIELDS) - set(value)
  if missing:
    raise ValueError(f"{path} missing required fields: {sorted(missing)}")
  return {field: _validate_positive_int(value[field], f"{path}.{field}") for field in SHAPE_FIELDS}


def _validate_timings_ms(value: Any, path: str) -> dict[str, int | float]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{path} must be a dict")
  if len(value) == 0:
    raise ValueError(f"{path} must not be empty")
  timings: dict[str, int | float] = {}
  for key, timing in value.items():
    _validate_non_empty_string(key, f"{path} key")
    timings[key] = _validate_non_negative_number(timing, f"{path}.{key}")
  return timings


def _validate_blockers(value: Any, path: str) -> list[str]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
    raise ValueError(f"{path} must be a sequence of strings")
  blockers: list[str] = []
  for idx, blocker in enumerate(value):
    _validate_non_empty_string(blocker, f"{path}[{idx}]")
    blockers.append(blocker)
  return blockers


def build_mmq_timing_result_bundle(
  *,
  candidate_id: str,
  backend: str,
  shape: Mapping[str, Any],
  comparator_id: str,
  timing_status: str,
  timings_ms: Mapping[str, int | float] | None = None,
  speedup_vs_comparator: int | float | None = None,
  blockers: Sequence[str] | None = None,
) -> dict[str, Any]:
  _validate_non_empty_string(candidate_id, "candidate_id")
  _validate_non_empty_string(backend, "backend")
  shape_out = _validate_shape(shape, "shape")
  _validate_non_empty_string(comparator_id, "comparator_id")
  if timing_status not in TIMING_STATUSES:
    raise ValueError(f"timing_status must be one of {TIMING_STATUSES}")

  bundle: dict[str, Any] = {
    "schema": SCHEMA,
    "candidate_id": candidate_id,
    "backend": backend,
    "shape": shape_out,
    "comparator_id": comparator_id,
    "production_dispatch_changed": False,
    "timing_status": timing_status,
  }
  if timings_ms is not None:
    bundle["timings_ms"] = _validate_timings_ms(timings_ms, "timings_ms")
  if speedup_vs_comparator is not None:
    bundle["speedup_vs_comparator"] = _validate_positive_number(speedup_vs_comparator, "speedup_vs_comparator")
  if blockers is not None:
    bundle["blockers"] = _validate_blockers(blockers, "blockers")
  return bundle


def validate_mmq_timing_result_bundle(bundle: Any) -> dict[str, Any]:
  if not isinstance(bundle, dict):
    raise ValueError("bundle must be a dict")
  unknown = set(bundle) - set(TOP_LEVEL_FIELDS)
  if unknown:
    raise ValueError(f"bundle contains unknown fields: {sorted(unknown)}")
  if bundle.get("schema") != SCHEMA:
    raise ValueError(f"schema must be {SCHEMA}")
  if bundle.get("production_dispatch_changed") is not False:
    raise ValueError("production_dispatch_changed must be False")

  required = ("candidate_id", "backend", "shape", "comparator_id", "timing_status")
  missing = [field for field in required if field not in bundle]
  if missing:
    raise ValueError(f"bundle missing required fields: {missing}")

  _validate_non_empty_string(bundle["candidate_id"], "candidate_id")
  _validate_non_empty_string(bundle["backend"], "backend")
  _validate_shape(bundle["shape"], "shape")
  _validate_non_empty_string(bundle["comparator_id"], "comparator_id")
  if bundle["timing_status"] not in TIMING_STATUSES:
    raise ValueError(f"timing_status must be one of {TIMING_STATUSES}")
  if "timings_ms" in bundle:
    _validate_timings_ms(bundle["timings_ms"], "timings_ms")
  if "speedup_vs_comparator" in bundle:
    _validate_positive_number(bundle["speedup_vs_comparator"], "speedup_vs_comparator")
  if "blockers" in bundle:
    _validate_blockers(bundle["blockers"], "blockers")
  return dict(bundle)


def build_mmq_timing_result_from_bounded_harness_report(report: Mapping[str, Any]) -> dict[str, Any]:
  if report.get("schema") != "q4k-q8-1-mmq-bounded-harness.v1":
    raise ValueError("report must be a q4k-q8-1-mmq-bounded-harness.v1 bundle")
  metadata = report.get("metadata")
  if not isinstance(metadata, Mapping):
    raise ValueError("report.metadata must be a dict")
  shape = metadata.get("bounded_shape")
  timing = report.get("timing")
  if not isinstance(timing, Mapping):
    raise ValueError("report.timing must be a dict")

  candidate_route_id = metadata.get("candidate_route_id")
  backend = metadata.get("backend")
  comparator_id = timing.get("comparator_id", metadata.get("comparator_id"))
  _validate_non_empty_string(candidate_route_id, "report.metadata.candidate_route_id")
  _validate_non_empty_string(backend, "report.metadata.backend")
  _validate_non_empty_string(comparator_id, "report.timing.comparator_id")

  blockers = report.get("blockers", [])
  shape_out = _validate_shape(shape, "report.metadata.bounded_shape")
  activation_layout = metadata.get("activation_layout")
  _validate_non_empty_string(activation_layout, "report.metadata.activation_layout")
  candidate_id = (
    f"{candidate_route_id}."
    f"{backend}.m{shape_out['M']}.n{shape_out['N']}.k{shape_out['K']}."
    f"{activation_layout}"
  )
  if blockers:
    timing_status = "blocked"
  elif backend == "llama_mmq_q4k_q8_1_coop_tile_oracle":
    timing_status = "oracle_only"
  elif "min_ms" in timing:
    timing_status = "measured"
  else:
    timing_status = "not_run"

  timings_ms: dict[str, int | float] = {}
  if "min_ms" in timing:
    timings_ms["candidate_min_ms"] = timing["min_ms"]
  direct = timing.get("direct_packed")
  if isinstance(direct, Mapping) and "min_ms" in direct:
    timings_ms["comparator_min_ms"] = direct["min_ms"]

  speedup = None
  if "candidate_min_ms" in timings_ms and "comparator_min_ms" in timings_ms and timings_ms["candidate_min_ms"] > 0:
    speedup = timings_ms["comparator_min_ms"] / timings_ms["candidate_min_ms"]

  return build_mmq_timing_result_bundle(
    candidate_id=candidate_id,
    backend=str(backend),
    shape=shape_out,
    comparator_id=str(comparator_id),
    timing_status=timing_status,
    timings_ms=timings_ms or None,
    speedup_vs_comparator=speedup,
    blockers=blockers if blockers else None,
  )
