from __future__ import annotations

from collections.abc import Sequence
from typing import Any


SCHEMA = "tinygrad.kernel_resource_trace.v1"
RESOURCE_FIELDS = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup", "grid", "occupancy")
_INTEGER_RESOURCE_FIELDS = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes")
_DIMENSION_RESOURCE_FIELDS = ("workgroup", "grid")


def _validate_non_empty_string(value: Any, path: str) -> None:
  if not isinstance(value, str) or value == "":
    raise ValueError(f"{path} must be a non-empty string")


def _validate_optional_sha256(value: Any, path: str) -> None:
  if value is None: return
  if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise ValueError(f"{path} must be a lowercase hex sha256 string")


def _validate_non_negative_int(value: Any, path: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f"{path} must be a non-negative integer")
  return value


def _validate_positive_int_sequence(value: Any, path: str) -> list[int]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
    raise ValueError(f"{path} must be a sequence of positive integers")
  if len(value) == 0:
    raise ValueError(f"{path} must not be empty")
  dims: list[int] = []
  for idx, dim in enumerate(value):
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
      raise ValueError(f"{path}[{idx}] must be a positive integer")
    dims.append(dim)
  return dims


def _validate_occupancy(value: Any, path: str) -> int | float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
    raise ValueError(f"{path} must be a non-negative number")
  return value


def build_kernel_resource_trace_bundle(
  *,
  candidate_id: str,
  kernel_name: str,
  source_sha256: str | None = None,
  binary_sha256: str | None = None,
  vgpr: int | None = None,
  sgpr: int | None = None,
  lds_bytes: int | None = None,
  scratch_bytes: int | None = None,
  workgroup: Sequence[int] | None = None,
  grid: Sequence[int] | None = None,
  occupancy: int | float | None = None,
) -> dict[str, Any]:
  _validate_non_empty_string(candidate_id, "candidate_id")
  _validate_non_empty_string(kernel_name, "kernel_name")
  _validate_optional_sha256(source_sha256, "source_sha256")
  _validate_optional_sha256(binary_sha256, "binary_sha256")

  resources: dict[str, Any] = {}
  for field, value in (("vgpr", vgpr), ("sgpr", sgpr), ("lds_bytes", lds_bytes), ("scratch_bytes", scratch_bytes)):
    if value is not None:
      resources[field] = _validate_non_negative_int(value, f"resources.{field}")
  for field, value in (("workgroup", workgroup), ("grid", grid)):
    if value is not None:
      resources[field] = _validate_positive_int_sequence(value, f"resources.{field}")
  if occupancy is not None:
    resources["occupancy"] = _validate_occupancy(occupancy, "resources.occupancy")

  bundle: dict[str, Any] = {
    "schema": SCHEMA,
    "candidate_id": candidate_id,
    "kernel_name": kernel_name,
  }
  if source_sha256 is not None: bundle["source_sha256"] = source_sha256
  if binary_sha256 is not None: bundle["binary_sha256"] = binary_sha256
  if resources: bundle["resources"] = resources
  return bundle


def validate_kernel_resource_trace_bundle(bundle: Any) -> dict[str, Any]:
  if not isinstance(bundle, dict):
    raise ValueError("bundle must be a dict")
  if bundle.get("schema") != SCHEMA:
    raise ValueError(f"schema must be {SCHEMA}")
  _validate_non_empty_string(bundle.get("candidate_id"), "candidate_id")
  _validate_non_empty_string(bundle.get("kernel_name"), "kernel_name")
  _validate_optional_sha256(bundle.get("source_sha256"), "source_sha256")
  _validate_optional_sha256(bundle.get("binary_sha256"), "binary_sha256")

  if "resources" in bundle:
    resources = bundle["resources"]
    if not isinstance(resources, dict):
      raise ValueError("resources must be a dict")
    unknown = set(resources) - set(RESOURCE_FIELDS)
    if unknown:
      raise ValueError(f"resources contains unknown fields: {sorted(unknown)}")
    for field in _INTEGER_RESOURCE_FIELDS:
      if field in resources: _validate_non_negative_int(resources[field], f"resources.{field}")
    for field in _DIMENSION_RESOURCE_FIELDS:
      if field in resources: _validate_positive_int_sequence(resources[field], f"resources.{field}")
    if "occupancy" in resources: _validate_occupancy(resources["occupancy"], "resources.occupancy")
  return dict(bundle)
