from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tinygrad.renderer.isa.amd import amd_isa_proof_manifest, reset_amd_isa_proof_manifest

SCHEMA = "tinygrad.amd_isa_proof_manifest.v1"
ROW_SCHEMA = "amd-isa-renderer-proof-manifest-row.v1"
REQUIRED_ROW_FIELDS = ("schema", "kind", "logical_op", "emitted")
DEFAULT_MAX_ROWS = 4096


def _validate_non_empty_string(value: Any, path: str) -> None:
  if not isinstance(value, str) or value == "":
    raise ValueError(f"{path} must be a non-empty string")


def _validate_optional_sha256(value: Any, path: str) -> None:
  if value is None: return
  if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise ValueError(f"{path} must be a lowercase hex sha256 string")


def validate_amd_isa_proof_rows(rows: Iterable[Mapping[str, Any]], *, max_rows: int = DEFAULT_MAX_ROWS) -> tuple[dict[str, Any], ...]:
  if not isinstance(max_rows, int) or max_rows < 0:
    raise ValueError("max_rows must be a non-negative integer")

  validated: list[dict[str, Any]] = []
  for idx, row in enumerate(rows):
    if idx >= max_rows:
      raise ValueError(f"rows exceeds max_rows={max_rows}")
    if not isinstance(row, Mapping):
      raise ValueError(f"$.rows[{idx}] must be a mapping")
    for field in REQUIRED_ROW_FIELDS:
      if field not in row:
        raise ValueError(f"$.rows[{idx}].{field} missing")
      _validate_non_empty_string(row[field], f"$.rows[{idx}].{field}")
    if row["schema"] != ROW_SCHEMA:
      raise ValueError(f"$.rows[{idx}].schema must be {ROW_SCHEMA}")
    validated.append(dict(row))
  return tuple(validated)


def build_amd_isa_proof_manifest_bundle(
  *,
  candidate_id: str,
  kernel_name: str,
  rows: Iterable[Mapping[str, Any]],
  source_sha256: str | None = None,
  binary_sha256: str | None = None,
  max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
  _validate_non_empty_string(candidate_id, "candidate_id")
  _validate_non_empty_string(kernel_name, "kernel_name")
  _validate_optional_sha256(source_sha256, "source_sha256")
  _validate_optional_sha256(binary_sha256, "binary_sha256")
  manifest_rows = validate_amd_isa_proof_rows(rows, max_rows=max_rows)

  bundle: dict[str, Any] = {
    "schema": SCHEMA,
    "candidate_id": candidate_id,
    "kernel_name": kernel_name,
    "rows": list(manifest_rows),
  }
  if source_sha256 is not None: bundle["source_sha256"] = source_sha256
  if binary_sha256 is not None: bundle["binary_sha256"] = binary_sha256
  return bundle


def export_current_amd_isa_proof_manifest_bundle(
  *,
  candidate_id: str,
  kernel_name: str,
  source_sha256: str | None = None,
  binary_sha256: str | None = None,
  max_rows: int = DEFAULT_MAX_ROWS,
  reset_after: bool = False,
) -> dict[str, Any]:
  bundle = build_amd_isa_proof_manifest_bundle(
    candidate_id=candidate_id, kernel_name=kernel_name, rows=amd_isa_proof_manifest(),
    source_sha256=source_sha256, binary_sha256=binary_sha256, max_rows=max_rows)
  if reset_after: reset_amd_isa_proof_manifest()
  return bundle
