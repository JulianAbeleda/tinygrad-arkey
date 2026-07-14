from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tinygrad.renderer.isa.amd import amd_isa_proof_manifest, reset_amd_isa_proof_manifest

SCHEMA = "tinygrad.amd_isa_proof_manifest.v1"
ROW_SCHEMA = "amd-isa-renderer-proof-manifest-row.v1"
REQUIRED_ROW_FIELDS = ("schema", "kind", "logical_op", "emitted")
DEFAULT_MAX_ROWS = 4096
OPERAND_PATH_FIELDS = frozenset(("operand_id", "source_operand_id", "fetch_group", "cache_policy", "width_bytes",
                                 "vector_width_bytes", "retained_fragment", "semantic_owner", "semantic_ownership"))


def _validate_non_empty_string(value: Any, path: str) -> None:
  if not isinstance(value, str) or value == "":
    raise ValueError(f"{path} must be a non-empty string")


def _validate_optional_sha256(value: Any, path: str) -> None:
  if value is None: return
  if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise ValueError(f"{path} must be a lowercase hex sha256 string")


def _validate_json_value(value: Any, path: str) -> None:
  if value is None or isinstance(value, (str, int, float, bool)): return
  if isinstance(value, Mapping):
    for key, item in value.items():
      _validate_non_empty_string(key, f"{path} key")
      _validate_json_value(item, f"{path}.{key}")
    return
  if isinstance(value, (list, tuple)):
    for idx, item in enumerate(value): _validate_json_value(item, f"{path}[{idx}]")
    return
  raise ValueError(f"{path} must contain only JSON-compatible values")


def _validate_metadata_mapping(value: Any, path: str) -> dict[str, Any]:
  if not isinstance(value, Mapping): raise ValueError(f"{path} must be a mapping")
  copied = dict(value)
  _validate_json_value(copied, path)
  return copied


def _validate_operand_path_fields(row: Mapping[str, Any], path: str) -> None:
  for field in ("operand_id", "source_operand_id"):
    if field in row: _validate_non_empty_string(row[field], f"{path}.{field}")
  for field in ("width_bytes", "vector_width_bytes"):
    if field in row and (not isinstance(row[field], int) or isinstance(row[field], bool) or row[field] <= 0):
      raise ValueError(f"{path}.{field} must be a positive integer")
  if "fetch_group" in row and (isinstance(row["fetch_group"], bool) or not isinstance(row["fetch_group"], (str, int)) or row["fetch_group"] == ""):
    raise ValueError(f"{path}.fetch_group must be a non-empty string or integer")
  for field in OPERAND_PATH_FIELDS - {"operand_id", "source_operand_id", "width_bytes", "vector_width_bytes", "fetch_group"}:
    if field in row: _validate_json_value(row[field], f"{path}.{field}")


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
    _validate_json_value(row, f"$.rows[{idx}]")
    _validate_operand_path_fields(row, f"$.rows[{idx}]")
    validated.append(dict(row))
  return tuple(validated)


def build_amd_isa_proof_manifest_bundle(
  *,
  candidate_id: str,
  kernel_name: str,
  rows: Iterable[Mapping[str, Any]],
  source_sha256: str | None = None,
  binary_sha256: str | None = None,
  abi_metadata: Mapping[str, Any] | None = None,
  ownership_metadata: Mapping[str, Any] | None = None,
  digest_metadata: Mapping[str, Any] | None = None,
  max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
  _validate_non_empty_string(candidate_id, "candidate_id")
  _validate_non_empty_string(kernel_name, "kernel_name")
  _validate_optional_sha256(source_sha256, "source_sha256")
  _validate_optional_sha256(binary_sha256, "binary_sha256")
  abi = None if abi_metadata is None else _validate_metadata_mapping(abi_metadata, "abi_metadata")
  ownership = None if ownership_metadata is None else _validate_metadata_mapping(ownership_metadata, "ownership_metadata")
  digests = None if digest_metadata is None else _validate_metadata_mapping(digest_metadata, "digest_metadata")
  if digests is not None:
    for name, digest in digests.items(): _validate_optional_sha256(digest, f"digest_metadata.{name}")
  manifest_rows = validate_amd_isa_proof_rows(rows, max_rows=max_rows)

  bundle: dict[str, Any] = {
    "schema": SCHEMA,
    "candidate_id": candidate_id,
    "kernel_name": kernel_name,
    "rows": list(manifest_rows),
  }
  if source_sha256 is not None: bundle["source_sha256"] = source_sha256
  if binary_sha256 is not None: bundle["binary_sha256"] = binary_sha256
  if abi is not None: bundle["abi_metadata"] = abi
  if ownership is not None: bundle["ownership_metadata"] = ownership
  if digests is not None: bundle["digest_metadata"] = digests
  return bundle


def export_current_amd_isa_proof_manifest_bundle(
  *,
  candidate_id: str,
  kernel_name: str,
  source_sha256: str | None = None,
  binary_sha256: str | None = None,
  abi_metadata: Mapping[str, Any] | None = None,
  ownership_metadata: Mapping[str, Any] | None = None,
  digest_metadata: Mapping[str, Any] | None = None,
  max_rows: int = DEFAULT_MAX_ROWS,
  reset_after: bool = False,
) -> dict[str, Any]:
  bundle = build_amd_isa_proof_manifest_bundle(
    candidate_id=candidate_id, kernel_name=kernel_name, rows=amd_isa_proof_manifest(),
    source_sha256=source_sha256, binary_sha256=binary_sha256, abi_metadata=abi_metadata,
    ownership_metadata=ownership_metadata, digest_metadata=digest_metadata, max_rows=max_rows)
  if reset_after: reset_amd_isa_proof_manifest()
  return bundle


def summarize_amd_isa_proof_rows(rows: Iterable[Mapping[str, Any]], *, max_rows: int = DEFAULT_MAX_ROWS) -> dict[str, Any]:
  """Count final renderer structures without assigning semantic ownership from physical operands."""
  manifest_rows = validate_amd_isa_proof_rows(rows, max_rows=max_rows)
  kinds = {"global_load": 0, "ds_load": 0, "ds_store": 0, "wait": 0, "barrier": 0, "wmma": 0}
  operand_paths: list[dict[str, Any]] = []
  for row_index, row in enumerate(manifest_rows):
    kind = row["kind"]
    if kind.startswith("global_load"): kinds["global_load"] += 1
    if kind.startswith("ds_load"): kinds["ds_load"] += 1
    if kind.startswith("ds_store"): kinds["ds_store"] += 1
    if kind in ("wait", "waitcnt") or "waitcnt" in row["emitted"]: kinds["wait"] += 1
    if kind == "barrier": kinds["barrier"] += 1
    if kind == "wmma": kinds["wmma"] += 1
    explicit = {key: row[key] for key in OPERAND_PATH_FIELDS if key in row}
    if explicit: operand_paths.append({"row_index": row_index, "kind": kind, **explicit})
  return {"schema": "tinygrad.amd_isa_structure_summary.v1", "row_count": len(manifest_rows),
          "counts": kinds, "operand_paths": operand_paths,
          "operand_ownership_authority": "explicit_compiler_metadata_only"}
