#!/usr/bin/env python3
"""Small, dispatch-free resource evidence surface for generated decode code objects.

The capture contract is intentionally independent of the GPU runtime: callers provide the
worker's final source/code-object bytes and launch geometry, and this module emits a stable
JSON row after checking expected kernel-name positive controls.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any, Iterable

from extra.qk.mmq_compile_evidence import parse_amdgpu_metadata

REQUIRED_RESOURCES = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills")


def _sha256(value: bytes | str) -> str:
  return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True)
class DecodeResourceRow:
  kernel_name: str
  source_sha256: str
  code_object_sha256: str
  source_bytes: int
  code_object_bytes: int
  vgpr: int
  sgpr: int
  lds_bytes: int
  scratch_bytes: int
  vgpr_spills: int
  sgpr_spills: int
  workgroup: tuple[int, int, int]
  grid: tuple[int, int, int]
  expected_name_matches: int

  def to_json(self) -> dict[str, Any]:
    row = asdict(self)
    row["workgroup"] = list(self.workgroup)
    row["grid"] = list(self.grid)
    return row


def capture_row(*, kernel_name: str, source: bytes | str, code_object: bytes,
                expected_names: Iterable[str], workgroup: tuple[int, int, int],
                grid: tuple[int, int, int], metadata: dict[str, Any] | None = None) -> DecodeResourceRow:
  """Build one evidence row. Empty captures, unknown resources, and missing names fail closed."""
  if not kernel_name or not source or not code_object: raise ValueError("source/code-object capture must be non-empty")
  matches = sum(kernel_name == name for name in expected_names)
  if matches < 1: raise ValueError(f"expected kernel-name positive control failed for {kernel_name!r}")
  facts = metadata if metadata is not None else parse_amdgpu_metadata(code_object)
  missing = [key for key in REQUIRED_RESOURCES if key not in facts]
  if missing: raise ValueError(f"resource metadata missing {missing}")
  return DecodeResourceRow(kernel_name, _sha256(source), _sha256(code_object),
                           len(source), len(code_object), *(int(facts[k]) for k in REQUIRED_RESOURCES),
                           tuple(workgroup), tuple(grid), matches)


def capture_report(rows: Iterable[DecodeResourceRow], *, geometry: dict[str, Any]) -> dict[str, Any]:
  rows = tuple(rows)
  if not rows: raise ValueError("empty resource capture")
  if any(row.expected_name_matches < 1 for row in rows): raise ValueError("resource capture lacks positive name match")
  return {"_schema": "decode-resource-capture.v1", "geometry": geometry,
          "rows": [row.to_json() for row in rows],
          "positive_expected_name_matches": sum(row.expected_name_matches for row in rows)}


def write_report(path: str, report: dict[str, Any]) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
