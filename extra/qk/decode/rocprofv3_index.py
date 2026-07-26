#!/usr/bin/env python3
"""Fail-closed indexer for ROCProfiler SDK 1.1.0 dispatch evidence.

This module never launches a profiler or a workload.  It indexes a completed
rocprofv3 CSV, JSON, or SQLite rocpd artifact only when its sidecar manifest
binds the artifact hash to an explicit positive control.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, pathlib, sqlite3
from collections import Counter
from typing import Any, Iterable

SCHEMA = "luna-rocprofv3-evidence.v1"
SUPPORTED_FORMATS = frozenset(("csv", "json", "rocpd"))
REQUIRED_MANIFEST = frozenset(("schema", "profiler", "profiler_version", "output_format", "trace_path",
                               "trace_sha256", "expected_kernel_name", "positive_control_expected_matches"))


def sha256(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
  return digest.hexdigest()


def _name_key(row: dict[str, Any]) -> str | None:
  normalized = {str(key).strip().lower().replace(" ", "_"): value for key, value in row.items()}
  for key in ("kernel_name", "kernel", "demangled_kernel_name", "formatted_kernel_name"):
    value = normalized.get(key)
    if isinstance(value, str) and value: return value
  return None


def _json_rows(value: Any) -> list[dict[str, Any]]:
  # Accept only the documented, unambiguous container names. Do not recursively
  # search arbitrary JSON because API and marker records can also contain names.
  if isinstance(value, list): rows = value
  elif isinstance(value, dict): rows = value.get("kernel_dispatches", value.get("kernel_dispatch"))
  else: rows = None
  if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
    raise ValueError("JSON trace must be a dispatch list or contain a dispatch-list key")
  return rows


def _rocpd_rows(path: pathlib.Path) -> list[dict[str, Any]]:
  # Some exporters materialize a dispatch view with the kernel name. Requiring
  # that projection is deliberate: guessing joins across evolving rocpd schemas
  # would turn an evidence parser into an inference engine.
  with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    if "rocpd_kernel_dispatch" not in tables:
      raise ValueError("rocpd artifact lacks rocpd_kernel_dispatch")
    columns = [row[1] for row in db.execute("PRAGMA table_info(rocpd_kernel_dispatch)")]
    name_column = next((col for col in columns if col.lower() in
                        ("kernel_name", "kernel", "demangled_kernel_name", "formatted_kernel_name")), None)
    if name_column is None:
      raise ValueError("rocpd dispatch relation has no materialized kernel-name column; export a named dispatch CSV")
    return [{"kernel_name": row[0]} for row in db.execute(
      f'SELECT "{name_column.replace(chr(34), chr(34) * 2)}" FROM rocpd_kernel_dispatch')]


def dispatch_names(path: pathlib.Path, output_format: str) -> list[str]:
  if output_format == "csv":
    with path.open(newline="", encoding="utf-8") as handle: rows: Iterable[dict[str, Any]] = list(csv.DictReader(handle))
  elif output_format == "json": rows = _json_rows(json.loads(path.read_text(encoding="utf-8")))
  elif output_format == "rocpd": rows = _rocpd_rows(path)
  else: raise ValueError(f"unsupported output_format: {output_format}")
  names = [name for row in rows if (name := _name_key(row)) is not None]
  if not names: raise ValueError("trace contains no named kernel dispatch rows")
  return names


def index_evidence(manifest_path: str | pathlib.Path) -> dict[str, Any]:
  manifest_file = pathlib.Path(manifest_path).resolve()
  manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
  if not isinstance(manifest, dict): raise ValueError("evidence manifest must be a JSON object")
  missing = sorted(REQUIRED_MANIFEST - manifest.keys())
  if missing: raise ValueError(f"evidence manifest missing required fields: {missing}")
  if manifest["schema"] != SCHEMA: raise ValueError("unsupported evidence manifest schema")
  if manifest["profiler"] != "rocprofv3" or manifest["profiler_version"] != "1.1.0":
    raise ValueError("evidence must be pinned to rocprofv3 1.1.0")
  output_format = manifest["output_format"]
  if output_format not in SUPPORTED_FORMATS: raise ValueError("unsupported rocprofv3 output format")
  expected = manifest["expected_kernel_name"]
  required_matches = manifest["positive_control_expected_matches"]
  if not isinstance(expected, str) or not expected or not isinstance(required_matches, int) or required_matches < 1:
    raise ValueError("positive control requires a non-empty exact name and positive expected match count")
  trace = pathlib.Path(manifest["trace_path"])
  if not trace.is_absolute(): trace = manifest_file.parent / trace
  trace = trace.resolve()
  if not trace.is_file(): raise ValueError("trace artifact does not exist")
  if sha256(trace) != manifest["trace_sha256"]: raise ValueError("trace_sha256 does not match artifact")
  names = dispatch_names(trace, output_format)
  counts = Counter(names)
  observed = counts[expected]
  if observed != required_matches:
    raise ValueError(f"positive control mismatch for {expected!r}: expected {required_matches}, observed {observed}")
  return {"schema": "luna-rocprofv3-index.v1", "manifest_path": str(manifest_file), "trace_path": str(trace),
          "output_format": output_format, "expected_kernel_name": expected,
          "positive_control_expected_matches": required_matches, "positive_control_observed_matches": observed,
          "kernel_dispatch_counts": dict(sorted(counts.items()))}


def main() -> None:
  parser = argparse.ArgumentParser(description="Index a positive-control-bound rocprofv3 artifact")
  parser.add_argument("--manifest", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  result = index_evidence(args.manifest)
  pathlib.Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
