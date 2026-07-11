#!/usr/bin/env python3
"""Truthful gfx11 counter/tool capability discovery for bounded MMQ research."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

SCHEMA = "tinygrad.amd_counter_capability.v1"
STATUSES = frozenset(("advertised", "live", "zero_suspect", "unsupported", "blocked"))
DEFAULT_ROCM_ROOT = Path("/opt/rocm-7.2.4")


def _tool(name: str, rocm_root: Path) -> Path | None:
  found = shutil.which(name)
  if found: return Path(found)
  candidate = rocm_root / "bin" / name
  return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def _run(argv: list[str], timeout: int = 15) -> dict[str, Any]:
  try:
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"status": "live" if proc.returncode == 0 else "blocked", "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}
  except FileNotFoundError as exc:
    return {"status": "unsupported", "error": str(exc)}
  except (subprocess.TimeoutExpired, OSError) as exc:
    return {"status": "blocked", "error": str(exc)}


def parse_rocprof_counter_names(text: str) -> tuple[str, ...]:
  names: set[str] = set()
  for token in text.replace("\t", " ").split():
    if token.isidentifier() and (token.startswith(("SQ_", "SQC_", "GL2C_", "TA_", "GRBM_")) or token in (
      "VALUInsts", "SALUInsts", "GPUBusy", "OccupancyPercent", "MeanOccupancyPerCU", "LDSBankConflict")):
      names.add(token)
  return tuple(sorted(names))


def probe_amd_counter_capabilities(device: int = 0, *, rocm_root: Path = DEFAULT_ROCM_ROOT,
                                   requested: Iterable[str] = ()) -> dict[str, Any]:
  rocprof, avail = _tool("rocprofv3", rocm_root), _tool("rocprofv3-avail", rocm_root)
  tools = {}
  for name, path in (("rocprofv3", rocprof), ("rocprofv3_avail", avail)):
    tools[name] = {"path": str(path) if path else None, "status": "advertised" if path else "unsupported"}
  list_result = _run([str(avail), "-d", str(device), "list", "--pmc", "--agent", "--pc-sampling"]) if avail else {
    "status": "unsupported", "error": "rocprofv3-avail not found"}
  advertised = parse_rocprof_counter_names(list_result.get("stdout", ""))
  requested_names = tuple(dict.fromkeys(requested))
  scheduling = None
  if avail and requested_names:
    scheduling = _run([str(avail), "-d", str(device), "pmc-check", *requested_names])
  paths = {path: {"exists": Path(path).exists(), "readable": os.access(path, os.R_OK),
                  "writable": os.access(path, os.W_OK)} for path in ("/dev/kfd", "/dev/dri/renderD128")}
  native = ()
  native_error = None
  try:
    from tinygrad.runtime.support.amd import import_pmc
    native = tuple(sorted(import_pmc((11, 0, 0))))
  except Exception as exc: native_error = f"{type(exc).__name__}: {exc}"
  counters = sorted(set(advertised) | set(native) | set(requested_names))
  rows = [{"name": name, "status": "advertised" if name in advertised or name in native else "unsupported",
           "advertised_by": [source for source, values in (("rocprofv3", advertised), ("tinygrad_native", native)) if name in values]}
          for name in counters]
  return {"schema": SCHEMA, "device": device, "metric_status_vocabulary": sorted(STATUSES), "tools": tools,
          "permissions": paths, "rocprof_query": {k: v for k, v in list_result.items() if k != "stdout"},
          "rocprof_advertised_count": len(advertised), "tinygrad_native_count": len(native),
          "tinygrad_native_error": native_error, "scheduling_check": scheduling, "counters": rows,
          "notes": ["advertised and schedulable do not establish liveness", "zero requires positive/negative control classification"]}


def validate_counter_capability(artifact: dict[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  for idx, row in enumerate(artifact.get("counters", [])):
    if row.get("status") not in STATUSES: raise ValueError(f"counters[{idx}].status is invalid")
    if not isinstance(row.get("name"), str) or not row["name"]: raise ValueError(f"counters[{idx}].name is invalid")


if __name__ == "__main__": print(json.dumps(probe_amd_counter_capabilities(), indent=2, sort_keys=True))
