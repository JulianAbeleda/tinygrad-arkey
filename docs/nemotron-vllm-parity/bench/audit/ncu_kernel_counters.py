#!/usr/bin/env python3
"""Shared Nsight Compute collector for the Nemotron-H vs vLLM/cuBLAS kernel audit.

This module is the single place that (a) builds the ``ncu`` command line used to
collect both audit targets -- tinygrad's own GEMM routes (through the DEV=CUDA
proxy target ``ours_ncu.py``, or a captured cubin via ``nv_cubin_ncu_launcher.py``)
and the vLLM/cuBLAS reference GEMMs (``vllm_gemm.py`` in the vLLM venv) -- and
(b) parses ``ncu --page raw --csv`` output into a small, versioned JSON evidence
document. ``oncu.sh``/``vncu.sh`` are thin shell callers of ``build-cmd``;
``audit_table.py`` is a thin caller of ``parse_kernel_row``. Neither duplicates
the section list, cache/clock-control flags, or the counter-column parsing.

Per Julian's 2026-09-26 direction (docs/nemotron-vllm-parity/goal-board.md),
ongoing NCU collection/import/cross-run comparison ownership is moving to
BoltBeam; tinygrad keeps the cubin+launch-spec exporter (nv_cubin_capture.py /
nv_cubin_ncu_launcher.py). This module is the finishing pass on the audit run
already collected under ~/scratchpad/audit/ (oncu_raw.csv, vncu_raw.csv) and a
CPU-testable reference for the counter-column mapping BoltBeam's importer can
reuse; it does not start a new open-ended in-tree pipeline.

Schema: "tinygrad.nv_ncu_kernel_counters.v1"
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable

SCHEMA = "tinygrad.nv_ncu_kernel_counters.v1"

# Single source of truth for the sections/cache/clock-control the audit needs.
# oncu.sh and vncu.sh no longer hardcode this list; they call `build-cmd`.
NCU_SECTIONS = (
  "SpeedOfLight", "LaunchStats", "Occupancy", "WarpStateStats",
  "MemoryWorkloadAnalysis", "ComputeWorkloadAnalysis",
)


def build_ncu_command(
  *,
  out: str,
  target_cmd: Iterable[str],
  ncu: str = "ncu",
  sections: Iterable[str] = NCU_SECTIONS,
  cache_control: str = "all",
  clock_control: str = "none",
  target_processes: str = "all",
  profile_from_start: str = "off",
  nvtx: bool = False,
  kernel_name: str | None = None,
  launch_count: int | None = None,
) -> list[str]:
  """Build one ``ncu`` argv. Both audit targets (ours, vLLM/cuBLAS) share this."""
  cmd = [ncu, "--profile-from-start", profile_from_start, "--target-processes", target_processes]
  for section in sections:
    cmd += ["--section", section]
  if nvtx:
    cmd.append("--nvtx")
  if kernel_name:
    cmd += ["--kernel-name", kernel_name]
  if launch_count is not None:
    cmd += ["--launch-count", str(launch_count)]
  cmd += ["--cache-control", cache_control, "--clock-control", clock_control, "-f", "-o", str(out)]
  cmd += list(target_cmd)
  return cmd


def sudo_memcap_scope(cmd: list[str], *, mem_max: str = "20G", env: dict[str, str] | None = None) -> list[str]:
  """Wrap a command in the memory-capped sudo scope (a root profiler once OOM-crashed this box)."""
  env_args = [f"{k}={v}" for k, v in (env or {}).items()]
  return ["sudo", "systemd-run", "--scope", "-q", "-p", f"MemoryMax={mem_max}", "env", *env_args, *cmd]


def export_raw_csv(report_path: str, out_csv: str, *, ncu: str = "ncu", page: str = "raw") -> str:
  """``ncu --import REPORT --page raw --csv`` -> out_csv. Shared by both audit targets."""
  cmd = [ncu, "--import", str(report_path), "--page", page, "--csv"]
  with open(out_csv, "w") as f:
    subprocess.run(cmd, check=True, stdout=f)
  return out_csv


# --------------------------------------------------------------------------
# Pure parsing (CPU-only, unit-testable: no ncu, no CUDA, no GPU needed).
# --------------------------------------------------------------------------

_STALL_RE = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")
_BANK_CONFLICT_SUBSTR = "bank_conflict"


def read_ncu_csv_rows(path: str | pathlib.Path) -> list[dict[str, str]]:
  """Read an ``ncu --page raw --csv`` export. Row 0 is the header, row 1 is units."""
  with open(path, newline="") as f:
    rows = list(csv.reader(f))
  if len(rows) < 2:
    raise ValueError(f"{path}: expected header + units + data rows, got {len(rows)} rows")
  header = rows[0]
  return [dict(zip(header, row)) for row in rows[2:]]


def read_ncu_csv_units(path: str | pathlib.Path) -> dict[str, str]:
  """Column -> unit, from an ``ncu --page raw --csv`` export's row 1 (one unit per column, whole file)."""
  with open(path, newline="") as f:
    rows = list(csv.reader(f))
  if len(rows) < 2:
    raise ValueError(f"{path}: expected header + units + data rows, got {len(rows)} rows")
  return dict(zip(rows[0], rows[1]))


_TIME_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
_BYTES_PER_SEC_TO_BASE = {"byte/s": 1.0, "kbyte/s": 1e3, "mbyte/s": 1e6, "gbyte/s": 1e9, "tbyte/s": 1e12}


def _unit_scale(unit: str | None, table: dict[str, float]) -> float | None:
  return table.get((unit or "").strip().lower())


def _f(value: Any, default: float = 0.0) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def top_stall_reasons(raw: dict[str, str], *, top_n: int = 3) -> list[dict[str, Any]]:
  """Relative share (%) of each stall reason among the reasons this kernel reports."""
  stalls = sorted(
    ((_f(v), m.group(1)) for c, v in raw.items() if (m := _STALL_RE.match(c)) and v and "selected" not in c),
    reverse=True,
  )
  total = sum(v for v, _ in stalls) or 1.0
  return [{"reason": name, "pct": value / total * 100.0} for value, name in stalls[:top_n]]


def bank_conflicts(raw: dict[str, str]) -> dict[str, float]:
  """Any bank-conflict counters ncu collected (absent unless a relevant section requested them)."""
  out = {}
  for key, value in raw.items():
    if _BANK_CONFLICT_SUBSTR in key.lower() and value not in (None, ""):
      parsed = _f(value, default=None)  # type: ignore[arg-type]
      if parsed is not None:
        out[key] = parsed
  return out


def parse_kernel_row(
  raw: dict[str, str], *, side: str, role: str | None = None, shape: int | None = None, aux: bool = False,
  units: dict[str, str] | None = None,
) -> dict[str, Any]:
  """One ``ncu --page raw --csv`` row -> one evidence-schema kernel row.

  ``units`` is the column->unit map from ``read_ncu_csv_units`` on the same file (ncu auto-picks a
  unit per report, e.g. Tbyte/s here vs Gbyte/s on a lower-throughput run). Without it, duration falls
  back to the magnitude heuristic this audit's raw exports already satisfy (row values are always ms,
  so "< 1e3 -> already us" never actually triggers; multiplying by 1e3 is the ms->us conversion).
  """
  units = units or {}
  duration = _f(raw.get("gpu__time_duration.sum"))
  duration_scale = _unit_scale(units.get("gpu__time_duration.sum"), _TIME_TO_US)
  dram_raw = _f(raw.get("dram__bytes.sum.per_second"))
  dram_scale = _unit_scale(units.get("dram__bytes.sum.per_second"), _BYTES_PER_SEC_TO_BASE)
  row: dict[str, Any] = {
    "side": side,
    "role": role,
    "shape": shape,
    "aux": aux,
    "kernel": raw.get("Kernel Name"),
    "grid": raw.get("Grid Size"),
    "block": raw.get("Block Size"),
    "registers_per_thread": int(_f(raw.get("launch__registers_per_thread"))),
    # ncu reports *_static in Kbyte/block but *_dynamic in byte/block; normalize both to bytes.
    "static_shared_bytes": _f(raw.get("launch__shared_mem_per_block_static")) * 1024.0,
    "dynamic_shared_bytes": _f(raw.get("launch__shared_mem_per_block_dynamic")),
    "duration_us": duration * (duration_scale if duration_scale is not None else (1e3 if duration < 1e3 else 1.0)),
    "achieved_occupancy_pct": _f(raw.get("sm__warps_active.avg.pct_of_peak_sustained_active")),
    "tensor_pipe_util_pct": _f(raw.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")),
    "dram_throughput_bytes_per_sec": dram_raw * (dram_scale if dram_scale is not None else 1.0),
    "sm_throughput_pct": _f(raw.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
    "top_stall_reasons": top_stall_reasons(raw),
  }
  conflicts = bank_conflicts(raw)
  if conflicts:
    row["bank_conflicts"] = conflicts
  return row


def build_evidence(rows: list[dict[str, Any]], *, provenance: dict[str, Any]) -> dict[str, Any]:
  return {"schema": SCHEMA, "provenance": provenance, "rows": rows}


def default_provenance(*, ncu_version: str | None, device_name: str | None, source_report: str,
                        cache_control: str = "all", clock_control: str = "none") -> dict[str, Any]:
  return {
    "ncu_version": ncu_version,
    "device_name": device_name,
    "cache_control": cache_control,
    "clock_control": clock_control,
    "source_report": source_report,
  }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cmd_build_cmd(args: argparse.Namespace) -> int:
  env = dict(kv.split("=", 1) for kv in (args.env or []))
  cmd = build_ncu_command(
    out=args.out, target_cmd=args.target_cmd, ncu=args.ncu, nvtx=args.nvtx,
    kernel_name=args.kernel_name, launch_count=args.launch_count,
  )
  if args.sudo_scope:
    cmd = sudo_memcap_scope(cmd, mem_max=args.mem_max, env=env)
  elif env:
    cmd = ["env", *(f"{k}={v}" for k, v in env.items()), *cmd]
  print(shlex.join(cmd))
  return 0


def _cmd_to_json(args: argparse.Namespace) -> int:
  units = read_ncu_csv_units(args.csv)
  rows = [parse_kernel_row(raw, side=args.side, units=units) for raw in read_ncu_csv_rows(args.csv)]
  evidence = build_evidence(rows, provenance=default_provenance(
    ncu_version=args.ncu_version, device_name=args.device_name, source_report=args.csv,
  ))
  out = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
  if args.out:
    pathlib.Path(args.out).write_text(out)
  else:
    sys.stdout.write(out)
  return 0


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  sub = ap.add_subparsers(dest="cmd", required=True)

  bc = sub.add_parser("build-cmd", help="print the ncu argv for one audit target (shell-quoted)")
  bc.add_argument("--out", required=True, help="ncu -o report prefix")
  bc.add_argument("--ncu", default="ncu")
  bc.add_argument("--nvtx", action="store_true")
  bc.add_argument("--kernel-name", default=None)
  bc.add_argument("--launch-count", type=int, default=None)
  bc.add_argument("--sudo-scope", action="store_true")
  bc.add_argument("--mem-max", default="20G")
  bc.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
  bc.add_argument("target_cmd", nargs=argparse.REMAINDER, help="-- python3 script.py args...")
  bc.set_defaults(fn=_cmd_build_cmd)

  tj = sub.add_parser("to-json", help="parse an ncu --page raw --csv export into the evidence schema")
  tj.add_argument("--csv", required=True)
  tj.add_argument("--side", required=True, choices=["tinygrad", "reference"])
  tj.add_argument("--ncu-version", default=None)
  tj.add_argument("--device-name", default=None)
  tj.add_argument("--out", default=None)
  tj.set_defaults(fn=_cmd_to_json)

  args = ap.parse_args(argv)
  if args.cmd == "build-cmd" and args.target_cmd and args.target_cmd[0] == "--":
    args.target_cmd = args.target_cmd[1:]
  return args.fn(args)


if __name__ == "__main__":
  raise SystemExit(main())
