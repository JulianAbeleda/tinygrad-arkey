#!/usr/bin/env python3
"""One-off conversion: the 2026-09-26 kernel-audit ncu raw CSVs (~/scratchpad/audit/, not in git) ->
a committed docs/nemotron-vllm-parity/bench/audit/results/ncu-kernel-counters-*.json evidence file.

Uses ncu_kernel_counters.py's parser (no duplicated CSV/counter parsing); this script only adds the
role/shape mapping audit_table.py already uses to walk the launch-ordered raw CSV rows, and marks the
per-shape GEMM kernel (gemm_of, same rule as audit_table.py) so results stay small when a shape has
several launches (e.g. split-K reduce, matvec fallback fixups).

usage: build_ncu_evidence.py DIR OUT.json   (DIR holds oncu_raw.csv, oncu.log, vncu_raw.csv, vllm_gemm.json)
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ncu_kernel_counters import SCHEMA, default_provenance, parse_kernel_row, read_ncu_csv_rows, read_ncu_csv_units

MS_DEC = [8, 16, 32, 64, 128]
MS_PF = [512, 1024, 2048, 4096, 8192]
V_ROLES = ["ssm_in", "ssm_out", "qkv", "attn_q", "attn_kv", "attn_o", "ffn_up", "ffn_down", "output"]


def _device_and_ncu_version(csv_path: pathlib.Path) -> tuple[str | None, ...]:
  with csv_path.open(newline="") as f:
    r = csv.reader(f)
    header = next(r)
    row = next(r); row = next(r)  # header, units, first data row
  by_col = dict(zip(header, row))
  return by_col.get("device__attribute_display_name"), by_col.get("CC")


def _mark_gemm(rows: list[dict]) -> None:
  """Same rule as audit_table.py's gemm_of: the longest tensor-pipe kernel is the GEMM; the rest are aux."""
  if not rows:
    return
  cand = [r for r in rows if "splitKreduce" not in (r["kernel"] or "") and r["tensor_pipe_util_pct"] > 0.5]
  gemm = max(cand or rows, key=lambda r: r["duration_us"])
  for r in rows:
    r["aux"] = r is not gemm


def _ours_rows(d: pathlib.Path) -> list[dict]:
  csv_path = d / "oncu_raw.csv"
  raw, units = read_ncu_csv_rows(csv_path), read_ncu_csv_units(csv_path)
  rows, i = [], 0
  for line in (d / "oncu.log").open():
    if not line.startswith("SHAPE "):
      continue
    _, role, m, nk = line.split()
    nk = int(nk)
    group = [parse_kernel_row(k, side="tinygrad", role=role, shape=int(m), units=units) for k in raw[i:i + nk]]
    _mark_gemm(group)
    rows += group
    i += nk
  assert i == len(raw), (i, len(raw))
  return rows


def _reference_rows(d: pathlib.Path) -> list[dict]:
  vj = json.load((d / "vllm_gemm.json").open())
  counts = {(r["role"], r["m"]): len(r["kernels"]) for r in vj}
  csv_path = d / "vncu_raw.csv"
  raw, units = read_ncu_csv_rows(csv_path), read_ncu_csv_units(csv_path)
  rows, i = [], 0
  for role in V_ROLES:
    for m in MS_DEC + MS_PF:
      nk = counts.get((role, m))
      if nk is None:
        continue
      group = [parse_kernel_row(k, side="reference", role=role, shape=m, units=units) for k in raw[i:i + nk]]
      _mark_gemm(group)
      rows += group
      i += nk
  assert i == len(raw), (i, len(raw))
  return rows


def main() -> int:
  if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} DIR OUT.json")
  d, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
  device, cc = _device_and_ncu_version(d / "oncu_raw.csv")
  rows = _ours_rows(d) + _reference_rows(d)
  evidence = {
    "schema": SCHEMA,
    "provenance": default_provenance(
      ncu_version="2026.2.1.0", device_name=f"{device} (sm_{cc.replace('.', '')})" if device else None,
      source_report="oncu.ncu-rep + vncu.ncu-rep (~/scratchpad/audit/, 2026-09-26; not in git)",
    ),
    "rows": rows,
  }
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
  print(f"wrote {len(rows)} rows ({sum(1 for r in rows if not r['aux'])} GEMM, "
        f"{sum(1 for r in rows if r['aux'])} aux) to {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
