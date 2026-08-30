#!/usr/bin/env python3
"""Run the exact-image NCU counter bridge for every captured production row kernel.

Reads a cubin-capture JSON produced by nv_cubin_capture.py, launches each captured
cubin with its exact production buffer sizes through the CUDA driver harness, and
collects the fixed counter set under Nsight Compute. NCU needs admin perf-counter
access, so the child command runs under sudo. Output is a consolidated row ledger.
"""
from __future__ import annotations

import argparse, json, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "extra/llm_research/decode/nv_cubin_ncu_launcher.py"
PYTHON = "/home/ubuntu/tinygrad-arkey/.venv/bin/python"

METRICS = ",".join([
  "gpu__time_duration.sum",
  "dram__bytes.sum",
  "dram__bytes_op_read.sum",
  "dram__bytes_op_write.sum",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
  "sm__inst_executed.sum",
  "lts__t_bytes.sum",
  "lts__t_sector_op_read_hit_rate.pct",
  "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
])

def parse_metrics(text: str) -> dict[str, float]:
  rows: dict[str, float] = {}
  for line in text.splitlines():
    line = line.rstrip()
    if not line:
      continue
    stripped = line.strip()
    if not stripped or stripped.startswith("Section") or stripped.startswith("----") or stripped.startswith("Metric"):
      continue
    parts = line.split()
    if len(parts) < 3 or "__" not in parts[0]:
      continue
    try:
      rows[parts[0]] = float(parts[-1])
    except ValueError:
      pass
  return rows


def run_ncu(cubin: pathlib.Path, symbol: str, sizes: list[int], reps: int,
            grid: list[int], block: list[int]) -> tuple[str, int]:
  out = pathlib.Path("/tmp/ncu-row-launch.json")
  cmd = [
    "sudo", "-E", "ncu",
    "--launch-count", "1", "--launch-skip", "0",
    "--metrics", METRICS,
    "--target-processes", "all",
    PYTHON, str(LAUNCHER),
    "--cubin", str(cubin),
    "--symbol", symbol,
    "--n-bufs", str(len(sizes)),
    "--buf-sizes", ",".join(map(str, sizes)),
    "--grid", ",".join(map(str, grid)),
    "--block", ",".join(map(str, block)),
    "--reps", str(reps),
    "--out", str(out),
  ]
  proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
  return proc.stdout + "\n" + proc.stderr, proc.returncode


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--capture", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--only", help="comma-separated kernel-name substrings to run")
  args = ap.parse_args()

  capture = json.loads(args.capture.read_text())
  only = [s for s in (args.only.split(",") if args.only else []) if s]
  rows = []
  for rec in capture["captured"]:
    name = rec["name"]
    if only and not any(s in name for s in only):
      continue
    first = rec["calls"][0]
    sizes = [m["size"] for m in first["buf_meta"]]
    grid = first["global_size"]
    block = first["local_size"]
    cubin = pathlib.Path(rec["cubin_path"])
    text, rc = run_ncu(cubin, name, sizes, args.reps, grid, block)
    rows.append({
      "name": name,
      "cubin_sha256": rec["cubin_sha256"],
      "grid": first["global_size"],
      "block": first["local_size"],
      "buf_sizes": sizes,
      "ncu_returncode": rc,
      "metrics": parse_metrics(text),
      "raw_tail": "\n".join(text.splitlines()[-24:]),
    })
    print(json.dumps(rows[-1], indent=2, sort_keys=True), flush=True)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  result = {"schema": "tinygrad.nv_row_counter_bridge.v1", "rows": rows}
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(f"\nWROTE {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
