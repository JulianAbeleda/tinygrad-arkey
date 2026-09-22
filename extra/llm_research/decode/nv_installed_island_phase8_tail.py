#!/usr/bin/env python3
"""Phase 8 tail decomposition: exact body B and clean HCQ C for tail kernels.

Reads the retained tail cubin capture, replays every material tail kernel
(production node mass above 2 us/token) as an exact cubin under nsys to get
the pure body duration B, and replays it through the NVComputeQueue chained
HCQ path to get the clean chained duration C.  Production P is read from the
frozen census capture.  It writes one JSON row per kernel with
P = B + D + R.

Measurement tooling only; no production model/runtime path is changed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv/bin/python")
LAUNCHER = str(ROOT / "extra/llm_research/decode/nv_cubin_ncu_launcher.py")
HCQ = str(ROOT / "extra/llm_research/decode/nv_hcq_dispatch_slope_general.py")


def _read_capture(path: pathlib.Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def _sqlite_body(sqlite_path: pathlib.Path) -> float:
  con = sqlite3.connect(sqlite_path)
  try:
    rows = con.execute(
      "select start, end from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    durs = [(e - s) / 1000.0 for s, e in rows]
  finally:
    con.close()
  if not durs:
    raise RuntimeError(f"no kernel rows in {sqlite_path}")
  return statistics.median(durs)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
  return subprocess.run(cmd, text=True, capture_output=True)


def _measure_exact(cubin: pathlib.Path, symbol: str, grid: list[int],
                   block: list[int], buf_sizes: list[int], vals: list[int],
                   out: pathlib.Path) -> float:
  rep = out.with_suffix(".nsys-rep")
  launch_out = out.with_name(out.stem + "-launch.json")
  g = ",".join(str(x) for x in grid)
  b = ",".join(str(x) for x in block)
  sizes = ",".join(str(x) for x in buf_sizes)
  cmd = ["nsys", "profile", "--trace=cuda", "--force-overwrite=true",
         "--output", str(rep), PY, LAUNCHER,
         "--cubin", str(cubin), "--symbol", symbol,
         "--grid", g, "--block", b, "--n-bufs", str(len(buf_sizes)),
         "--buf-sizes", sizes, "--reps", "2000", "--out", str(launch_out)]
  if vals:
    cmd = cmd[:-2] + ["--vals", ",".join(str(v) for v in vals)] + cmd[-2:]
  r = _run(cmd)
  (out.with_suffix(".nsys.stdout")).write_text(r.stdout)
  (out.with_suffix(".nsys.stderr")).write_text(r.stderr)
  sqlite = rep.with_suffix(".sqlite")
  e = _run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
            "--output", str(sqlite), str(rep)])
  (out.with_suffix(".export.stderr")).write_text(e.stderr)
  return _sqlite_body(sqlite)


def _measure_hcq(cubin: pathlib.Path, symbol: str, grid: list[int],
                 block: list[int], buf_sizes: list[int], out: pathlib.Path) -> dict:
  g = ",".join(str(x) for x in grid)
  b = ",".join(str(x) for x in block)
  sizes = ",".join(str(x) for x in buf_sizes)
  cmd = [PY, HCQ, "--cubin", str(cubin), "--symbol", symbol,
         "--grid", g, "--block", b, "--buf-sizes", sizes,
         "--warmup", "2", "--reps", "3", "--out", str(out)]
  r = _run(cmd)
  (out.with_suffix(".stdout")).write_text(r.stdout)
  (out.with_suffix(".stderr")).write_text(r.stderr)
  if r.returncode != 0:
    raise RuntimeError(f"HCQ failed for {symbol}: {r.stderr}")
  return json.loads(out.read_text(encoding="utf-8"))["slopes"]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--capture", type=pathlib.Path, required=True)
  ap.add_argument("--p-map", type=pathlib.Path, required=True)
  ap.add_argument("--evidence-dir", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--min-mass-us", type=float, default=2.0)
  ap.add_argument("--island-id", default="I_TAIL")
  ap.add_argument("--semantic-role", default="sampler_tail")
  args = ap.parse_args()

  capture = _read_capture(args.capture)
  p_map = json.loads(args.p_map.read_text(encoding="utf-8"))

  args.evidence_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for rec in capture["captured"]:
    symbol = rec["name"]
    p_us = p_map.get(symbol)
    if p_us is None or p_us < args.min_mass_us:
      continue
    cubin = ROOT / rec["cubin_path"]
    call = rec["calls"][0]
    grid = [int(x) for x in call["global_size"]]
    block = [int(x) for x in call["local_size"]]
    buf_sizes = [int(m["size"]) for m in call["buf_meta"]]
    vals = [int(v) for v in call.get("vals", [])]
    stem = symbol
    exact_out = args.evidence_dir / f"{stem}.exact"
    hcq_out = args.evidence_dir / f"{stem}.hcq.json"

    print(f"[B] {symbol}", flush=True)
    b_us = _measure_exact(cubin, symbol, grid, block, buf_sizes, vals, exact_out)
    print(f"[C] {symbol}", flush=True)
    slopes = _measure_hcq(cubin, symbol, grid, block, buf_sizes, hcq_out)
    c_us = slopes["plain_drain_us_per_kernel"]
    rows.append({
      "schema": "tinygrad.nv_installed_island.v1",
      "island_id": args.island_id,
      "semantic_role": args.semantic_role,
      "symbol": symbol,
      "cubin_sha256": rec["cubin_sha256"],
      "grid": grid,
      "block": block,
      "buf_sizes": buf_sizes,
      "vals": vals,
      "shmem_usage": rec.get("shmem_usage"),
      "regs_usage": rec.get("regs_usage"),
      "production_p_us": round(p_us, 3),
      "body_b_us": round(b_us, 4),
      "clean_hcq_c_us": round(c_us, 4),
      "clean_dispatch_d_us": round(c_us - b_us, 4),
      "production_residual_r_us": round(p_us - c_us, 4),
      "identity_residual_us": round(p_us - b_us - (c_us - b_us) - (p_us - c_us), 4),
    })

  args.out.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
  for r in rows:
    print(json.dumps(r, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
