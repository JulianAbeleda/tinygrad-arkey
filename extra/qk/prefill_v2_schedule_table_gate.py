#!/usr/bin/env python3
"""Validate the frozen PREFILL_V2 warmstart schedule table on representative 8B/14B shapes.

This gate answers a narrow but important question for prefill lowering: does the pure generated graph-GEMM
path still apply the table-selected TC/LOCAL schedule for real model shapes, and what throughput does it
measure now? It intentionally reuses `prefill_v2_schedule_search._run_config` so there is one benchmark path
for the schedule table instead of another hand-rolled harness.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from extra.qk.prefill_v2_schedule_search import M_DEFAULT, TABLE_PATH, _run_config
from extra.qk.timing_harness import add_clock_pin_arg

SCHEMA = "prefill-v2-schedule-table-gate.v1"
ARTIFACT_DIR = pathlib.Path("bench/prefill-v2-schedule-table")
DEFAULT_SHAPES = ("4096x4096", "5120x5120")


def _read_table(path: pathlib.Path = TABLE_PATH) -> dict[str, Any]:
  return json.loads(path.read_text())


def _shape_tuple(shape: str) -> tuple[int, int]:
  out_f, in_f = shape.split("x", 1)
  return int(out_f), int(in_f)


def _params_from_opts(opts: list[list[Any]]) -> dict[str, int]:
  params = {"u0": 0, "u1": 0, "loc": 0, "unr": 0}
  upcast_seen = 0
  for name, _axis, arg in opts:
    if name == "UPCAST":
      params["u0" if upcast_seen == 0 else "u1"] = int(arg)
      upcast_seen += 1
    elif name == "LOCAL":
      params["loc"] = int(arg)
    elif name == "UNROLL":
      params["unr"] = int(arg)
  missing = [k for k, v in params.items() if k != "loc" and v == 0]
  if missing: raise ValueError(f"schedule table opts missing {missing}: {opts}")
  return params


def _table_row(shape: str, table: dict[str, Any]) -> dict[str, Any]:
  if shape not in table: raise KeyError(f"{shape} missing from {TABLE_PATH}")
  row = table[shape]
  params = _params_from_opts(row["opts"])
  return {
    "shape": shape,
    "out_f": _shape_tuple(shape)[0],
    "in_f": _shape_tuple(shape)[1],
    "table_tflops": float(row["tflops"]),
    "table_default_tflops": float(row["default_tflops"]),
    "table_speedup": float(row["tflops"]) / max(float(row["default_tflops"]), 1e-20),
    "params": params,
  }


def _run_shape(shape: str, row: dict[str, Any], m: int, *, pin_clock: bool = False) -> dict[str, Any]:
  result = _run_config(m, row["out_f"], row["in_f"], **row["params"], pin_clock=pin_clock)
  return {**row, "measured": result}


def build_report(*, run_amd: bool = False, shapes: tuple[str, ...] = DEFAULT_SHAPES,
                 m: int = M_DEFAULT, pin_clock: bool = False, artifact: bool = True) -> dict[str, Any]:
  table = _read_table()
  rows = [_table_row(shape, table) for shape in shapes]
  if run_amd:
    rows = [_run_shape(row["shape"], row, m, pin_clock=pin_clock) for row in rows]

  missing = [row["shape"] for row in rows if row["params"]["loc"] == 0]
  executed = [row for row in rows if "measured" in row]
  measured_ok = [row for row in executed if row["measured"].get("status") == "ok" and row["measured"].get("tflops", 0.0) > 0.0]
  verdict = "PREFILL_V2_SCHEDULE_TABLE_APPLIES_PASS" if rows and not missing and (not run_amd or len(measured_ok) == len(rows)) \
    else "PREFILL_V2_SCHEDULE_TABLE_BLOCKED"
  report = {
    "schema": SCHEMA,
    "route_id": "prefill_v2_scheduler_matmul_default",
    "m": m,
    "shape_count": len(rows),
    "shapes": list(shapes),
    "verdict": verdict,
    "evidence": {
      "table_exists": TABLE_PATH.exists(),
      "all_selected_shapes_present": len(rows) == len(shapes),
      "all_selected_shapes_use_local": not missing,
      "run_amd": run_amd,
      "pin_clock": pin_clock,
      "all_measured_shapes_ok": (not run_amd or len(measured_ok) == len(rows)),
    },
    "rows": rows,
    "remaining_blocker": None if verdict.endswith("_PASS") else "warmstart table missing LOCAL schedule or AMD measurement failed",
  }
  if artifact:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "latest.json").write_text(json.dumps(report, indent=2))
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--run-amd", action="store_true", help="execute the representative table shapes on AMD")
  add_clock_pin_arg(ap)
  ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES), help="comma-separated out_fxin_f shapes from the table")
  ap.add_argument("--M", type=int, default=M_DEFAULT)
  args = ap.parse_args(argv)
  report = build_report(run_amd=args.run_amd, shapes=tuple(x for x in args.shapes.split(",") if x), m=args.M,
                        pin_clock=args.pin_clock)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
