#!/usr/bin/env python3
"""Validate the frozen PREFILL_V2 warmstart schedule table on representative 8B/14B shapes.

This gate answers a narrow but important question for prefill lowering: does the pure generated graph-GEMM
path still apply the table-selected TC/LOCAL schedule for real model shapes, and what throughput does it
measure now? It intentionally reuses `prefill_v2_schedule_search._run_config` so there is one benchmark path
for the schedule table instead of another hand-rolled harness.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
from typing import Any

from tinygrad.codegen import to_program_cache
from tinygrad.helpers import getenv
from extra.qk.prefill_v2_schedule_search import M_DEFAULT, TABLE_PATH, _compile_resource_summary, _run_config
from extra.qk.timing_harness import add_clock_pin_arg
from extra.qk.pure_search_guard import effective_routes

SCHEMA = "prefill-v2-schedule-table-gate.v1"
ARTIFACT_DIR = pathlib.Path("bench/prefill-v2-schedule-table")
DEFAULT_SHAPES = ("4096x4096", "5120x5120")
DEFAULT_RESOURCE_STAGES = ("both", "A", "B")
DEFAULT_RESOURCE_U = (2, 4)
DEFAULT_RESOURCE_LOC = (0, 2)
DEFAULT_RESOURCE_UNR = (2, 4, 8)
LDS_LIMIT_BYTES = 65536
A_WINDOW_QUANTUM_BYTES = 4096

def _allow_parked_4x4() -> bool:
  return os.environ.get("PREFILL_ALLOW_PARKED_4X4", "0").strip() == "1"

def _dbuf_slots() -> int:
  return getenv("PREFILL_DBUF_NBUF", 2) if getenv("PREFILL_DBUF", 0) else 1


def _compressed_a_window_key_estimate(stage: str, u0: int, u1: int, loc: int, unr: int) -> dict[str, Any] | None:
  """Prospective resource model for a bounded A fragment-window identity.

  This is deliberately opt-in: it describes the target sidecar model for the
  next primitive, not the current compiler's authoritative allocation. The raw
  A GLOBAL identity is known to be 64KiB per DBUF slot, so it cannot be used as
  the LDS key. The proposed compressed key charges only the live A window:

    total_bytes = NBUF * 4KiB * (A_upcast_fragments + B_tile_fragments)

  for the supported A-window frontier (`loc=2`, `u1=2`). `unr` is accepted only
  when the implementation proves the live producer/consumer window is bounded;
  the model exposes that assumed live window as metadata.
  """
  if not (getenv("PREFILL_DBUF_A_WINDOW_KEY_MODEL", 0) and getenv("PREFILL_TC_LOCAL_STAGE_POST", 0) and getenv("PREFILL_DBUF", 0)):
    return None
  st = stage.lower()
  if st not in ("a", "both"): return None
  nbuf = _dbuf_slots()
  live_unr_window = getenv("PREFILL_DBUF_A_WINDOW_LIVE_UNR", 2)
  b_fragments = u1 if st == "both" else 0
  a_fragments = u0
  estimate = nbuf * A_WINDOW_QUANTUM_BYTES * (a_fragments + b_fragments)
  reasons = []
  if loc != 2: reasons.append("requires loc=2 A local-y identity proof")
  if st == "both" and u1 != 2 and not getenv("PREFILL_DBUF_B_WINDOW_KEY_MODEL", 0):
    reasons.append("u1>2 needs a separate bounded B-window proof")
  if live_unr_window > 2:
    reasons.append("live UNR window must be proven <=2 before treating total UNR as bounded")
  return {
    "static_dbuf_lds_estimate_bytes": estimate,
    "static_dbuf_lds_estimate_model": "future_compressed_a_window_key_v1",
    "dbuf_slots": nbuf,
    "a_window_quantum_bytes": A_WINDOW_QUANTUM_BYTES,
    "a_upcast_fragments": a_fragments,
    "b_tile_fragments": b_fragments,
    "a_lidy_slices": loc,
    "schedule_unr": unr,
    "assumed_live_unr_window": live_unr_window,
    "raw_a_global_identity_bytes": 65536 * nbuf,
    "candidate_filter": "accept" if not reasons else "reject",
    "candidate_filter_reasons": reasons,
  }


def _dbuf_static_lds_estimate(stage: str, u0: int, u1: int, loc: int, unr: int) -> dict[str, Any] | None:
  """Fast fail-closed estimate for current packed WITH_LOCAL DBUF staging.

  The native resource compiler is expensive on production shapes. For the
  pre-poststage DBUF staging substrate, measured ELF group-segment bytes scale as
  one 16KiB chunk per staged output upcast unit:
    A:    16KiB * u0
    B:    16KiB * u1
    both: 16KiB * (u0 + u1)

  The current post-stage A+B path is keyed after LOCAL has been introduced.
  For the bounded production grid, measured ELF group-segment bytes are:
    unr=2: 2x2=32KiB, 4x2=48KiB, 2x4=80KiB, 4x4=96KiB
    unr=4: 2x2=64KiB edge/unsafe
    unr=8: 2x2=128KiB

  This is only a prefilter; below-limit candidates still compile and report the
  authoritative ELF group-segment size.
  """
  compressed = _compressed_a_window_key_estimate(stage, u0, u1, loc, unr)
  if compressed is not None: return compressed
  st = stage.lower()
  if getenv("PREFILL_TC_LOCAL_STAGE_POST", 0) and getenv("PREFILL_DBUF", 0):
    if st == "both":
      if u0 > 2 and getenv("PREFILL_DBUF_LDS_CONST_IMM_UNSAFE", 0) and not getenv("PREFILL_DBUF_BOTH_U0_GT2_PROVEN", 0):
        return {
          "static_dbuf_lds_estimate_bytes": 48 * 1024 if (u0, u1, unr) == (4, 2, 2) else 16 * 1024 * (u0 + u1),
          "static_dbuf_lds_estimate_model": "post_stage_both_unsafe_lds_imm_guard_v2",
          "candidate_filter": "reject",
          "candidate_filter_reasons": ["DBUF both-stage u0>2 is numerically wrong when DS offset0 folding is forced with PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1"],
        }
      if unr > 2: estimate = 16 * 1024 * unr
      elif (u0, u1) == (2, 2): estimate = 32 * 1024
      elif (u0, u1) == (4, 2): estimate = 48 * 1024
      elif (u0, u1) == (2, 4): estimate = 80 * 1024
      elif (u0, u1) == (4, 4): estimate = 96 * 1024
      else: estimate = 128 * 1024
      return {"static_dbuf_lds_estimate_bytes": estimate, "static_dbuf_lds_estimate_model": "post_stage_both_grid_v2"}
    return None
  if st == "a": estimate = 16 * 1024 * u0
  elif st == "b": estimate = 16 * 1024 * u1
  elif st == "both": estimate = 16 * 1024 * (u0 + u1)
  else: return None
  return {"static_dbuf_lds_estimate_bytes": estimate, "static_dbuf_lds_estimate_model": "pre_poststage_upcast_v1"}


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

def _active_params(params: dict[str, int]) -> tuple[dict[str, int], str | None]:
  if (params.get("u0"), params.get("u1")) == (4, 4) and not _allow_parked_4x4():
    return {**params, "u1": 2}, "table 4x4 sanitized to 4x2 because generated 4x4 is parked on gfx1100"
  return params, None


def _table_row(shape: str, table: dict[str, Any]) -> dict[str, Any]:
  if shape not in table: raise KeyError(f"{shape} missing from {TABLE_PATH}")
  row = table[shape]
  params, parked_note = _active_params(_params_from_opts(row["opts"]))
  out = {
    "shape": shape,
    "out_f": _shape_tuple(shape)[0],
    "in_f": _shape_tuple(shape)[1],
    "table_tflops": float(row["tflops"]),
    "table_default_tflops": float(row["default_tflops"]),
    "table_speedup": float(row["tflops"]) / max(float(row["default_tflops"]), 1e-20),
    "params": params,
  }
  if parked_note is not None: out["parked_4x4_policy"] = parked_note
  return out


def _run_shape(shape: str, row: dict[str, Any], m: int, *, pin_clock: bool = False) -> dict[str, Any]:
  result = _run_config(m, row["out_f"], row["in_f"], **row["params"], pin_clock=pin_clock)
  return {**row, "measured": result}


def _resource_search(row: dict[str, Any], m: int, *, stages: tuple[str, ...] = DEFAULT_RESOURCE_STAGES,
                     u_values: tuple[int, ...] = DEFAULT_RESOURCE_U, loc_values: tuple[int, ...] = DEFAULT_RESOURCE_LOC,
                     unr_values: tuple[int, ...] = DEFAULT_RESOURCE_UNR, lds_limit_bytes: int = LDS_LIMIT_BYTES,
                     run_below_limit: bool = False, run_limit: int = 12, pin_clock: bool = False) -> dict[str, Any]:
  old_stage = os.environ.get("PREFILL_TC_LOCAL_STAGE")
  candidates = []
  try:
    for stage, u0, u1, loc, unr in itertools.product(stages, u_values, u_values, loc_values, unr_values):
      os.environ["PREFILL_TC_LOCAL_STAGE"] = stage
      getenv.cache_clear()
      to_program_cache.clear()
      candidate = {"stage": stage, "u0": u0, "u1": u1, "loc": loc, "unr": unr}
      if (u0, u1) == (4, 4) and not _allow_parked_4x4():
        candidate.update({
          "status": "parked-4x4",
          "over_limit": True,
          "below_limit": False,
          "message": "skipped: generated 4x4 path is parked on gfx1100; set PREFILL_ALLOW_PARKED_4X4=1 for diagnostics",
        })
        candidates.append(candidate)
        continue
      estimate = _dbuf_static_lds_estimate(stage, u0, u1, loc, unr)
      if estimate is not None: candidate.update(estimate)
      estimate_bytes = candidate.get("static_dbuf_lds_estimate_bytes")
      if candidate.get("candidate_filter") == "reject":
        candidate.update({
          "status": "static-filtered",
          "over_limit": estimate_bytes is not None and estimate_bytes >= lds_limit_bytes,
          "below_limit": False,
          "message": "skipped full native compile: candidate does not satisfy static DBUF window-key filters",
        })
        candidates.append(candidate)
        continue
      if estimate_bytes is not None and estimate_bytes >= lds_limit_bytes:
        candidate.update({
          "status": "static-over-limit",
          "binary_group_segment_bytes": estimate_bytes,
          "local_bytes": estimate_bytes,
          "over_limit": True,
          "below_limit": False,
          "message": "skipped full native compile: static DBUF LDS estimate is at/over limit",
        })
        candidates.append(candidate)
        continue
      try:
        resource = _compile_resource_summary(m, row["out_f"], row["in_f"], u0, u1, loc, unr)
        candidate.update({
          "status": "compile-ok",
          "binary_group_segment_bytes": resource.get("binary_group_segment_bytes"),
          "local_bytes": resource.get("local_bytes"),
          "n_threads": resource.get("n_threads"),
          "over_limit": resource.get("over_limit"),
          "below_limit": resource.get("binary_group_segment_bytes") is not None and resource["binary_group_segment_bytes"] < lds_limit_bytes,
          "warmstart_apply_count": resource.get("warmstart_apply_count"),
        })
      except Exception as e:
        candidate.update({"status": type(e).__name__, "message": str(e)[:240]})
      candidates.append(candidate)
  finally:
    if old_stage is None: os.environ.pop("PREFILL_TC_LOCAL_STAGE", None)
    else: os.environ["PREFILL_TC_LOCAL_STAGE"] = old_stage
    getenv.cache_clear()
    to_program_cache.clear()
  compiled = [c for c in candidates if c.get("status") == "compile-ok"]
  below = [c for c in compiled if c.get("below_limit")]
  safe_order = sorted(compiled, key=lambda c: (c.get("binary_group_segment_bytes") is None,
                                              c.get("binary_group_segment_bytes", 1 << 60),
                                              c["stage"], c["u0"], c["u1"], c["loc"], c["unr"]))
  runtime_rows = []
  if run_below_limit:
    try:
      for candidate in sorted(below, key=lambda c: (c["stage"], c["binary_group_segment_bytes"], c["u0"], c["u1"], c["loc"], c["unr"]))[:run_limit]:
        os.environ["PREFILL_TC_LOCAL_STAGE"] = candidate["stage"]
        getenv.cache_clear()
        to_program_cache.clear()
        measured = _run_config(m, row["out_f"], row["in_f"], candidate["u0"], candidate["u1"], candidate["loc"], candidate["unr"],
                               pin_clock=pin_clock)
        runtime_rows.append({**candidate, "measured": measured})
    finally:
      if old_stage is None: os.environ.pop("PREFILL_TC_LOCAL_STAGE", None)
      else: os.environ["PREFILL_TC_LOCAL_STAGE"] = old_stage
      getenv.cache_clear()
      to_program_cache.clear()
  runtime_ok = [c for c in runtime_rows if c["measured"].get("status") == "ok"]
  return {
    "shape": row["shape"],
    "lds_limit_bytes": lds_limit_bytes,
    "candidate_count": len(candidates),
    "compile_ok_count": len(compiled),
    "below_limit_count": len(below),
    "best_by_resource": safe_order[:12],
    "runtime_probe_count": len(runtime_rows),
    "best_by_runtime": sorted(runtime_ok, key=lambda c: c["measured"].get("tflops", 0.0), reverse=True)[:8],
    "runtime_probes": runtime_rows,
    "candidates": candidates,
  }


def build_report(*, run_amd: bool = False, shapes: tuple[str, ...] = DEFAULT_SHAPES,
                 m: int = M_DEFAULT, pin_clock: bool = False, artifact: bool = True,
                 resource_search: bool = False, resource_stages: tuple[str, ...] = DEFAULT_RESOURCE_STAGES,
                 resource_u_values: tuple[int, ...] = DEFAULT_RESOURCE_U,
                 resource_loc_values: tuple[int, ...] = DEFAULT_RESOURCE_LOC,
                 resource_unr_values: tuple[int, ...] = DEFAULT_RESOURCE_UNR,
                 resource_run_below_limit: bool = False, resource_run_limit: int = 12) -> dict[str, Any]:
  table = _read_table()
  rows = [_table_row(shape, table) for shape in shapes]
  if run_amd:
    rows = [_run_shape(row["shape"], row, m, pin_clock=pin_clock) for row in rows]
  resource_rows = [_resource_search(row, m, stages=resource_stages, u_values=resource_u_values,
                                    loc_values=resource_loc_values, unr_values=resource_unr_values,
                                    run_below_limit=resource_run_below_limit, run_limit=resource_run_limit,
                                    pin_clock=pin_clock) for row in rows] if resource_search else []

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
      "resource_search": resource_search,
      "resource_run_below_limit": resource_run_below_limit,
      "all_measured_shapes_ok": (not run_amd or len(measured_ok) == len(rows)),
    },
    "route_attribution": next((r for r in effective_routes() if r.get("family") == "prefill_gemm"), None),
    "rows": rows,
    "resource_search": resource_rows,
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
  ap.add_argument("--no-artifact", action="store_true", help="do not write bench/prefill-v2-schedule-table/latest.json")
  ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES), help="comma-separated out_fxin_f shapes from the table")
  ap.add_argument("--M", type=int, default=M_DEFAULT)
  ap.add_argument("--resource-search", action="store_true", help="compile-only bounded DBUF resource search over existing schedule knobs")
  ap.add_argument("--resource-stages", default=",".join(DEFAULT_RESOURCE_STAGES), help="comma-separated PREFILL_TC_LOCAL_STAGE values for --resource-search")
  ap.add_argument("--resource-u", default=",".join(str(x) for x in DEFAULT_RESOURCE_U), help="comma-separated UPCAST values for --resource-search")
  ap.add_argument("--resource-loc", default=",".join(str(x) for x in DEFAULT_RESOURCE_LOC), help="comma-separated LOCAL values for --resource-search")
  ap.add_argument("--resource-unr", default=",".join(str(x) for x in DEFAULT_RESOURCE_UNR), help="comma-separated UNROLL values for --resource-search")
  ap.add_argument("--resource-run-below-limit", action="store_true", help="benchmark below-limit DBUF resource candidates")
  ap.add_argument("--resource-run-limit", type=int, default=12, help="maximum below-limit resource candidates to benchmark per shape")
  args = ap.parse_args(argv)
  parse_ints = lambda s: tuple(int(x) for x in s.split(",") if x)
  report = build_report(run_amd=args.run_amd, shapes=tuple(x for x in args.shapes.split(",") if x), m=args.M,
                        pin_clock=args.pin_clock, artifact=not args.no_artifact, resource_search=args.resource_search,
                        resource_stages=tuple(x for x in args.resource_stages.split(",") if x),
                        resource_u_values=parse_ints(args.resource_u), resource_loc_values=parse_ints(args.resource_loc),
                        resource_unr_values=parse_ints(args.resource_unr),
                        resource_run_below_limit=args.resource_run_below_limit, resource_run_limit=args.resource_run_limit)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
