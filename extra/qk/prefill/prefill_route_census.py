#!/usr/bin/env python3
"""One-table census for generated and hand prefill machine-code routes.

This is the starting-line tool: every route is normalized to the final AMD
instruction stream and reported with the same counters. Generated routes also
run the existing correctness/timing worker unless --structural-only is set.
"""
from __future__ import annotations

import argparse, json, os, sys, traceback
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.getcwd())

from tinygrad.helpers import getenv
from tinygrad.codegen import to_program_cache

from extra.qk.prefill import kernel_lifecycle_trace as life
from extra.qk.prefill.hand_vs_generated_shape_matrix import _run_config


DIRECT_ENV = {
  "DEV": "AMD:ISA",
  "AMD_ISA_SCHED": "1",
  "AMD_ISA_WAITCNT_TARGETED": "0",
  "AMD_ISA_WMMA_B128_FRAG": "1",
  "AMD_ISA_REG_ACCUM": "1",
  "PREFILL_DBUF": "0",
}

KMAJOR_LDS_ENV = {
  "DEV": "AMD:ISA",
  "AMD_ISA_SCHED": "1",
  "AMD_ISA_WAITCNT_TARGETED": "0",
  "AMD_ISA_WMMA_B128_FRAG": "1",
  "AMD_ISA_REG_ACCUM": "1",
  "PREFILL_TC_LOCAL_STAGE": "both",
  "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": "1",
  "PREFILL_TC_LOCAL_STAGE_B_TILEKEY": "1",
  "PREFILL_TC_LOCAL_STAGE_POST": "1",
  "PREFILL_LDS_PACK_WITHLOCAL_B128": "1",
  "PREFILL_DBUF": "1",
  "PREFILL_DBUF_LDS_CONST_IMM": "1",
  "PREFILL_DBUF_LDS_REGION_BASE_SPLIT": "1",
  "PREFILL_DBUF_LDS_REGION_BASE_MEMO": "1",
  "PREFILL_DBUF_LDS_INDEX_SPLIT": "1",
  "PREFILL_DBUF_LDS_STORE_BASE_SPLIT": "1",
  "PREFILL_DBUF_DIRECT_B128_CHAIN": "1",
  "PREFILL_DBUF_LDS_ADDR_USE_DEP": "1",
  "REGALLOC_ADDR_REMAT": "1",
  "PREFILL_WMMA_AB_PROOF_META": "1",
  "PREFILL_WMMA_AB_PROOF_KEY": "1",
  "PREFILL_WMMA_AB_PROOF_FROM_LDS_STORES": "1",
  "PREFILL_WMMA_AB_PHASE_SCOPED_KEY": "1",
  "PREFILL_WMMA_KMAJOR_PHASE": "1",
  "PREFILL_WMMA_CHAIN_AB_RESIDENT": "1",
  "PREFILL_STAGE_PRESERVE_TAGS": "1",
}


@contextmanager
def patched_env(env: dict[str, str]):
  old = {k: os.environ.get(k) for k in env}
  os.environ.update(env)
  getenv.cache_clear()
  to_program_cache.clear()
  try:
    yield
  finally:
    for k, v in old.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()
    to_program_cache.clear()


def parse_shapes(raw: str) -> list[tuple[int, int]]:
  out = []
  for item in raw.split(";"):
    if not item.strip(): continue
    a, b = item.split(",", 1)
    out.append((int(a), int(b)))
  return out


def per_wmma(report: dict[str, Any], opname: str) -> float:
  tc = report.get("track_counts", {})
  wmma = tc.get(life.sp.WMMA_NAME, 0)
  return round(tc.get(opname, 0) / wmma, 3) if wmma else 0.0


def structural_fields(report: dict[str, Any]) -> dict[str, Any]:
  if not report.get("ok", True):
    return {"structure_status": "error", "structure_error": report.get("error")}
  tc = report.get("track_counts", {})
  wmma = tc.get(life.sp.WMMA_NAME, 0)
  inst = report.get("instruction_total", 0)
  active = report.get("active_shape_dbuf_cadence", {})
  d7 = report.get("dbuf_gate_summary", {}).get("D7_scheduler_readiness", {})
  return {
    "structure_status": "ok",
    "instruction_count": inst,
    "byte_count": report.get("byte_count", 0),
    "wmma_count": wmma,
    "inst_per_wmma": round(inst / wmma, 3) if wmma else 0.0,
    "waitcnt_per_wmma": report.get("waitcnt_summary", {}).get("per_wmma_avg", 0.0),
    "global_b128_per_wmma": per_wmma(report, "global_load_b128"),
    "ds_store_b128_per_wmma": per_wmma(report, "ds_store_b128"),
    "ds_load_b128_per_wmma": per_wmma(report, "ds_load_b128"),
    "barrier_per_wmma": per_wmma(report, "s_barrier"),
    "between_global_regions": len(report.get("global_work_between_wmmas", [])),
    "future_slot_before_compute": active.get("future_slot_work_before_current_compute", False),
    "operand_origins": report.get("wmma_operand_origin_counts", {}),
    "dbuf_d7_ok": d7.get("ok", False),
    "dbuf_d7_reason": d7.get("reason", ""),
  }


def lifecycle_args(args: argparse.Namespace, shape: tuple[int, int]) -> SimpleNamespace:
  return SimpleNamespace(
    m=args.m, n=args.n, k=args.k, loc=args.loc, unr=args.unr, target=args.target,
    wm=shape[0], wn=shape[1], waves_m=args.waves_m, waves_n=args.waves_n,
    bk=args.bk, pad=args.pad, dbuf=args.dbuf, tm=args.tm, tn=args.tn,
    plra=0, plrab=0, leanaddr=0, dshalf=0,
  )


def generated_row(args: argparse.Namespace, route: str, shape: tuple[int, int], env: dict[str, str]) -> dict[str, Any]:
  label = f"generated-{route}-{shape[0]}x{shape[1]}"
  row: dict[str, Any] = {"route": label, "family": "generated", "shape": f"{shape[0]}x{shape[1]}", "env": route}
  with patched_env(env):
    la = lifecycle_args(args, shape)
    try:
      insts, meta = life._generated_active_insts(la, shape)
      report = life._report(label, insts, meta, full_rows=False)
      row.update(structural_fields(report))
    except Exception as e:
      row.update({"structure_status": type(e).__name__, "structure_error": str(e),
                  "structure_traceback_tail": traceback.format_exc().splitlines()[-8:]})
    if not args.structural_only:
      try:
        timed = _run_config(args.m, args.n, args.k, shape[0], shape[1], args.loc, args.unr, pin_clock=args.pin_clock)
        row.update({"status": timed.get("status"), "tflops": timed.get("tflops", 0.0), "ms_min": timed.get("ms_min"),
                    "rel_rmse": timed.get("rel_rmse"), "message": timed.get("message", "")})
      except Exception as e:
        row.update({"status": type(e).__name__, "tflops": 0.0, "message": str(e),
                    "timing_traceback_tail": traceback.format_exc().splitlines()[-8:]})
  return row


def hand_row(args: argparse.Namespace, shape: tuple[int, int]) -> dict[str, Any]:
  label = f"hand-lds2-{shape[0]}x{shape[1]}"
  row: dict[str, Any] = {"route": label, "family": "hand", "shape": f"{shape[0]}x{shape[1]}", "env": "hand-lds2"}
  la = lifecycle_args(args, shape)
  try:
    insts, meta = life._hand_insts("hand-lds2", la)
    report = life._report(label, insts, meta, full_rows=False)
    row.update(structural_fields(report))
  except Exception as e:
    row.update({"structure_status": type(e).__name__, "structure_error": str(e),
                "structure_traceback_tail": traceback.format_exc().splitlines()[-8:]})
  return row


def print_table(rows: list[dict[str, Any]]) -> None:
  cols = ["route", "status", "tflops", "wmma_count", "inst_per_wmma", "waitcnt_per_wmma",
          "global_b128_per_wmma", "ds_store_b128_per_wmma", "ds_load_b128_per_wmma",
          "between_global_regions", "future_slot_before_compute", "dbuf_d7_ok"]
  print("| " + " | ".join(cols) + " |")
  print("|" + "|".join(["---"] * len(cols)) + "|")
  for r in rows:
    vals = []
    for c in cols:
      v = r.get(c, "")
      if isinstance(v, float): v = round(v, 3)
      vals.append(str(v))
    print("| " + " | ".join(vals) + " |")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--routes", default="generated-direct,generated-kmajor,hand-lds2",
                  help="comma list: generated-direct,generated-kmajor,hand-lds2")
  ap.add_argument("--shapes", default="2,2", help="semicolon-separated shapes, e.g. '2,2;4,2;2,4'")
  ap.add_argument("--m", type=int, default=512)
  ap.add_argument("--n", type=int, default=5120)
  ap.add_argument("--k", type=int, default=5120)
  ap.add_argument("--loc", type=int, default=0)
  ap.add_argument("--unr", type=int, default=2)
  ap.add_argument("--target", default="AMD:ISA:gfx1100")
  ap.add_argument("--waves-m", type=int, default=1)
  ap.add_argument("--waves-n", type=int, default=1)
  ap.add_argument("--bk", type=int, default=32)
  ap.add_argument("--pad", type=int, default=0)
  ap.add_argument("--dbuf", type=int, default=1)
  ap.add_argument("--tm", type=int, default=2)
  ap.add_argument("--tn", type=int, default=4)
  ap.add_argument("--pin-clock", action="store_true")
  ap.add_argument("--structural-only", action="store_true")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args()

  rows: list[dict[str, Any]] = []
  shapes = parse_shapes(args.shapes)
  route_names = [x.strip() for x in args.routes.split(",") if x.strip()]
  for shape in shapes:
    for route in route_names:
      if route == "generated-direct":
        rows.append(generated_row(args, "direct", shape, DIRECT_ENV))
      elif route == "generated-kmajor":
        rows.append(generated_row(args, "kmajor", shape, KMAJOR_LDS_ENV))
      elif route == "hand-lds2":
        rows.append(hand_row(args, shape))
      else:
        rows.append({"route": route, "shape": f"{shape[0]}x{shape[1]}", "status": "unknown-route"})

  payload = {"m": args.m, "n": args.n, "k": args.k, "loc": args.loc, "unr": args.unr,
             "routes": route_names, "shapes": [f"{a}x{b}" for a, b in shapes], "rows": rows}
  if args.json: print(json.dumps(payload, indent=2))
  else: print_table(rows)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
