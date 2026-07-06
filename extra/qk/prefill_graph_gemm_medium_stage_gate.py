#!/usr/bin/env python3
"""Medium-shape gate for generated TC LOCAL staging on warmstart schedules.

The tiny route-bound probe proves that generated `Ops.STAGE` can be attached to
an fp16 WMMA operand. This gate checks the next required step: whether that
staging composes with the real warmstart `OptOps.LOCAL` schedule used by 8B/14B
prefill-v2 GEMMs.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from contextlib import contextmanager
from typing import Any, Iterator

from extra.qk.prefill_v2_schedule_search import M_DEFAULT, _run_config
from extra.qk.timing_harness import add_clock_pin_arg

SCHEMA = "prefill-graph-gemm-medium-stage-gate.v1"
ARTIFACT_DIR = pathlib.Path("bench/prefill-graph-gemm-medium-stage")


@contextmanager
def _patched_env(updates: dict[str, str | None]) -> Iterator[None]:
  old = {k: os.environ.get(k) for k in updates}
  try:
    for k, v in updates.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    yield
  finally:
    for k, v in old.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v


def _run_case(m: int, out_f: int, in_f: int, *, pin_clock: bool, env: dict[str, str | None]) -> dict[str, Any]:
  with _patched_env(env):
    return _run_config(m, out_f, in_f, 4, 4, 4, 8, pin_clock=pin_clock)


def build_report(*, run_amd: bool = False, m: int = M_DEFAULT, out_f: int = 4096, in_f: int = 4096,
                 pin_clock: bool = False, artifact: bool = True) -> dict[str, Any]:
  cases: dict[str, Any]
  if run_amd:
    cases = {
      "baseline_table_local": _run_case(m, out_f, in_f, pin_clock=pin_clock, env={
        "PREFILL_TC_LOCAL_STAGE": None,
        "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": None,
        "PREFILL_TC_LOCAL_STAGE_POST": None,
        "PREFILL_TC_LOCAL_STAGE_TILE_ONLY": None,
        "PREFILL_TC_LOCAL_STAGE_SCALAR_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_LIMIT": None,
      }),
      "pre_wmma_stage_forced_local": _run_case(m, out_f, in_f, pin_clock=pin_clock, env={
        "PREFILL_TC_LOCAL_STAGE": "a",
        "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": "1",
        "PREFILL_TC_LOCAL_STAGE_POST": None,
        "PREFILL_TC_LOCAL_STAGE_TILE_ONLY": None,
        "PREFILL_TC_LOCAL_STAGE_SCALAR_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_LIMIT": None,
      }),
      "post_local_stage": _run_case(m, out_f, in_f, pin_clock=pin_clock, env={
        "PREFILL_TC_LOCAL_STAGE": "a",
        "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": None,
        "PREFILL_TC_LOCAL_STAGE_POST": "1",
        "PREFILL_TC_LOCAL_STAGE_TILE_ONLY": None,
        "PREFILL_TC_LOCAL_STAGE_SCALAR_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_LIMIT": None,
      }),
      "post_tile_b_stage": _run_case(m, out_f, in_f, pin_clock=pin_clock, env={
        "PREFILL_TC_LOCAL_STAGE": "b",
        "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": None,
        "PREFILL_TC_LOCAL_STAGE_POST": "1",
        "PREFILL_TC_LOCAL_STAGE_TILE_ONLY": "1",
        "PREFILL_TC_LOCAL_STAGE_SCALAR_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_LIMIT": None,
      }),
      "post_coop_b_partition_stage": _run_case(m, out_f, in_f, pin_clock=pin_clock, env={
        "PREFILL_TC_LOCAL_STAGE": "b",
        "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": None,
        "PREFILL_TC_LOCAL_STAGE_POST": None,
        "PREFILL_TC_LOCAL_STAGE_TILE_ONLY": None,
        "PREFILL_TC_LOCAL_STAGE_SCALAR_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_POST": "1",
        # This is a lowering diagnostic, not a perf attempt. One
        # rewrite is enough to prove whether route-bound cooperative stores
        # compose through CFG, late vectorization, and verifier lowering.
        "PREFILL_TC_LOCAL_STAGE_COOP_B_LIMIT": "1",
      }),
      "scalar_post_local_stage": _run_case(m, out_f, in_f, pin_clock=pin_clock, env={
        "PREFILL_TC_LOCAL_STAGE": "a",
        "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": None,
        "PREFILL_TC_LOCAL_STAGE_POST": None,
        "PREFILL_TC_LOCAL_STAGE_TILE_ONLY": None,
        "PREFILL_TC_LOCAL_STAGE_SCALAR_POST": "1",
        "PREFILL_TC_LOCAL_STAGE_COOP_B_POST": None,
        "PREFILL_TC_LOCAL_STAGE_COOP_B_LIMIT": None,
      }),
    }
  else:
    cases = {"skipped": "pass --run-amd to execute medium-shape staging cases"}

  baseline_ok = run_amd and cases["baseline_table_local"].get("status") == "ok"
  pre_ok = run_amd and cases["pre_wmma_stage_forced_local"].get("status") == "ok"
  post_ok = run_amd and cases["post_local_stage"].get("status") == "ok"
  post_tile_b_ok = run_amd and cases["post_tile_b_stage"].get("status") == "ok"
  post_coop_b_ok = run_amd and cases["post_coop_b_partition_stage"].get("status") == "ok"
  scalar_post_ok = run_amd and cases["scalar_post_local_stage"].get("status") == "ok"
  staged_ok = bool(pre_ok or post_ok or post_tile_b_ok or post_coop_b_ok or scalar_post_ok)
  staged_best = max((cases[k].get("tflops", 0.0) for k in ("pre_wmma_stage_forced_local", "post_local_stage",
                                                           "post_tile_b_stage", "post_coop_b_partition_stage",
                                                           "scalar_post_local_stage")
                     if isinstance(cases.get(k), dict)), default=0.0)
  baseline_tflops = cases["baseline_table_local"].get("tflops", 0.0) if run_amd else 0.0
  staged_beats_baseline = bool(staged_ok and staged_best > baseline_tflops * 1.05)

  verdict = "PREFILL_GRAPH_GEMM_MEDIUM_LOCAL_STAGE_PASS" if baseline_ok and staged_beats_baseline \
    else "PREFILL_GRAPH_GEMM_MEDIUM_LOCAL_STAGE_BLOCKED"
  report = {
    "schema": SCHEMA,
    "route_id": "prefill_v2_scheduler_matmul_default",
    "shape": {"m": m, "n": out_f, "k": in_f},
    "schedule": {"u0": 4, "u1": 4, "loc": 4, "unr": 8},
    "verdict": verdict,
    "evidence": {
      "run_amd": run_amd,
      "pin_clock": pin_clock,
      "baseline_table_local_ok": bool(baseline_ok),
      "pre_wmma_forced_local_ok": bool(pre_ok),
      "post_local_stage_ok": bool(post_ok),
      "post_tile_b_stage_ok": bool(post_tile_b_ok),
      "post_coop_b_partition_stage_ok": bool(post_coop_b_ok),
      "scalar_post_local_stage_ok": bool(scalar_post_ok),
      "staged_beats_baseline": staged_beats_baseline,
    },
    "cases": cases,
    "remaining_blocker": None if verdict.endswith("_PASS") else
      "B-operand tile-only post-WMMA staging composes with warmstart LOCAL schedules but is performance-flat; the first route-bound cooperative B-partition attempt passes the previous CFG cycle but is rejected later as an invalid vector local-store shape; A/both/final-scalar staging remain wrong or unbuildable",
  }
  if artifact:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "latest.json").write_text(json.dumps(report, indent=2))
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--run-amd", action="store_true")
  add_clock_pin_arg(ap)
  ap.add_argument("--M", type=int, default=M_DEFAULT)
  ap.add_argument("--out-f", type=int, default=4096)
  ap.add_argument("--in-f", type=int, default=4096)
  ap.add_argument("--no-artifact", action="store_true")
  args = ap.parse_args(argv)
  report = build_report(run_amd=args.run_amd, m=args.M, out_f=args.out_f, in_f=args.in_f,
                        pin_clock=args.pin_clock, artifact=not args.no_artifact)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
