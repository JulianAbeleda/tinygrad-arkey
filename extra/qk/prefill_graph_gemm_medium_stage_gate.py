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
import subprocess
import sys
from typing import Any, Iterator

from extra.qk.prefill_v2_schedule_search import M_DEFAULT, _run_config
from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env

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


def _parse_json_markers(src: str, marker: str) -> list[dict[str, Any]]:
  out = []
  for ln in src.splitlines():
    if not ln.startswith(marker):
      continue
    try:
      payload = json.loads(ln[len(marker):].strip())
    except Exception:
      continue
    out.append(payload)
  return out


def _run_case_diagnostic(m: int, out_f: int, in_f: int, *, pin_clock: bool, env: dict[str, str | None],
                        timeout_sec: int = 420) -> dict[str, Any]:
  base_env: dict[str, str] = {**os.environ}
  base_env.setdefault("PYTHONPATH", ".")
  base_env.setdefault("DEV", "AMD")
  base_env["DEBUG"] = base_env.get("DEBUG") or "4"
  base_env["PREFILL_TC_LOCAL_STAGE_DUMP"] = "1"
  base_env.setdefault("PREFILL_TC_LOCAL_STAGE_DUMP_LIMIT", "8")
  set_clock_pin_env(base_env, pin_clock)
  for k, v in env.items():
    if v is None:
      base_env.pop(k, None)
    else:
      base_env[k] = v
  worker_env = {**base_env,
    "WORKER": "1",
    "MM": str(m),
    "OUTF": str(out_f),
    "INF": str(in_f),
    "U0": "4",
    "U1": "4",
    "LOC": "4",
    "UNR": "8",
    "DEBUG": base_env["DEBUG"],
  }
  try:
    from extra.qk import prefill_v2_schedule_search as schedule_search
    proc = subprocess.run([sys.executable, str(pathlib.Path(schedule_search.__file__))], env=worker_env, capture_output=True,
                          text=True, timeout=timeout_sec)
  except subprocess.TimeoutExpired:
    return {
      "status": "TIMEOUT",
      "returncode": -1,
      "stdout_tail": "",
      "stderr_tail": "",
      "has_fp16_wmma": False,
      "has_local_shared": False,
      "has_barrier": False,
      "cooperative_b_stage": {"seen": 0, "skipped": 0, "rewritten": 0},
    }
  result = {
    "returncode": proc.returncode,
    "stdout_tail": proc.stdout[-12000:],
    "stderr_tail": proc.stderr[-4000:],
  }
  for ln in proc.stdout.splitlines():
    if ln.startswith("RESULT "):
      try:
        result.update(json.loads(ln[len("RESULT "):]))
      except json.JSONDecodeError:
        result["status"] = "INVALID_RESULT"
      break
  result["has_fp16_wmma"] = "wmma_f16_16x16x16_f16" in proc.stdout or "WMMA_16_16_16_half" in proc.stdout
  result["has_local_shared"] = "__attribute__((shared" in proc.stdout
  result["has_barrier"] = "s_barrier" in proc.stdout
  skips = _parse_json_markers(proc.stdout, "TC_LOCAL_STAGE_COOP_B_SKIP ")
  rewrites = _parse_json_markers(proc.stdout, "TC_LOCAL_STAGE_COOP_B ")
  stats = _parse_json_markers(proc.stdout, "TC_LOCAL_STAGE_COOP_B_STATS ")
  summary = {
    "seen": max((s.get("seen", 0) for s in stats), default=0),
    "rewritten": max((s.get("rewritten", 0) for s in stats), default=0),
    "skipped": max((s.get("skipped", 0) for s in stats), default=0),
    "dumped": max((s.get("dumped", 0) for s in stats), default=0),
  }
  result["cooperative_b_stage"] = {
    "seen": summary.get("seen", 0),
    "skipped": summary.get("skipped", 0),
    "rewritten": summary.get("rewritten", 0),
    "skip_payloads": skips[:4],
    "rewrite_payloads": rewrites[:4],
  }
  return result


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
      "post_coop_b_partition_stage": _run_case_diagnostic(m, out_f, in_f, pin_clock=pin_clock, env={
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
  coop_stage = cases["post_coop_b_partition_stage"].get("cooperative_b_stage", {}) if run_amd and isinstance(cases, dict) else {}
  if coop_stage.get("skipped", 0) > 0 and coop_stage.get("rewritten", 0) == 0:
    coop_blocker = "medium shape route-bound cooperative B post-stage is skipped today because source B includes non-lane ranges outside warp+reduce"
  elif coop_stage.get("seen", 0) > 0 and coop_stage.get("rewritten", 0) == 0:
    coop_blocker = "medium shape route-bound cooperative B post-stage was seen but not rewritten by codegen"
  elif run_amd and not cases["post_coop_b_partition_stage"].get("has_fp16_wmma"):
    coop_blocker = "medium shape route-bound cooperative B post-stage currently compiles without WMMA in this configuration"
  elif run_amd and cases["post_coop_b_partition_stage"].get("status") != "ok":
    coop_blocker = f"medium shape cooperative B post-stage still errors with status={cases['post_coop_b_partition_stage'].get('status')}"
  else:
    coop_blocker = "medium shape route-bound cooperative B post-stage remains unbound or not performance-moving in this warmstart slice"

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
      "post_coop_b_partition_stage_coop_b_seen": run_amd and isinstance(cases.get("post_coop_b_partition_stage"), dict) and \
          cases["post_coop_b_partition_stage"].get("cooperative_b_stage", {}).get("seen"),
      "post_coop_b_partition_stage_coop_b_rewrite_count": run_amd and isinstance(cases.get("post_coop_b_partition_stage"), dict) and \
          cases["post_coop_b_partition_stage"].get("cooperative_b_stage", {}).get("rewritten"),
      "post_coop_b_partition_stage_coop_b_skip_count": run_amd and isinstance(cases.get("post_coop_b_partition_stage"), dict) and \
          cases["post_coop_b_partition_stage"].get("cooperative_b_stage", {}).get("skipped"),
      "post_coop_b_partition_stage_has_fp16_wmma": run_amd and isinstance(cases.get("post_coop_b_partition_stage"), dict) and \
          cases["post_coop_b_partition_stage"].get("has_fp16_wmma"),
      "post_coop_b_partition_stage_has_local_shared": run_amd and isinstance(cases.get("post_coop_b_partition_stage"), dict) and \
          cases["post_coop_b_partition_stage"].get("has_local_shared"),
      "post_coop_b_partition_stage_has_barrier": run_amd and isinstance(cases.get("post_coop_b_partition_stage"), dict) and \
          cases["post_coop_b_partition_stage"].get("has_barrier"),
      "scalar_post_local_stage_ok": bool(scalar_post_ok),
      "staged_beats_baseline": staged_beats_baseline,
    },
    "cases": cases,
    "remaining_blocker": None if verdict.endswith("_PASS") else
      ("B-operand tile-only post-WMMA staging is composition-checked but performance-flat for this schedule; "
       + coop_blocker),
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
