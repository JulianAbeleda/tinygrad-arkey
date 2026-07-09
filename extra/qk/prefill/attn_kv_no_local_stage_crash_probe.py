#!/usr/bin/env python3
"""Subprocess crash probe for S10 attn_kv generated pipe no-local-stage.

The composed whole-prefill path moved from a COMGR LDS overflow to signal 139
after selecting `generated_pipe_no_local_stage` for attn_kv. This probe isolates
that boundary by running the existing route-sampled correctness helper in child
processes, so a segfault classifies as data instead of killing the parent.
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CASE_SPECS: tuple[dict[str, Any], ...] = (
  {
    "name": "attn_kv_generated_no_local_stage",
    "role": "attn_kv", "m": 512, "n": 1024, "k": 4096,
    "target": "AMD:ISA:gfx1100",
    "env": {"PREFILL_WMMA_PIPE_PRIMITIVE": "1", "PREFILL_DBUF": "1", "PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE": "1"},
    "expected_route": "generated_pipe_no_local_stage",
  },
  {
    "name": "attn_kv_generated_no_local_stage_hip",
    "role": "attn_kv", "m": 512, "n": 1024, "k": 4096,
    "target": "AMD",
    "env": {"PREFILL_WMMA_PIPE_PRIMITIVE": "1", "PREFILL_DBUF": "1", "PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE": "1"},
    "expected_route": "generated_pipe_no_local_stage",
  },
  {
    "name": "attn_kv_raw_fallback_safety",
    "role": "attn_kv", "m": 512, "n": 1024, "k": 4096,
    "target": "AMD:ISA:gfx1100",
    "env": {"PREFILL_WMMA_PIPE_PRIMITIVE": "1", "PREFILL_DBUF": "1", "PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE": "0"},
    "expected_route": "pipe_resource_gated_raw_fallback",
  },
  {
    "name": "attn_qo_generated_pipe_control",
    "role": "attn_qo", "m": 512, "n": 4096, "k": 4096,
    "target": "AMD:ISA:gfx1100",
    "env": {"PREFILL_WMMA_PIPE_PRIMITIVE": "1", "PREFILL_DBUF": "1", "PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE": "1"},
    "expected_route": "allow",
  },
)


def _child_code(case: dict[str, Any], sample_cols: int) -> str:
  return f"""
import json
from extra.qk.prefill_pipe_mvp_artifact import run_route_sample_correctness
out = run_route_sample_correctness(m={case['m']}, n={case['n']}, k={case['k']}, role={case['role']!r}, sample_cols={sample_cols}, target={case['target']!r})
print('ATTN_KV_NO_LOCAL_STAGE_CHILD_JSON ' + json.dumps(out, allow_nan=False))
"""


def _default_runner(case: dict[str, Any], *, sample_cols: int, timeout_s: int) -> dict[str, Any]:
  env = {**os.environ, **case["env"], "PYTHONPATH": str(ROOT)}
  proc = subprocess.run([sys.executable, "-c", _child_code(case, sample_cols)], cwd=ROOT, env=env,
                        text=True, capture_output=True, timeout=timeout_s)
  marker = "ATTN_KV_NO_LOCAL_STAGE_CHILD_JSON "
  payload = None
  for line in proc.stdout.splitlines():
    if line.startswith(marker):
      payload = json.loads(line[len(marker):])
  return {
    "returncode": proc.returncode,
    "signal": -proc.returncode if proc.returncode < 0 else None,
    "stdout_tail": proc.stdout.splitlines()[-40:],
    "stderr_tail": proc.stderr.splitlines()[-80:],
    "payload": payload,
  }


def _classify_case(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
  payload = run.get("payload")
  signal = run.get("signal")
  if signal is None and run.get("returncode", 0) < 0:
    signal = -run["returncode"]
  if signal == 11 or run.get("returncode") == 139:
    status = "signal_139"
  elif run.get("returncode") != 0:
    status = "process_error"
  elif not isinstance(payload, dict):
    status = "missing_json"
  elif payload.get("passed") is True:
    status = "correctness_pass"
  else:
    status = "correctness_fail"
  return {
    "name": case["name"],
    "role": case["role"],
    "shape": {"m": case["m"], "n": case["n"], "k": case["k"]},
    "target": case["target"],
    "expected_route": case["expected_route"],
    "env": case["env"],
    "status": status,
    "returncode": run.get("returncode"),
    "signal": signal,
    "payload_summary": None if not isinstance(payload, dict) else {
      "passed": payload.get("passed"),
      "finite": payload.get("finite"),
      "nonzero": payload.get("nonzero"),
      "rel_rmse": payload.get("rel_rmse"),
      "route_transport": payload.get("route_transport"),
      "warmstart_key_present_after_route": payload.get("warmstart_key_present_after_route"),
      "warmstart_stats": payload.get("warmstart_stats"),
    },
    "stdout_tail": run.get("stdout_tail", []),
    "stderr_tail": run.get("stderr_tail", []),
  }


def _verdict(cases: list[dict[str, Any]]) -> str:
  by_name = {c["name"]: c for c in cases}
  no_local = by_name.get("attn_kv_generated_no_local_stage", {})
  no_local_hip = by_name.get("attn_kv_generated_no_local_stage_hip", {})
  fallback = by_name.get("attn_kv_raw_fallback_safety", {})
  control = by_name.get("attn_qo_generated_pipe_control", {})
  if no_local.get("status") == "correctness_pass" and no_local_hip.get("status") in ("process_error", "signal_139"):
    return "S10_ATTN_KV_NO_LOCAL_STAGE_HIP_BACKEND_BLOCKED"
  if no_local.get("status") == "correctness_pass":
    return "S10_ATTN_KV_NO_LOCAL_STAGE_ISOLATED_PASS"
  if no_local.get("status") == "signal_139" and fallback.get("status") == "correctness_pass":
    return "S10_ATTN_KV_NO_LOCAL_STAGE_ISOLATED_SIGNAL_139"
  if no_local.get("status") == "signal_139" and control.get("status") == "correctness_pass":
    return "S10_ATTN_KV_NO_LOCAL_STAGE_ATTN_KV_SPECIFIC_SIGNAL_139"
  if no_local.get("status") == "signal_139":
    return "S10_ATTN_KV_NO_LOCAL_STAGE_SIGNAL_139_UNCLASSIFIED"
  return "S10_ATTN_KV_NO_LOCAL_STAGE_INCONCLUSIVE"


def build_report(*, execute: bool = False, sample_cols: int = 4, timeout_s: int = 120,
                 runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
  cases = []
  if execute:
    for case in CASE_SPECS:
      run = runner(case) if runner is not None else _default_runner(case, sample_cols=sample_cols, timeout_s=timeout_s)
      cases.append(_classify_case(case, run))
  else:
    cases = [{**case, "status": "not_run"} for case in CASE_SPECS]
  return {
    "schema": "s10-attn-kv-no-local-stage-crash-probe.v1",
    "execute": execute,
    "sample_cols": sample_cols,
    "cases": cases,
    "verdict": _verdict(cases) if execute else "S10_ATTN_KV_NO_LOCAL_STAGE_CRASH_PROBE_SCOPED",
    "classification_goal": "isolate signal 139 as attn_kv no-local-stage vs fallback/control/whole-prefill composition",
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--execute", action="store_true", help="run GPU child processes")
  ap.add_argument("--sample-cols", type=int, default=4)
  ap.add_argument("--timeout-s", type=int, default=120)
  ap.add_argument("--report", type=pathlib.Path, default=ROOT / "bench/prefill-s10-lds2-ownership/attn-kv-no-local-stage-crash-probe.json")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)
  report = build_report(execute=args.execute, sample_cols=args.sample_cols, timeout_s=args.timeout_s)
  report_path = args.report if args.report.is_absolute() else ROOT / args.report
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  if args.json:
    print(json.dumps(report, indent=2, allow_nan=False))
  else:
    print(f"{report['verdict']} report={report_path}")
    for case in report["cases"]:
      print(f"  {case['name']}: {case['status']}")
  return 0 if not args.execute or report["verdict"].endswith("_PASS") else 1


if __name__ == "__main__":
  raise SystemExit(main())
