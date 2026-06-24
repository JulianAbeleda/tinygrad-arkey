#!/usr/bin/env python3
"""Periodic decode lifecycle recheck orchestrator.

This script executes the full lifecycle recheck bundle and emits a compact drift
artifact comparing the current run against the prior `latest.json` snapshot.

Use cases:
- one-off periodic refresh (default): run bundle and diff against latest
- compare-only audit: `--compare-only` skips the expensive run and only computes
drift against the previous snapshot.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECHECK_SCRIPT = ROOT / "extra/qk_decode_lifecycle_recheck_bundle.py"
RECHECK_OUT = ROOT / "bench/qk-decode-lifecycle-recheck-bundle"


def _python() -> str:
  local = ROOT / ".venv/bin/python3"
  if local.exists():
    return str(local)
  return sys.executable


def _run(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> int:
  cp = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
  return cp.returncode


def _load_json(path: pathlib.Path) -> dict[str, Any]:
  with path.open() as fh:
    return json.load(fh)


def _load_latest_bundle(latest_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
  latest = _load_json(latest_path)
  bundle = pathlib.Path(latest["bundle"])
  snapshot = _load_json(bundle)
  return bundle, snapshot


def _extract_current_min_delta(bundle: dict[str, Any]) -> float | None:
  try:
    return float(bundle["sweeps"]["current"]["sweep_summary"]["delta_pct_min"])
  except Exception:
    pass
  try:
    deltas = [
      float(v)
      for v in bundle["sweeps"]["current"]["sweep_summary"].get("delta_pct_by_ctx", {}).values()
    ]
    return min(deltas) if deltas else None
  except Exception:
    return None


def _extract_decision(bundle: dict[str, Any]) -> str:
  return str(bundle.get("decision", {}).get("verdict", "UNKNOWN"))


def _extract_gate_ok(bundle: dict[str, Any], key: str) -> str:
  checks = bundle.get("decision", {}).get("checks", {})
  return str(checks.get(key, {}).get("ok", False)).upper()


def _extract_unknown_label(bundle: dict[str, Any], key: str) -> str:
  checks = bundle.get("decision", {}).get("checks", {})
  return str(checks.get(key, {}).get("summary", {}).get("label", "UNKNOWN"))


def _extract_ctx_delta_points(bundle: dict[str, Any], sweep: str) -> dict[str, float]:
  result: dict[str, float] = {}
  try:
    rows = bundle["sweeps"][sweep]["sweep_summary"].get("delta_pct_by_ctx", {})
    for k, v in rows.items():
      try:
        result[str(k)] = float(v)
      except Exception:
        pass
  except Exception:
    pass
  return result


def _extract_ctx_a_tok(bundle: dict[str, Any], sweep: str) -> dict[str, float]:
  result: dict[str, float] = {}
  try:
    deltas = bundle["sweeps"][sweep]["wd_by_ctx"]["delta_A_vs_B"]
    for k, v in deltas.items():
      try:
        result[str(k)] = float(v["A_tok_s"])
      except Exception:
        pass
  except Exception:
    pass
  return result


def _extract_ctx_b_tok(bundle: dict[str, Any], sweep: str) -> dict[str, float]:
  result: dict[str, float] = {}
  try:
    deltas = bundle["sweeps"][sweep]["wd_by_ctx"]["delta_A_vs_B"]
    for k, v in deltas.items():
      try:
        result[str(k)] = float(v["B_tok_s"])
      except Exception:
        pass
  except Exception:
    pass
  return result


def _drift_value(current: float | None, previous: float | None) -> float | None:
  if current is None or previous is None:
    return None
  return current - previous


def _make_unique_run_id(out_root: pathlib.Path, run_id: str) -> str:
  candidate = run_id
  suffix = 1
  while (out_root / f"decode-lifecycle-recheck-{candidate}").exists():
    candidate = f"{run_id}-{suffix:02d}"
    suffix += 1
  return candidate


def _run_bundle(out_root: pathlib.Path, run_id: str | None, timeout_sec: int | None) -> tuple[pathlib.Path, str, dict[str, Any]]:
  if not run_id:
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  else:
    run_id = _make_unique_run_id(out_root, run_id)

  cmd = [_python(), str(RECHECK_SCRIPT), "--out-root", str(out_root)]
  if run_id:
    cmd.extend(["--run-id", run_id])

  env = os.environ.copy()
  env.setdefault("PYTHONPATH", ".")
  env.setdefault("DEV", "AMD")
  env.setdefault("JIT", "1")

  if timeout_sec:
    cmd = ["timeout", str(timeout_sec)] + cmd

  rc = _run(cmd, ROOT, env)
  if rc != 0:
    raise RuntimeError(f"decode lifecycle recheck bundle exited with code {rc}")

  latest = _load_json(out_root / "latest.json")
  run = str(latest.get("run")) if latest.get("run") is not None else out_root / "unknown"
  bundle = pathlib.Path(latest["bundle"])
  snapshot = _load_json(bundle)
  return bundle, run, snapshot


def _build_diff(
  current: dict[str, Any],
  previous: dict[str, Any] | None,
  run_id: str,
  prev_run_id: str | None,
) -> dict[str, Any]:
  current_min = _extract_current_min_delta(current)
  prev_min = _extract_current_min_delta(previous) if previous else None

  diff: dict[str, Any] = {
    "protocol": "DECODE_LIFECYCLE_PERIODIC_DIFF",
    "date_local": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    "run": {
      "current": run_id,
      "previous": prev_run_id,
    },
    "verdict": {
      "current": _extract_decision(current),
      "previous": _extract_decision(previous) if previous else None,
      "changed": _extract_decision(current) != (_extract_decision(previous) if previous else None),
    },
    "current_ctx": {
      "delta_min_pct": current_min,
      "delta_min_delta_from_previous": _drift_value(current_min, prev_min),
      "points_current_pct": _extract_ctx_delta_points(current, "current"),
      "points_previous_pct": _extract_ctx_delta_points(previous, "current") if previous else {},
    },
    "pillars": {
      "oracle_gate_pre_ok": _extract_gate_ok(current, "oracle_gate_pre"),
      "oracle_gate_post_ok": _extract_gate_ok(current, "oracle_gate_post"),
      "unknown_lockstep_pre": _extract_unknown_label(current, "unknown_lockstep_pre"),
      "unknown_lockstep_post": _extract_unknown_label(current, "unknown_lockstep_post"),
    },
    "sweep_payload": {
      "current_A_tok_s": _extract_ctx_a_tok(current, "current"),
      "current_B_tok_s": _extract_ctx_b_tok(current, "current"),
      "long_A_tok_s": _extract_ctx_a_tok(current, "long"),
      "long_B_tok_s": _extract_ctx_b_tok(current, "long"),
      "alternative_capture_A_tok_s": _extract_ctx_a_tok(current, "alternative_capture"),
      "alternative_capture_B_tok_s": _extract_ctx_b_tok(current, "alternative_capture"),
    },
    "alerts": [],
  }

  alerts: list[str] = []
  if diff["verdict"]["current"] != "DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS":
    alerts.append("decision_regressed_or_review_required")
  if prev_run_id and previous and _extract_decision(previous) != "DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS" and _extract_decision(current) == "DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS":
    alerts.append("recovered_to_pass")
  if current_min is not None and current_min < 10.0:
    alerts.append("current_ctx_delta_min_below_10pct")
  if current_min is not None and prev_min is not None and current_min < (prev_min - 1.0):
    alerts.append("current_ctx_min_delta_drop_gt_1pct")
  if diff["pillars"].get("unknown_lockstep_pre") != "DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN":
    alerts.append("unknown_lockstep_pre_not_proven")
  if diff["pillars"].get("unknown_lockstep_post") != "DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN":
    alerts.append("unknown_lockstep_post_not_proven")
  if diff["pillars"].get("oracle_gate_pre_ok") != "TRUE":
    alerts.append("oracle_gate_pre_failed")
  if diff["pillars"].get("oracle_gate_post_ok") != "TRUE":
    alerts.append("oracle_gate_post_failed")

  diff["alerts"] = alerts
  return diff


def _write_md(diff: dict[str, Any], path: pathlib.Path) -> None:
  alerts = [f"- {a}" for a in diff.get("alerts", [])]
  if not alerts:
    alerts = ["- none"]
  lines = [
    "# Decode Lifecycle Recheck Periodic Drift",
    f"- run: {diff['run']['current']}",
    f"- previous: {diff['run']['previous'] or 'none'}",
    f"- verdict: {diff['verdict']['current']}",
    "",
    "## Quick guardrail summary",
    f"- oracle_gate_pre_ok: {diff['pillars']['oracle_gate_pre_ok']}",
    f"- oracle_gate_post_ok: {diff['pillars']['oracle_gate_post_ok']}",
    f"- unknown_lockstep_pre: {diff['pillars']['unknown_lockstep_pre']}",
    f"- unknown_lockstep_post: {diff['pillars']['unknown_lockstep_post']}",
    f"- current_ctx min delta%: {diff['current_ctx'].get('delta_min_pct')}",
    f"- current_ctx min delta change vs previous: {diff['current_ctx'].get('delta_min_delta_from_previous')}",
    "",
    "## Alerts",
    *alerts,
    "",
    "## Outputs",
    "- periodic_diff.json",
    "- periodic_diff.md",
  ]
  path.write_text("\n".join(lines) + "\n")


def _main() -> int:
  parser = argparse.ArgumentParser(description="Run decode lifecycle recheck bundle + periodic drift diff")
  parser.add_argument("--out-root", default=str(RECHECK_OUT), help="Recheck bundle root")
  parser.add_argument("--run-id", default=None, help="Optional explicit run-id for bundle")
  parser.add_argument("--compare-only", action="store_true", help="Skip expensive bundle run and only diff latest vs previous")
  parser.add_argument("--timeout-sec", type=int, default=None, help="Optional timeout wrapper (seconds) for bundle")
  args = parser.parse_args()

  out_root = pathlib.Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)

  latest_path = out_root / "latest.json"
  previous = None
  prev_run_id = None

  if latest_path.exists():
    latest = _load_json(latest_path)
    prev_run_id = latest.get("run")
    _, previous = _load_latest_bundle(latest_path)

  if not args.compare_only:
    bundle_path, run_id, current = _run_bundle(out_root, args.run_id, args.timeout_sec)
    out_dir = bundle_path.parent
  else:
    if not latest_path.exists():
      raise RuntimeError("--compare-only requires an existing latest.json in out-root")
    _, current = _load_latest_bundle(latest_path)
    run_id = prev_run_id or "unknown"
    out_dir = pathlib.Path(_load_json(latest_path)["bundle"]).resolve().parent

  diff = _build_diff(current, previous, run_id, prev_run_id)
  diff_json = out_dir / "periodic_diff.json"
  diff_md = out_dir / "periodic_diff.md"
  diff_json.write_text(json.dumps(diff, indent=2) + "\n")
  _write_md(diff, diff_md)

  print(f"PERIODIC_BASELINE_DIFF run={run_id} prev={prev_run_id}")
  print(f"verdict: current={diff['verdict']['current']} previous={diff['verdict']['previous']}")
  print(f"current_ctx_min_delta={diff['current_ctx'].get('delta_min_pct')} ({diff['current_ctx'].get('delta_min_delta_from_previous'):+})")
  print(f"artifact: {diff_json}")
  print(f"summary: {diff_md}")
  return 0


if __name__ == "__main__":
  raise SystemExit(_main())
