#!/usr/bin/env python3
"""Decode lifecycle recheck bundle runner (authority + W==D + unknown-lockstep).

This tool executes a periodic decode bundle across benchmark capture variants and
stores a single handoff artifact folder with three pillars:

1) correctness/reproducibility
   - oracle-gate integrity (route + materialization + ISA + token correctness)
   - unknown-bucket lockstep (preflight + postflight)

2) W==D performance by context
   - interleaved A/B W==D sweeps for current-context + long-context + alternative capture mode

3) unknown closure
   - unknown-lockstep residual/mapping evidence snapshot

Only existing benchmark scripts are orchestrated; no new modeling logic is introduced.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "bench/qk-decode-lifecycle-recheck-bundle"
RUNTIME_RESULT = ROOT / "bench/qk-decode-runtime-overhead/result.json"
SLOPE_OUT = ROOT / "bench/qk-decode-ctx-slope-audit/wd_by_ctx.json"
LOCKSTEP_DIR = ROOT / "bench/qk-decode-unknown-bucket-lockstep-20260624"
BASELINE_ORACLE = ROOT / "bench/qk-decode-search-readiness/baseline_oracle.json"
LOCKSTEP_CTXS = [512, 1024, 2048, 4096]
RUNTIME_CTX_MAX = 4096
SCRIPTS = {
  "gate": ROOT / "extra/qk_decode_search_gate.py",
  "ctx_slope": ROOT / "extra/qk_ctx_slope_driver.py",
  "unknown_lockstep": ROOT / "extra/qk_decode_unknown_bucket_lockstep_audit.py",
}


def _to_str_env(env: dict[str, str]) -> str:
  return " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k and v is not None)


def _run(cmd: list[str], env: dict[str, str], label: str) -> tuple[int, str, str]:
  cp = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, **env})
  print(f"[{label}] return={cp.returncode} cmd={' '.join(cmd)}")
  if cp.stdout:
    print(cp.stdout)
  if cp.stderr:
    print(cp.stderr, file=sys.stderr)
  return cp.returncode, cp.stdout, cp.stderr


def _parse_gate(stdout: str) -> dict[str, Any]:
  for line in stdout.splitlines()[::-1]:
    if line.startswith("RESULT "):
      try:
        return json.loads(line[len("RESULT "):])
      except Exception:
        pass
  raise RuntimeError("gate output did not contain a RESULT json payload")


def _copy_json(src: pathlib.Path, dst: pathlib.Path, required: bool = False) -> None:
  if not src.exists():
    if required:
      raise FileNotFoundError(f"required artifact missing: {src}")
    return
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
  with path.open() as fh:
    return json.load(fh)


def _normalize_ctxs(csv: str) -> list[int]:
  vals = [x.strip() for x in csv.split(",") if x.strip()]
  return [int(v) for v in vals]


def _normalize_runtime_ctxs(csv: str) -> list[int]:
  requested = _normalize_ctxs(csv)
  supported = [c for c in requested if c <= RUNTIME_CTX_MAX]
  if not supported:
    raise ValueError(f"no supported runtime ctxs from {requested}; max supported={RUNTIME_CTX_MAX}")
  if len(supported) != len(requested):
    dropped = [c for c in requested if c > RUNTIME_CTX_MAX]
    print(f"[protocol] runtime contexts trimmed to supported scope: dropped={dropped}, retained={supported}")
  return list(dict.fromkeys(supported))


def _normalize_lockstep_ctxs(csv: str) -> list[int]:
  requested = _normalize_ctxs(csv)
  filtered = [c for c in requested if c in LOCKSTEP_CTXS]
  if not filtered:
    raise ValueError(f"no valid lockstep contexts from {requested}; supported={LOCKSTEP_CTXS}")
  if len(filtered) != len(requested):
    dropped = [c for c in requested if c not in LOCKSTEP_CTXS]
    print(f"[protocol] lockstep contexts trimmed to supported scope: dropped={dropped}, retained={filtered}")
  # preserve requested order while de-duplicating for stable reproducibility.
  return list(dict.fromkeys(filtered))


def _slope_summary(bundle: dict[str, Any], dst: pathlib.Path) -> dict[str, Any]:
  # Keep the key W==D signal easy to consume by handoff consumers.
  out = {"ckpts": bundle["wd_by_ctx"]["ckpts"], "reps": bundle["wd_by_ctx"]["reps"],
         "interleave": bundle["wd_by_ctx"]["interleave"], "configs": {}}
  for cname, crows in bundle["wd_by_ctx"]["configs"].items():
    out["configs"][cname] = {
      "tok_s_median": {ck: crows[ck]["tok_s"]["median"] for ck in crows},
      "tok_s_mean": {ck: crows[ck]["tok_s"]["mean"] for ck in crows},
      "delta_vs_B": bundle["wd_by_ctx"]["delta_A_vs_B"] if cname == "A_whole_default" else None,
    }
  if "kernel_attribution" in bundle["wd_by_ctx"]:
    out["kernel_attribution"] = bundle["wd_by_ctx"]["kernel_attribution"]
  return out


def run_gate(env: dict[str, str], out_dir: pathlib.Path, label: str) -> dict[str, Any]:
  run_env = {
    "DEV": "AMD",
    "JIT": "1",
    "PYTHONPATH": ".",
    "Q4K_GEMV_WARP": "1",
    "Q4K_GEMV_WARP_DOWN": "1",
    "Q4K_GEMV_WARP_PROJ": "0",
    **env,
  }
  rc, out, err = _run([str(ROOT / ".venv/bin/python"), str(SCRIPTS["gate"]), "--oracle-tokens", str(BASELINE_ORACLE)],
                      run_env, label=label)
  if rc != 0:
    raise RuntimeError(f"oracle gate failed ({label}): {err}")
  result = _parse_gate(out)
  (out_dir / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
  return result


def run_unknown_lockstep(ctx_csv: str, run_env: dict[str, str], out_dir: pathlib.Path, label: str) -> dict[str, Any]:
  env = {**run_env, "DEV": "AMD", "PYTHONPATH": "."}
  env["JIT"] = env.get("JIT", "1")
  rc, out, err = _run([str(ROOT / ".venv/bin/python"), str(SCRIPTS["unknown_lockstep"]), "--contexts", ctx_csv],
                      env, label=label)
  if rc != 0:
    raise RuntimeError(f"unknown lockstep failed ({label}): {err}")
  snap = {"command_env": env, "stdout": out, "stderr": err}
  snap_path = out_dir / f"{label}.json"
  snap_path.write_text(json.dumps(snap, indent=2) + "\n")

  # Snapshot key lockstep artifacts for this run ID.
  copy_dir = out_dir / "artifacts"
  for name in [
    "decision.json",
    "latest.json",
    "math_assertions.json",
    "residual_unmapped_by_ctx.json",
    "unknown_bucket_source_map.json",
    "summary.md",
  ]:
    _copy_json(LOCKSTEP_DIR / name, copy_dir / name, required=name == "decision.json")
  return _read_json(LOCKSTEP_DIR / "decision.json")


def run_slope_variant(ctx_csv: str, extra_env: dict[str, str], out_dir: pathlib.Path, label: str) -> dict[str, Any]:
  env = {
    "DEV": "AMD",
    "JIT": "1",
    "PYTHONPATH": ".",
    "QK_CKPTS": ctx_csv,
    "Q4K_GEMV_WARP": "1",
    "Q4K_GEMV_WARP_DOWN": "1",
    "Q4K_GEMV_WARP_PROJ": "0",
    **extra_env,
  }
  rc, out, err = _run([str(ROOT / ".venv/bin/python"), str(SCRIPTS["ctx_slope"])], env, label=label)
  if rc != 0:
    raise RuntimeError(f"ctx slope run failed ({label}): {err}")
  if not SLOPE_OUT.exists():
    raise RuntimeError(f"expected slope artifact not found: {SLOPE_OUT}")
  bundle = {"command_env": env, "command_output": out, "command_stderr": err}
  slope_artifact = _read_json(SLOPE_OUT)
  bundle["wd_by_ctx"] = slope_artifact
  out_dir.mkdir(parents=True, exist_ok=True)
  out_dir.joinpath("wd_by_ctx.json").write_text(json.dumps(slope_artifact, indent=2) + "\n")
  # keep short hand for handoff consumers
  bundle["handoff_summary"] = _slope_summary(bundle, out_dir / "wd_by_ctx.json")
  return bundle


def _slope_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  if not rows:
    return {"status": "missing"}
  deltas = []
  for r in rows:
    if "A_tok_s" in r and "B_tok_s" in r:
      b = float(r["B_tok_s"])
      if b > 0:
        deltas.append(round((float(r["A_tok_s"]) - b) / b * 100.0, 2))
  return {
    "ctx_points": [r["ctx"] for r in rows],
    "delta_pct_by_ctx": {str(r["ctx"]): r["delta_pct_tok_s"] for r in rows if "delta_pct_tok_s" in r},
    "delta_pct_min": min(deltas) if deltas else None,
    "delta_pct_max": max(deltas) if deltas else None,
    "delta_pct_mean": round(statistics.mean(deltas), 3) if deltas else None,
  }


def build_decision(bundle: dict[str, Any]) -> dict[str, Any]:
  gate_pre = bundle["correctness"]["gate_pre"]
  gate_post = bundle["correctness"]["gate_post"]
  lock_pre = bundle["correctness"]["unknown_lockstep_pre"]
  lock_post = bundle["correctness"]["unknown_lockstep_post"]
  gate_ok = all(item.get("verdict") == "PASS" for item in (gate_pre, gate_post))
  lock_ok = (lock_pre.get("label") == "DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN"
             and lock_post.get("label") == "DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN")
  current = bundle["sweeps"]["current"]
  delta_ok = current["sweep_summary"].get("delta_pct_min", 0) >= 0.0

  if gate_ok and lock_ok and delta_ok:
    verdict = "DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS"
  elif not gate_ok:
    verdict = "DECODE_GATE_REVIEW_REQUIRED"
  elif not lock_ok:
    verdict = "DECODE_UNKNOWN_CLOSURE_REVIEW_REQUIRED"
  else:
    verdict = "DECODE_PERF_DELTA_REVIEW_REQUIRED"

  return {
    "date": bundle["authority"]["date_local"],
    "phase": "DECODE_LIFECYCLE_RECHECK_BUNDLE_DECISION",
    "verdict": verdict,
    "checks": {
      "oracle_gate_pre": {"ok": gate_pre.get("verdict") == "PASS", "summary": gate_pre},
      "oracle_gate_post": {"ok": gate_post.get("verdict") == "PASS", "summary": gate_post},
      "unknown_lockstep_pre": {"ok": lock_pre.get("label") == "DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN", "summary": lock_pre},
      "unknown_lockstep_post": {"ok": lock_post.get("label") == "DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN", "summary": lock_post},
      "current_ctx_A_beats_B": {"ok": delta_ok, "summary": current["sweep_summary"]},
    },
    "next_step": ("DECODE_ORACLE_EXPLANATION_BASELINE_UPDATE" if verdict == "DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS"
                  else "DECODE_RECONCILE_AND_RE_RUN"),
    "notes": ("Bundle is fully closed on gate integrity + unknown closure; update decode oracle baseline snapshot."
              if verdict == "DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS"
              else "One or more pillars require review; keep baseline snapshot as reference and block default changes."),
  }


def run(args: argparse.Namespace) -> int:
  run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  out_root = pathlib.Path(args.out_root)
  out_dir = out_root / f"decode-lifecycle-recheck-{run_id}"
  out_dir.mkdir(parents=True, exist_ok=True)
  correctness_dir = out_dir / "correctness"
  correctness_dir.mkdir(parents=True, exist_ok=True)

  current_contexts = _normalize_runtime_ctxs(args.current_contexts)
  long_contexts = _normalize_runtime_ctxs(args.long_contexts)
  baseline_env = {
    "DECODE_ATTN_KV_IDENTITY": "1",
    "DECODE_ATTN_AMDGCN_TILE": "1",
  }
  ctx_sweep_all = _normalize_lockstep_ctxs(args.lockstep_contexts)

  authority = {
    "date_local": run_id,
    "command": "extra/qk_decode_lifecycle_recheck_bundle.py",
    "phase": "DECODE_LIFECYCLE_RECHECK_BUNDLE",
    "repo_root": str(ROOT),
    "user": os.environ.get("USER"),
    "python": str((ROOT / ".venv/bin/python").resolve()),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
    "git_status": subprocess.check_output(["git", "status", "--short"], cwd=str(ROOT), text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=str(ROOT), text=True).strip(),
    "protocol": [
      "A/B W==D interleaved sweep: current-context + long-context + legacy capture env",
      "oracle gate pre/post with baseline oracle tokens",
      "unknown lockstep pre/post with shared contexts",
      "single handoff bundle snapshot",
    ],
    "contexts": {
      "current": current_contexts,
      "long": long_contexts,
      "lockstep": ctx_sweep_all,
    },
  }

  runs = {
    "gate_pre": run_gate(baseline_env, correctness_dir, "gate_pre"),
    "gate_post": None,
  }

  lockstep_pre = run_unknown_lockstep(",".join(str(c) for c in ctx_sweep_all), baseline_env,
                                     correctness_dir, "unknown_lockstep_pre")
  sweeps: dict[str, Any] = {}

  # Current-context sweep with default route (A/B) interleaved.
  current = run_slope_variant(",".join(str(c) for c in current_contexts), baseline_env, out_dir / "throughput/current_context", "ctx_slope_current")
  current_summary = _slope_summary(current, out_dir / "throughput/current_context" / "summary.json")
  current_delta = _slope_delta_summary([{"ctx": ck, **vals} for ck, vals in current["wd_by_ctx"]["delta_A_vs_B"].items()]) if "delta_A_vs_B" in current["wd_by_ctx"] else {}

  # Long-context sweep (single point) for current route at high context.
  long = run_slope_variant(",".join(str(c) for c in long_contexts), baseline_env, out_dir / "throughput/long_context", "ctx_slope_long")
  long_summary = _slope_summary(long, out_dir / "throughput/long_context" / "summary.json")
  long_delta = _slope_delta_summary([{"ctx": ck, **vals} for ck, vals in long["wd_by_ctx"]["delta_A_vs_B"].items()]) if "delta_A_vs_B" in long["wd_by_ctx"] else {}

  # Alternative capture mode: legacy gqa route path for same current context set.
  alt_env = {**baseline_env, "DECODE_ATTN_AMDGCN_TILE": "0"}
  alt = run_slope_variant(",".join(str(c) for c in current_contexts), alt_env, out_dir / "throughput/alternative_route", "ctx_slope_alternative_route")
  alt_summary = _slope_summary(alt, out_dir / "throughput/alternative_route" / "summary.json")
  alt_delta = _slope_delta_summary([{"ctx": ck, **vals} for ck, vals in alt["wd_by_ctx"]["delta_A_vs_B"].items()]) if "delta_A_vs_B" in alt["wd_by_ctx"] else {}

  runs["gate_post"] = run_gate(baseline_env, correctness_dir, "gate_post")
  lockstep_post = run_unknown_lockstep(",".join(str(c) for c in ctx_sweep_all), baseline_env, correctness_dir, "unknown_lockstep_post")

  bundle = {
    "authority": authority,
    "correctness": {
      "gate_pre": runs["gate_pre"],
      "gate_post": runs["gate_post"],
      "unknown_lockstep_pre": lockstep_pre,
      "unknown_lockstep_post": lockstep_post,
    },
    "sweeps": {
      "current": {
        "contexts": current_contexts,
        "command_output": current["command_output"],
        "wd_by_ctx": current["wd_by_ctx"],
        "sweep_summary": current_delta,
        "summary": current_summary,
      },
      "long": {
        "contexts": long_contexts,
        "command_output": long["command_output"],
        "wd_by_ctx": long["wd_by_ctx"],
        "sweep_summary": long_delta,
        "summary": long_summary,
      },
      "alternative_capture": {
        "contexts": current_contexts,
        "notes": "DECODE_ATTN_AMDGCN_TILE=0",
        "command_output": alt["command_output"],
        "wd_by_ctx": alt["wd_by_ctx"],
        "sweep_summary": alt_delta,
        "summary": alt_summary,
      },
    },
  }

  # Copy reference artifacts for quick handoff.
  _copy_json(LOCKSTEP_DIR / "decision.json", out_dir / "correctness/unknown_lockstep_decision_latest.json", required=True)
  _copy_json(LOCKSTEP_DIR / "latest.json", out_dir / "correctness/unknown_bucket_latest_current.json", required=True)
  _copy_json(RUNTIME_RESULT, out_dir / "runtime_latest_result.json", required=False)

  decision = build_decision(bundle)
  bundle["decision"] = decision

  out = out_dir / "bundle_snapshot.json"
  out.write_text(json.dumps(bundle, indent=2) + "\n")
  decision_path = out_dir / "decision.json"
  decision_path.write_text(json.dumps(decision, indent=2) + "\n")

  summary = [
    "# Decode Lifecycle Recheck Bundle",
    "",
    f"- run: `{run_id}`",
    f"- authority: `oracle gate PASS` pre={runs['gate_pre'].get('verdict')} post={runs['gate_post'].get('verdict')}",
    f"- unknown closure: pre={lockstep_pre.get('label')} post={lockstep_post.get('label')}",
    f"- current ctx result: `{decision['checks']['current_ctx_A_beats_B'].get('summary', {}).get('delta_pct_min')}` mean delta%",
    f"- decision: `{decision['verdict']}`",
    "",
    "## Commands executed",
    "",
    "1. `qk_decode_search_gate.py --oracle-tokens bench/qk-decode-search-readiness/baseline_oracle.json`",
    "2. `qk_decode_unknown_bucket_lockstep_audit.py --contexts " + ",".join(str(c) for c in ctx_sweep_all) + "`",
    "3. `qk_ctx_slope_driver.py` (current contexts)",
    "4. `qk_ctx_slope_driver.py` (long context)",
    "5. `qk_ctx_slope_driver.py` (`DECODE_ATTN_AMDGCN_TILE=0` alternative mode)",
    "6. `qk_decode_search_gate.py --oracle-tokens ...` (postflight)",
    "7. `qk_decode_unknown_bucket_lockstep_audit.py --contexts ...` (postflight)",
    "",
    "## Outputs",
    "",
    f"- `{out}`",
    f"- `{decision_path}`",
    f"- `{out_dir / 'correctness/unknown_lockstep_pre.json'}`",
    f"- `{out_dir / 'correctness/unknown_lockstep_post.json'}`",
    f"- `{out_dir / 'throughput/current_context/wd_by_ctx.json'}`",
    f"- `{out_dir / 'throughput/long_context/wd_by_ctx.json'}`",
    f"- `{out_dir / 'throughput/alternative_route/wd_by_ctx.json'}`",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary) + "\n")

  # Keep a direct pointer with top-level metadata.
  (OUT_BASE / "latest.json").write_text(json.dumps({"run": run_id, "bundle": str(out)}, indent=2) + "\n")
  print(f"DECODE_RECHECK_BUNDLE_DONE {out}")
  return 0


def main() -> int:
  p = argparse.ArgumentParser(description="Run full decode lifecycle recheck bundle with fixed authority + periodic protocol.")
  p.add_argument("--run-id", default=None, help="Bundle ID (YYYYMMDD-HHMMSS)")
  p.add_argument("--out-root", default=str(OUT_BASE), help="Output base directory")
  p.add_argument("--current-contexts", default="512,1024,2048,4096", help="Current-context point set")
  p.add_argument("--long-contexts", default="4096", help="Long-context points")
  p.add_argument("--lockstep-contexts", default="512,1024,2048,4096", help="Lockstep probe points")
  args = p.parse_args()
  try:
    return run(args)
  except Exception as e:
    raise SystemExit(f"decode recheck bundle failed: {e}")


if __name__ == "__main__":
  raise SystemExit(main())
