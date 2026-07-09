#!/usr/bin/env python3
"""Aggregate S9 wait-search and whole-prefill artifacts into a promotion report."""
from __future__ import annotations

import argparse, json, math, pathlib, statistics
from typing import Any

DEFAULT_S9_DIR = pathlib.Path("bench/prefill-lds2-s9")
DEFAULT_WHOLE_DIR = pathlib.Path("bench/prefill-whole-synced")
DEFAULT_OUTPUT = DEFAULT_S9_DIR / "report.json"
ZERO_WAIT_POLICY = {"vm_after_coop_load": 0, "lgkm_after_coop_store": 0, "lgkm_after_frag_load": 0}


def _load_json(path: pathlib.Path) -> tuple[dict[str, Any] | None, str | None]:
  try:
    data = json.loads(path.read_text())
  except FileNotFoundError:
    return None, "missing"
  except Exception as e:
    return None, f"invalid_json: {e}"
  return (data, None) if isinstance(data, dict) else (None, "json_root_not_object")


def _finite_float(value: Any) -> float | None:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return None
  return out if math.isfinite(out) else None


def _median(values: list[float]) -> float | None:
  return statistics.median(values) if values else None


def _is_ok(row: dict[str, Any]) -> bool:
  return str(row.get("status", "")).lower() == "ok"


def _candidate_key(row: dict[str, Any]) -> str:
  if "candidate_id" in row: return f"id:{row['candidate_id']}"
  policy = row.get("wait_policy")
  if isinstance(policy, dict): return "wait:" + json.dumps(policy, sort_keys=True, separators=(",", ":"))
  return "unknown"


def _summarize_wait_artifacts(paths: list[pathlib.Path]) -> dict[str, Any]:
  errors: list[dict[str, str]] = []
  by_candidate: dict[str, dict[str, Any]] = {}
  for path in paths:
    data, err = _load_json(path)
    if err:
      errors.append({"path": str(path), "error": err})
      continue
    rows = data.get("rows", [])
    if not isinstance(rows, list):
      errors.append({"path": str(path), "error": "rows_not_list"})
      continue
    for row in rows:
      if not isinstance(row, dict): continue
      key = _candidate_key(row)
      item = by_candidate.setdefault(key, {
        "candidate_key": key,
        "candidate_id": row.get("candidate_id"),
        "wait_policy": row.get("wait_policy"),
        "ok_tflops": [],
        "bad_statuses": [],
        "artifact_paths": [],
      })
      item["artifact_paths"].append(str(path))
      tflops = _finite_float(row.get("tflops"))
      if _is_ok(row) and tflops is not None:
        item["ok_tflops"].append(tflops)
      elif not _is_ok(row):
        item["bad_statuses"].append(str(row.get("status", "missing_status")))

  candidates = []
  for item in by_candidate.values():
    samples = item.pop("ok_tflops")
    item["ok_count"] = len(samples)
    item["bad_count"] = len(item["bad_statuses"])
    item["micro_tflops_median"] = _median(samples)
    item["micro_tflops_samples"] = samples
    candidates.append(item)
  candidates.sort(key=lambda c: (c["micro_tflops_median"] is not None, c["micro_tflops_median"] or 0.0), reverse=True)

  baseline = next((c for c in candidates if c.get("wait_policy") == ZERO_WAIT_POLICY), None)
  best = candidates[0] if candidates and candidates[0]["micro_tflops_median"] is not None else None
  return {
    "artifact_count": len(paths),
    "errors": errors,
    "candidates": candidates,
    "baseline": baseline,
    "best": best,
  }


def _summarize_whole_artifacts(paths: list[pathlib.Path]) -> dict[str, Any]:
  errors: list[dict[str, str]] = []
  pp512: list[float] = []
  pp4096: list[float] = []
  binding_failures: list[dict[str, Any]] = []
  used: list[str] = []
  for path in paths:
    data, err = _load_json(path)
    if err:
      errors.append({"path": str(path), "error": err})
      continue
    whole_tok_s = data.get("whole_tok_s", {})
    if not isinstance(whole_tok_s, dict): continue
    v512 = _finite_float(whole_tok_s.get("512"))
    v4096 = _finite_float(whole_tok_s.get("4096"))
    if v512 is not None: pp512.append(v512)
    if v4096 is not None: pp4096.append(v4096)
    if v512 is not None or v4096 is not None: used.append(str(path))
    gate = data.get("prefill_route_binding_gate")
    if isinstance(gate, dict) and str(gate.get("verdict", "")).endswith("_FAIL"):
      binding_failures.append({"path": str(path), "verdict": gate.get("verdict"), "failures": gate.get("failures", [])})
  return {
    "artifact_count": len(paths),
    "used_paths": used,
    "errors": errors,
    "pp512_median": _median(pp512),
    "pp4096_median": _median(pp4096),
    "pp512_samples": pp512,
    "pp4096_samples": pp4096,
    "binding_failures": binding_failures,
  }


def _glob_existing(directory: pathlib.Path, pattern: str) -> list[pathlib.Path]:
  return sorted(directory.glob(pattern)) if directory.exists() else []


def build_report(s9_dir: pathlib.Path = DEFAULT_S9_DIR, whole_dir: pathlib.Path = DEFAULT_WHOLE_DIR,
                 min_micro_speedup: float = 0.01) -> dict[str, Any]:
  wait_paths = _glob_existing(s9_dir, "wait-search*.json")
  whole_paths = _glob_existing(whole_dir, "*s9*.json")
  wait = _summarize_wait_artifacts(wait_paths)
  whole = _summarize_whole_artifacts(whole_paths)
  failures: list[str] = []

  best = wait["best"]
  baseline = wait["baseline"]
  speedup = None
  if best and baseline and baseline.get("micro_tflops_median"):
    speedup = best["micro_tflops_median"] / baseline["micro_tflops_median"] - 1.0

  if not wait_paths:
    verdict = "inconclusive"
    failures.append("no S9 wait-search artifacts found")
  elif best is None:
    verdict = "reject_correctness"
    failures.append("no correct S9 wait-search candidate with finite TFLOPS")
  elif best.get("bad_count", 0):
    verdict = "reject_correctness"
    failures.append("best candidate has non-ok correctness/status samples")
  elif speedup is None:
    verdict = "inconclusive"
    failures.append("missing baseline candidate for performance comparison")
  elif speedup < min_micro_speedup:
    verdict = "reject_perf"
    failures.append(f"best micro TFLOPS speedup {speedup:.4f} < required {min_micro_speedup:.4f}")
  elif whole["pp512_median"] is None or whole["pp4096_median"] is None:
    verdict = "inconclusive"
    failures.append("missing whole-prefill pp512/pp4096 artifacts")
  elif whole["binding_failures"]:
    verdict = "keep_opt_in"
    failures.append("whole-prefill route binding gate is not promotion-clean")
  else:
    verdict = "promote"

  return {
    "schema": "prefill-lds2-s9-report.v1",
    "verdict": verdict,
    "failures": failures,
    "thresholds": {"min_micro_speedup": min_micro_speedup},
    "micro_tflops": {
      "baseline_median": baseline.get("micro_tflops_median") if baseline else None,
      "best_median": best.get("micro_tflops_median") if best else None,
      "best_speedup": speedup,
      "best_candidate_key": best.get("candidate_key") if best else None,
      "best_candidate_id": best.get("candidate_id") if best else None,
      "best_wait_policy": best.get("wait_policy") if best else None,
    },
    "whole_prefill": {
      "pp512_median": whole["pp512_median"],
      "pp4096_median": whole["pp4096_median"],
      "binding_failure_count": len(whole["binding_failures"]),
    },
    "inputs": {
      "s9_dir": str(s9_dir),
      "whole_dir": str(whole_dir),
      "wait_artifacts": [str(p) for p in wait_paths],
      "whole_artifacts": [str(p) for p in whole_paths],
    },
    "wait_search": wait,
    "whole_prefill_detail": whole,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--s9-dir", type=pathlib.Path, default=DEFAULT_S9_DIR)
  ap.add_argument("--whole-dir", type=pathlib.Path, default=DEFAULT_WHOLE_DIR)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--min-micro-speedup", type=float, default=0.01)
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args()

  report = build_report(args.s9_dir, args.whole_dir, args.min_micro_speedup)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  if args.json:
    print(json.dumps(report, indent=2, allow_nan=False))
  else:
    mt = report["micro_tflops"]
    wp = report["whole_prefill"]
    print(f"{report['verdict']} micro_best={mt['best_median']} pp512={wp['pp512_median']} pp4096={wp['pp4096_median']} output={args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
