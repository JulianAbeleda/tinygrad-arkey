#!/usr/bin/env python3
"""Final S9 report for LDS2 prefill decomposition artifacts."""
from __future__ import annotations

import argparse, json, math, pathlib, statistics
from typing import Any

DEFAULT_S9_DIR = pathlib.Path("bench/prefill-lds2-s9")
DEFAULT_WHOLE_DIR = pathlib.Path("bench/prefill-whole-synced")
DEFAULT_OUTPUT = DEFAULT_S9_DIR / "final-report.json"
AXIS_PATTERNS = {
  "wait": "wait-search*.json",
  "layout": "layout-search.json",
  "lifecycle": "lifecycle-search.json",
  "memory": "memory-search.json",
  "combined": "combined-search.json",
}
REQUIRED_AXES = ("wait", "layout", "lifecycle")


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


def _is_correct(row: dict[str, Any]) -> bool:
  status = str(row.get("status", "")).lower()
  if status != "ok": return False
  rr = _finite_float(row.get("rel_rmse"))
  return rr is None or rr <= 2e-2


def _candidate_id(row: dict[str, Any]) -> Any:
  for key in ("candidate_id", "name", "candidate", "id"):
    if key in row: return row[key]
  return None


def _candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
  keep = ("candidate_id", "name", "status", "tflops", "rel_rmse", "wait_policy", "layout", "memory_layout",
          "lifecycle_template", "reason", "message")
  out = {k: _json_safe(row[k]) for k in keep if k in row}
  if not out: out["candidate_id"] = _candidate_id(row)
  return out


def _json_safe(value: Any) -> Any:
  if isinstance(value, float): return value if math.isfinite(value) else None
  if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, list): return [_json_safe(v) for v in value]
  if isinstance(value, tuple): return [_json_safe(v) for v in value]
  return value


def _best_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
  correct = [r for r in rows if _is_correct(r) and _finite_float(r.get("tflops")) is not None]
  if not correct: return None
  return _candidate_summary(max(correct, key=lambda r: _finite_float(r.get("tflops")) or float("-inf")))


def _baseline_from_artifact(data: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
  baseline_id = data.get("baseline_candidate_id")
  if baseline_id is not None:
    for row in rows:
      if row.get("candidate_id") == baseline_id: return _candidate_summary(row)
  for row in rows:
    name = str(row.get("name", "")).lower()
    if name == "baseline": return _candidate_summary(row)
  return None


def _speedup(best: float | None, baseline: float | None) -> float | None:
  return best / baseline - 1.0 if best is not None and baseline is not None and baseline > 0 else None


def _axis_summary(axis: str, paths: list[pathlib.Path], material_threshold: float) -> dict[str, Any]:
  if not paths:
    return {"status": "missing", "artifact_paths": [], "errors": [], "best_correct_candidate": None,
            "materiality_vs_baseline": None, "rejected_or_wrong_candidates": []}

  errors: list[dict[str, str]] = []
  rejected: list[dict[str, Any]] = []
  best_samples: list[float] = []
  baseline_samples: list[float] = []
  best_candidate: dict[str, Any] | None = None
  artifact_verdicts: list[str] = []

  for path in paths:
    data, err = _load_json(path)
    if err:
      errors.append({"path": str(path), "error": err})
      continue
    artifact_verdicts.append(str(data.get("verdict", "")))
    rows = [r for r in data.get("rows", []) if isinstance(r, dict)] if isinstance(data.get("rows", []), list) else []
    for row in rows:
      if not _is_correct(row):
        rejected.append({"artifact": str(path), **_candidate_summary(row)})
    artifact_best = _best_from_rows(rows)
    if artifact_best is not None:
      artifact_best["artifact"] = str(path)
      tflops = _finite_float(artifact_best.get("tflops"))
      if tflops is not None:
        best_samples.append(tflops)
        if best_candidate is None or tflops > (_finite_float(best_candidate.get("tflops")) or float("-inf")):
          best_candidate = artifact_best
    artifact_baseline = _baseline_from_artifact(data, rows)
    baseline_tflops = _finite_float(artifact_baseline.get("tflops")) if artifact_baseline else _finite_float(data.get("baseline_tflops"))
    if baseline_tflops is not None: baseline_samples.append(baseline_tflops)

    if not rows:
      direct_best = _finite_float(data.get("best_tflops"))
      direct_base = _finite_float(data.get("baseline_tflops"))
      if direct_best is not None: best_samples.append(direct_best)
      if direct_base is not None: baseline_samples.append(direct_base)

  best_median = _median(best_samples)
  baseline_median = _median(baseline_samples)
  spd = _speedup(best_median, baseline_median)
  material = None if spd is None else abs(spd) >= material_threshold
  if errors and not best_samples:
    status = "invalid"
  elif best_median is None:
    status = "no_correct_candidate"
  elif material:
    status = "material_win" if (spd or 0.0) > 0 else "material_loss"
  else:
    status = "no_material_change"

  return {
    "status": status,
    "artifact_paths": [str(p) for p in paths],
    "artifact_verdicts": artifact_verdicts,
    "errors": errors,
    "baseline_tflops_median": baseline_median,
    "best_tflops_median": best_median,
    "best_correct_candidate": best_candidate,
    "materiality_vs_baseline": {
      "speedup": spd,
      "threshold": material_threshold,
      "is_material": material,
    },
    "rejected_or_wrong_candidates": rejected,
  }


def _authority_summary(paths: list[pathlib.Path], material_threshold: float) -> dict[str, Any]:
  samples_512: list[float] = []
  samples_4096: list[float] = []
  baseline_512: list[float] = []
  baseline_4096: list[float] = []
  errors: list[dict[str, str]] = []
  candidates: list[dict[str, Any]] = []
  for path in paths:
    data, err = _load_json(path)
    if err:
      errors.append({"path": str(path), "error": err})
      continue
    whole = data.get("whole_tok_s", {})
    if not isinstance(whole, dict): whole = {}
    v512, v4096 = _finite_float(whole.get("512")), _finite_float(whole.get("4096"))
    if v512 is not None: samples_512.append(v512)
    if v4096 is not None: samples_4096.append(v4096)
    lower_name = path.name.lower()
    if "default" in lower_name or "baseline" in lower_name:
      if v512 is not None: baseline_512.append(v512)
      if v4096 is not None: baseline_4096.append(v4096)
    gate = data.get("prefill_route_binding_gate", {})
    gate_verdict = gate.get("verdict") if isinstance(gate, dict) else None
    candidates.append({"artifact": str(path), "pp512": v512, "pp4096": v4096, "gate_verdict": gate_verdict})

  pp512, pp4096 = _median(samples_512), _median(samples_4096)
  b512, b4096 = _median(baseline_512), _median(baseline_4096)
  best512 = max(samples_512) if samples_512 else None
  best4096 = max(samples_4096) if samples_4096 else None
  best_candidate_512 = max((c for c in candidates if c.get("pp512") is not None), key=lambda c: c["pp512"], default=None)
  best_candidate_4096 = max((c for c in candidates if c.get("pp4096") is not None), key=lambda c: c["pp4096"], default=None)
  spd512, spd4096 = _speedup(best512, b512), _speedup(best4096, b4096)
  has_clean_gate = any(c.get("gate_verdict") in (None, "PREFILL_ROUTE_BINDING_PASS") for c in candidates)
  return {
    "status": "missing" if not paths else ("valid" if pp512 is not None or pp4096 is not None else "invalid"),
    "artifact_paths": [str(p) for p in paths],
    "errors": errors,
    "pp512_median": pp512,
    "pp4096_median": pp4096,
    "baseline_pp512_median": b512,
    "baseline_pp4096_median": b4096,
    "best_pp512": best512,
    "best_pp4096": best4096,
    "best_candidate_pp512": best_candidate_512,
    "best_candidate_pp4096": best_candidate_4096,
    "materiality_vs_baseline": {
      "pp512_speedup": spd512,
      "pp4096_speedup": spd4096,
      "threshold": material_threshold,
      "is_material": any(s is not None and s >= material_threshold for s in (spd512, spd4096)),
    },
    "has_clean_binding_gate": has_clean_gate,
    "candidates": candidates,
  }


def _glob_existing(directory: pathlib.Path, pattern: str) -> list[pathlib.Path]:
  return sorted(directory.glob(pattern)) if directory.exists() else []


def build_report(s9_dir: pathlib.Path = DEFAULT_S9_DIR, whole_dir: pathlib.Path = DEFAULT_WHOLE_DIR,
                 material_threshold: float = 0.03, required_axes: tuple[str, ...] = REQUIRED_AXES) -> dict[str, Any]:
  axes = {axis: _axis_summary(axis, _glob_existing(s9_dir, pattern), material_threshold)
          for axis, pattern in AXIS_PATTERNS.items()}
  authority_paths = sorted(set(_glob_existing(whole_dir, "*s9*.json") + _glob_existing(s9_dir, "*authority*.json")))
  authority = _authority_summary(authority_paths, material_threshold)

  missing_required = [axis for axis in required_axes if axes[axis]["status"] == "missing"]
  invalid_required = [axis for axis in required_axes if axes[axis]["status"] in ("invalid", "no_correct_candidate")]
  combined = axes["combined"]
  combined_material = bool((combined.get("materiality_vs_baseline") or {}).get("is_material") and
                           ((combined.get("materiality_vs_baseline") or {}).get("speedup") or 0.0) > 0)
  authority_material = bool((authority.get("materiality_vs_baseline") or {}).get("is_material"))
  any_axis_material = any(a["status"] == "material_win" for name, a in axes.items() if name != "combined")
  any_correct = any(a["best_correct_candidate"] is not None for a in axes.values())

  reasons: list[str] = []
  if missing_required:
    verdict = "blocked"
    reasons.append("missing required artifacts: " + ", ".join(missing_required))
  elif invalid_required:
    verdict = "reject"
    reasons.append("required artifacts have no valid correct candidate: " + ", ".join(invalid_required))
  elif authority_material:
    verdict = "promote_default"
  elif combined_material and authority["status"] == "missing":
    verdict = "blocked"
    reasons.append("combined micro win needs whole-prefill authority before default promotion")
  elif combined_material:
    verdict = "keep_opt_in"
    reasons.append("combined micro win did not clear whole-prefill authority threshold")
  elif combined["status"] == "missing" and authority["status"] == "missing":
    verdict = "blocked"
    reasons.append("missing combined-search or whole-prefill authority artifact for promotion decision")
  elif any_axis_material:
    verdict = "keep_opt_in"
    reasons.append("only axis-local material wins are present")
  elif any_correct:
    verdict = "reject"
    reasons.append("no correct candidate beats/default equals baseline at material threshold")
  else:
    verdict = "reject"
    reasons.append("no correct S9 candidate found")

  return {
    "schema": "prefill-lds2-s9-final-report.v1",
    "verdict": verdict,
    "reasons": reasons,
    "thresholds": {"material_speedup": material_threshold},
    "axis_status": {axis: summary["status"] for axis, summary in axes.items()},
    "axes": axes,
    "whole_prefill_authority": authority,
    "inputs": {
      "s9_dir": str(s9_dir),
      "whole_dir": str(whole_dir),
      "required_axes": list(required_axes),
    },
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--s9-dir", type=pathlib.Path, default=DEFAULT_S9_DIR)
  ap.add_argument("--whole-dir", type=pathlib.Path, default=DEFAULT_WHOLE_DIR)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--material-threshold", type=float, default=0.03)
  ap.add_argument("--json", action="store_true", help="Print the full report JSON to stdout.")
  args = ap.parse_args(argv)

  report = build_report(args.s9_dir, args.whole_dir, args.material_threshold)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  if args.json:
    print(json.dumps(report, indent=2, allow_nan=False))
  else:
    print(f"{report['verdict']} axes={report['axis_status']} output={args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
