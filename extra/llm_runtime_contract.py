#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.llm_eval_common import md_text

SUMMARY_KINDS = {
  "eval": ("llm_eval_summary", "qwen_eval_summary"),
  "rollout": ("llm_rollout_summary",),
  "compare": ("llm_rollout_compare_report",),
  "training_data": ("llm_training_data_probe",),
  "training_run": ("llm_sft_smoke_train_summary",),
}

def _load_json(path:pathlib.Path) -> Any:
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"{path}: invalid JSON: {exc}") from exc

def _repo_path(repo:pathlib.Path, value:str | pathlib.Path) -> pathlib.Path:
  path = pathlib.Path(value).expanduser()
  return path if path.is_absolute() else repo / path

def _is_repo_relative(value:str) -> bool:
  path = pathlib.PurePath(value)
  return not value.startswith("~") and not path.is_absolute()

def load_manifest(path:pathlib.Path) -> dict[str, Any]:
  manifest = _load_json(path)
  if manifest.get("kind") != "llm_runtime_contract_manifest":
    raise ValueError(f"{path}: expected kind=llm_runtime_contract_manifest")
  if not isinstance(manifest.get("rows"), list) or not manifest["rows"]:
    raise ValueError(f"{path}: expected non-empty rows")
  seen: set[str] = set()
  for idx, row in enumerate(manifest["rows"]):
    if not isinstance(row, dict): raise ValueError(f"{path}: row {idx} must be an object")
    for key in ("id", "type", "artifact"):
      if not isinstance(row.get(key), str) or not row[key]: raise ValueError(f"{path}: row {idx} missing string {key}")
    if row["type"] not in SUMMARY_KINDS: raise ValueError(f"{path}: row {idx} has unknown type {row['type']!r}")
    if row["id"] in seen: raise ValueError(f"{path}: duplicate row id {row['id']!r}")
    seen.add(row["id"])
    if not _is_repo_relative(row["artifact"]): raise ValueError(f"{path}: row {idx} artifact must be repo-relative")
    if "policy" in row and row["policy"] is not None and not _is_repo_relative(row["policy"]):
      raise ValueError(f"{path}: row {idx} policy must be repo-relative")
    if "dataset" in row and row["dataset"] is not None and not _is_repo_relative(row["dataset"]):
      raise ValueError(f"{path}: row {idx} dataset must be repo-relative")
  return manifest

def _summary_path(repo:pathlib.Path, row:dict[str, Any]) -> pathlib.Path:
  artifact = _repo_path(repo, row["artifact"])
  if row["type"] in ("eval", "rollout"): return artifact / "summary.json"
  if row["type"] == "compare": return artifact / "report.json"
  if row["type"] in ("training_data", "training_run"): return artifact / "summary.json"
  raise ValueError(f"unknown row type {row['type']!r}")

def _check_eval(row:dict[str, Any], summary:dict[str, Any]) -> list[str]:
  errors = []
  if summary.get("status") != "pass": errors.append(f"status={summary.get('status')!r}")
  if summary.get("tokens_match") is not True: errors.append("tokens_match is not true")
  if row.get("policy") and summary.get("policy") != row["policy"]: errors.append("policy mismatch")
  if row.get("storage") and summary.get("storage") != row["storage"]: errors.append("storage mismatch")
  if row.get("prompt_format") and summary.get("prompt_format") != row["prompt_format"]: errors.append("prompt_format mismatch")
  return errors

def _check_rollout(row:dict[str, Any], summary:dict[str, Any]) -> list[str]:
  errors = []
  quality = summary.get("quality") or {}
  if quality.get("status") != "pass": errors.append(f"quality={quality.get('status')!r}")
  if row.get("mode") and summary.get("mode") != row["mode"]: errors.append("mode mismatch")
  if row.get("policy") and summary.get("policy") != row["policy"]: errors.append("policy mismatch")
  if row.get("storage") and summary.get("storage") != row["storage"]: errors.append("storage mismatch")
  if row.get("dataset") and summary.get("dataset") != row["dataset"]: errors.append("dataset mismatch")
  return errors

def _check_compare(row:dict[str, Any], report:dict[str, Any]) -> list[str]:
  errors = []
  for comp in report.get("comparisons", []):
    q, out = comp.get("quality", {}), comp.get("outputs", {})
    if q.get("regressions"): errors.append(f"{comp.get('candidate')}: quality regressions={q.get('regressions')[:5]}")
    if out.get("tokens_changed", 0) != 0: errors.append(f"{comp.get('candidate')}: tokens_changed={out.get('tokens_changed')}")
    if row.get("require_text_equal", True) and out.get("text_changed", 0) != 0:
      errors.append(f"{comp.get('candidate')}: text_changed={out.get('text_changed')}")
  return errors

def _check_training(row:dict[str, Any], summary:dict[str, Any]) -> list[str]:
  errors = []
  if summary.get("exported_rows", 0) < row.get("min_rows", 1): errors.append("not enough exported rows")
  if summary.get("filtered_rows", 0) != 0 and row.get("allow_filtered", False) is not True:
    errors.append(f"filtered_rows={summary.get('filtered_rows')}")
  return errors

def _check_training_run(row:dict[str, Any], summary:dict[str, Any]) -> list[str]:
  errors = []
  if summary.get("status") != "pass": errors.append(f"status={summary.get('status')!r}")
  final_eval = (summary.get("final") or {}).get("eval") or {}
  deltas = summary.get("deltas") or {}
  if final_eval.get("accuracy", 0.0) < row.get("min_eval_accuracy", 0.0):
    errors.append(f"eval accuracy {final_eval.get('accuracy')} < {row.get('min_eval_accuracy')}")
  if deltas.get("eval_loss", 0.0) < row.get("min_eval_loss_delta", 0.0):
    errors.append(f"eval loss delta {deltas.get('eval_loss')} < {row.get('min_eval_loss_delta')}")
  return errors

def validate_contract(manifest:dict[str, Any], repo:pathlib.Path, *, require_models:bool=False) -> dict[str, Any]:
  rows = []
  for row in manifest["rows"]:
    errors = []
    if "model" in row and require_models and not pathlib.Path(row["model"]).expanduser().exists():
      errors.append(f"model missing: {row['model']}")
    for field in ("policy", "dataset"):
      if row.get(field) and not _repo_path(repo, row[field]).exists(): errors.append(f"{field} missing: {row[field]}")
    summary_path = _summary_path(repo, row)
    if not summary_path.exists():
      rows.append({**row, "summary_path": str(summary_path.relative_to(repo) if summary_path.is_relative_to(repo) else summary_path),
                   "status": "missing", "errors": [f"summary missing: {summary_path}"] + errors})
      continue
    summary = _load_json(summary_path)
    expected_kinds = SUMMARY_KINDS[row["type"]]
    if summary.get("kind") not in expected_kinds: errors.append(f"expected kind in {expected_kinds}, got {summary.get('kind')!r}")
    if row["type"] == "eval": errors += _check_eval(row, summary)
    elif row["type"] == "rollout": errors += _check_rollout(row, summary)
    elif row["type"] == "compare": errors += _check_compare(row, summary)
    elif row["type"] == "training_data": errors += _check_training(row, summary)
    elif row["type"] == "training_run":
      errors += _check_training_run(row, summary)
      weights = (summary.get("artifacts") or {}).get("weights")
      if row.get("require_weights", True):
        if not isinstance(weights, str) or not weights: errors.append("missing weights artifact name")
        elif not (summary_path.parent / weights).exists(): errors.append(f"weights missing: {weights}")
    rows.append({**row, "summary_path": str(summary_path.relative_to(repo) if summary_path.is_relative_to(repo) else summary_path),
                 "status": "pass" if not errors else "fail", "errors": errors})
  return {
    "kind": "llm_runtime_contract_report",
    "source_manifest": manifest.get("name"),
    "rows": rows,
    "summary": {
      "rows": len(rows),
      "passed": sum(1 for row in rows if row["status"] == "pass"),
      "failed": sum(1 for row in rows if row["status"] == "fail"),
      "missing": sum(1 for row in rows if row["status"] == "missing"),
    },
  }

def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# LLM Runtime Contract",
    "",
    "This report validates the artifact contract for generated-policy rollout/eval",
    "work. Model files may remain local, but committed datasets, policies, and",
    "evidence artifacts must be present and internally consistent.",
    "",
    "| id | type | status | artifact | errors |",
    "|---|---|---:|---|---|",
  ]
  for row in report["rows"]:
    errors = "; ".join(row.get("errors") or [])
    lines.append(f"| `{row['id']}` | `{row['type']}` | `{row['status']}` | `{row['artifact']}` | {md_text(errors)} |")
  lines += ["", "## Summary", "", "```json", json.dumps(report["summary"], indent=2, sort_keys=True), "```", ""]
  return "\n".join(lines)

def main() -> int:
  parser = argparse.ArgumentParser(description="Validate committed LLM rollout/eval runtime contract artifacts")
  parser.add_argument("--manifest", type=pathlib.Path, required=True)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--require-models", action="store_true")
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()

  repo = args.repo.resolve()
  manifest = load_manifest(_repo_path(repo, args.manifest))
  report = validate_contract(manifest, repo, require_models=args.require_models)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(report_markdown(report))
  if not args.json and not args.md: print(report_markdown(report))
  if report["summary"]["failed"] or report["summary"]["missing"]: return 1
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
