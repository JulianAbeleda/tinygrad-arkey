#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from collections import Counter
from typing import Any

from extra.qk_flywheel_dataset import LABELS, REASONS
from extra.qk_flywheel_triage_eval import (
  build_baseline_predictions,
  score_predictions,
)

REASON_ALIASES = {
  "full_decode_supported": "unsupported_runtime_scope",
  "full_decode_ready": "unsupported_runtime_scope",
  "static_gate_failed": "static_gate_fail",
  "microbench_fail": "microbench_regression",
  "compile_failed": "construction_blocked",
}

def _read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    if not raw.strip(): continue
    try:
      row = json.loads(raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    rows.append(row)
  return rows

def _jsonl(path:pathlib.Path, rows:list[dict[str, Any]]) -> None:
  with path.open("w") as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")

def _extract_first_json(text:str) -> tuple[dict[str, Any]|None, str|None, str|None]:
  start = text.find("{")
  if start < 0: return None, None, "no JSON object start"
  decoder = json.JSONDecoder()
  try:
    parsed, end = decoder.raw_decode(text[start:])
  except json.JSONDecodeError as exc:
    return None, None, exc.msg
  if not isinstance(parsed, dict): return None, None, "first JSON value is not an object"
  return parsed, text[start:start + end], None

def _prediction(row_id:str, method:str, label:str, reason:str, retry:bool, *, score:float=0.0,
                parse_ok:bool=False, schema_ok:bool=False, taxonomy_ok:bool=False,
                extracted:dict[str, Any]|None=None, error:str|None=None) -> dict[str, Any]:
  return {
    "id": row_id,
    "baseline": method,
    "label": label,
    "reason": reason,
    "retry": retry,
    "confidence": 0.5,
    "score": score,
    "parse_ok": parse_ok,
    "schema_ok": schema_ok,
    "taxonomy_ok": taxonomy_ok,
    "parsed": extracted,
    "error": error,
  }

def _label_score(label:str) -> float:
  return {"accept": 1.0, "raw_accept_unconfirmed": 0.75, "needs_rerun": 0.55, "tie": 0.35, "diagnostic_only": 0.25}.get(label, 0.0)

def _from_parsed(row_id:str, method:str, parsed:dict[str, Any]|None, error:str|None, *, repair:bool) -> dict[str, Any]:
  if parsed is None:
    return _prediction(row_id, method, "invalid_output", "invalid_output", False, error=error)
  label, reason, retry = parsed.get("label"), parsed.get("reason"), parsed.get("retry")
  schema_ok = isinstance(label, str) and isinstance(reason, str) and isinstance(retry, bool) and set(parsed) >= {"label", "reason", "retry"}
  if not schema_ok:
    return _prediction(row_id, method, "invalid_output", "invalid_output", False, parse_ok=True, extracted=parsed, error="schema")
  if repair and reason not in REASONS and reason in REASON_ALIASES:
    reason = REASON_ALIASES[reason]
  taxonomy_ok = label in LABELS and reason in REASONS
  return _prediction(
    row_id, method, label, reason, bool(retry), score=_label_score(label),
    parse_ok=True, schema_ok=True, taxonomy_ok=taxonomy_ok, extracted=parsed,
    error=None if taxonomy_ok else "taxonomy",
  )

def diagnostic_predictions(rollout_rows:list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  out = {"strict_text": [], "json_extract": [], "json_extract_taxonomy_repair": []}
  for row in rollout_rows:
    row_id = row["id"]
    text = str(row.get("text", ""))
    strict_parsed, _, strict_error = _extract_first_json(text.strip()) if text.strip().startswith("{") else (None, None, "extra text")
    out["strict_text"].append(_from_parsed(row_id, "strict_text", strict_parsed, strict_error, repair=False))
    parsed, _, error = _extract_first_json(text)
    out["json_extract"].append(_from_parsed(row_id, "json_extract", parsed, error, repair=False))
    out["json_extract_taxonomy_repair"].append(_from_parsed(row_id, "json_extract_taxonomy_repair", parsed, error, repair=True))
  return out

def _axis_counts(preds:list[dict[str, Any]]) -> dict[str, Any]:
  total = len(preds)
  return {
    "rows": total,
    "parse_ok": sum(pred.get("parse_ok") is True for pred in preds),
    "schema_ok": sum(pred.get("schema_ok") is True for pred in preds),
    "taxonomy_ok": sum(pred.get("taxonomy_ok") is True for pred in preds),
    "labels": dict(sorted(Counter(pred["label"] for pred in preds).items())),
    "reasons": dict(sorted(Counter(pred["reason"] for pred in preds).items())),
    "errors": dict(sorted(Counter(str(pred.get("error")) for pred in preds if pred.get("error")).items())),
  }

def run_diagnostic(examples_path:pathlib.Path, rollout_path:pathlib.Path, out:pathlib.Path) -> dict[str, Any]:
  examples = _read_jsonl(examples_path)
  rollouts = _read_jsonl(rollout_path / "rollouts.jsonl" if rollout_path.is_dir() else rollout_path)
  preds_by_method = diagnostic_predictions(rollouts)
  baseline_preds = build_baseline_predictions(examples)
  baseline_scores = {name: score_predictions(examples, preds) for name, preds in baseline_preds.items()}
  method_scores = {name: score_predictions(examples, preds) for name, preds in preds_by_method.items()}
  mechanism_prior = baseline_scores["mechanism_prior"]
  best_diag_name, best_diag = sorted(method_scores.items(), key=lambda item: (-item[1]["macro_f1"], -item[1]["accuracy"], item[0]))[0]
  if best_diag["macro_f1"] <= mechanism_prior["macro_f1"]:
    conclusion = "protocol_fix_not_enough"
  elif method_scores["strict_text"]["macro_f1"] == 0.0:
    conclusion = "protocol_blocking_possible_signal"
  else:
    conclusion = "diagnostic_signal"
  summary = {
    "kind": "qk_flywheel_protocol_diagnostic",
    "examples": str(examples_path),
    "rollout": str(rollout_path),
    "rows": len(rollouts),
    "baseline": {
      "mechanism_prior_macro_f1": mechanism_prior["macro_f1"],
      "mechanism_prior_accuracy": mechanism_prior["accuracy"],
      "mechanism_prior_false_positive_accept_rate": mechanism_prior["false_positive_accept_rate"],
    },
    "methods": method_scores,
    "axes": {name: _axis_counts(preds) for name, preds in preds_by_method.items()},
    "best_diagnostic": best_diag_name,
    "conclusion": conclusion,
    "note": "Diagnostic only. Extracting or repairing output is not flywheel proof and does not replace the official strict score.",
  }
  out.mkdir(parents=True, exist_ok=True)
  _jsonl(out / "predictions.jsonl", [pred for name in sorted(preds_by_method) for pred in preds_by_method[name]])
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_readme(summary))
  return summary

def _fmt(value:Any) -> str:
  return "n/a" if value is None else f"{value:.3f}"

def _readme(summary:dict[str, Any]) -> str:
  lines = [
    "# QK Flywheel Protocol Diagnostic",
    "",
    "This is a Phase 3.0 diagnostic over the existing no-adapter rollout. It",
    "separates strict-output failure from label/reason triage quality. It is not",
    "a promotion artifact and does not replace the strict Phase 2 score.",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- rows: `{summary['rows']}`",
    f"- baseline macro-F1: `{summary['baseline']['mechanism_prior_macro_f1']:.3f}`",
    f"- best diagnostic: `{summary['best_diagnostic']}`",
    "",
    "| method | parse | schema | taxonomy | accuracy | macro-F1 | false accept | p@3 | ndcg |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for name, row in summary["methods"].items():
    axes = summary["axes"][name]
    ranking = row["ranking"]
    lines.append(
      f"| `{name}` | {axes['parse_ok']}/{axes['rows']} | {axes['schema_ok']}/{axes['rows']} | {axes['taxonomy_ok']}/{axes['rows']} | "
      f"{_fmt(row['accuracy'])} | {_fmt(row['macro_f1'])} | {_fmt(row['false_positive_accept_rate'])} | "
      f"{_fmt(ranking['precision_at_3'])} | {_fmt(ranking['ndcg'])} |"
    )
  lines.append("")
  return "\n".join(lines)

def main() -> int:
  parser = argparse.ArgumentParser(description="Diagnose strict-output versus triage signal for QK flywheel rollouts")
  parser.add_argument("--examples", type=pathlib.Path, required=True)
  parser.add_argument("--rollout", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()
  summary = run_diagnostic(args.examples, args.rollout, args.out)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
