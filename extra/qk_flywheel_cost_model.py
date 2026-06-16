#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.qk_flywheel_triage_eval import build_baseline_predictions, score_predictions

# Cost-model feature extraction and prediction backends live in sibling modules.
# Re-exported here so existing import sites (shadow, feature_audit, tests) keep
# importing from extra.qk_flywheel_cost_model unchanged.
from extra.qk_flywheel_cost_model_features import (  # noqa: F401
  FORBIDDEN_FEATURE_SOURCES, SAFE_CONTEXT_CATEGORICAL, SAFE_SCHEDULE_CATEGORICAL, SAFE_TOP_LEVEL_CATEGORICAL,
  FeatureVectorizer, extract_feature_map,
)
from extra.qk_flywheel_cost_model_score import (  # noqa: F401
  RANK_RELEVANCE, _backend_available, _centroid_predictions, _fit_predict_backend, _label_policy,
  _prediction, _requested_backends, _xgboost_predictions,
)

from extra.llm_eval_common import read_jsonl as _read_jsonl

from extra.llm_eval_common import write_jsonl as _jsonl

def _conclusion(metrics:dict[str, Any], baselines:dict[str, Any]) -> str:
  mechanism_prior = baselines.get("mechanism_prior", {"macro_f1": 0.0, "false_positive_accept_rate": 1.0, "ranking": {}})
  best = sorted(metrics.values(), key=lambda row: (-row["macro_f1"], -row["accuracy"]))[0] if metrics else None
  if best is None: return "no_model"
  ranking = best.get("ranking", {})
  prior_ranking = mechanism_prior.get("ranking", {})
  ranking_improved = (
    ranking.get("ndcg") is not None and prior_ranking.get("ndcg") is not None and ranking["ndcg"] >= prior_ranking["ndcg"] + 0.02
  ) or (
    ranking.get("precision_at_3") is not None and prior_ranking.get("precision_at_3") is not None and ranking["precision_at_3"] > prior_ranking["precision_at_3"]
  )
  if best["macro_f1"] > mechanism_prior["macro_f1"] and best["false_positive_accept_rate"] <= 0.05 and ranking_improved:
    return "cost_model_shadow_candidate"
  if ranking_improved: return "ranking_signal_only"
  return "no_signal"

def _markdown(summary:dict[str, Any]) -> str:
  def fmt(v:Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"
  lines = [
    "# AMD Decode Flywheel Cost Model",
    "",
    "This Phase 3B artifact tests the learned-cost-model version of kernel triage.",
    "It uses only pre-result candidate/context features and scores on the same",
    "family-split holdout as the Phase 2 baselines.",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- backend request: `{summary['backend_request']}`",
    f"- xgboost available: `{summary['backends']['xgboost_available']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- feature policy: `{summary['feature_policy']}`",
    f"- feature count: `{summary['features']['feature_count']}`",
    "",
    "## Backends",
    "",
  ]
  for detail in summary["backends"].get("ran", []):
    extra = []
    if "xgboost_version" in detail: extra.append(f"xgboost `{detail['xgboost_version']}`")
    if "rank_score_model" in detail: extra.append(f"rank score `{detail['rank_score_model']}`")
    suffix = "" if not extra else f" ({', '.join(extra)})"
    lines.append(f"- `{detail['backend']}`: `{detail['status']}`{suffix}")
  lines += [
    "",
    "## Metrics",
    "",
    "| model | accuracy | macro-F1 | false accept | p@3 | ndcg |",
    "|---|---:|---:|---:|---:|---:|",
  ]
  rows = {name: row for name, row in summary["baselines"].items() if name in ("mechanism_prior", "simple_family_heuristic", "reject_all")}
  rows.update(summary["models"])
  for name, row in rows.items():
    ranking = row["ranking"]
    lines.append(f"| `{name}` | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['false_positive_accept_rate'])} | {fmt(ranking['precision_at_3'])} | {fmt(ranking['ndcg'])} |")
  lines += [
    "",
    "## Leakage Audit",
    "",
    f"- raw ids used as categorical features: `{summary['leakage_audit']['raw_ids_used_as_features']}`",
    f"- target/result fields used: `{summary['leakage_audit']['target_or_result_fields_used']}`",
    f"- excluded fields: `{', '.join(summary['leakage_audit']['excluded_feature_sources'])}`",
    "",
    "## Interpretation",
    "",
    "XGBoost is the right off-the-shelf backend for the larger version of this",
    "test, but the feature extractor and holdout are the load-bearing pieces.",
    "This artifact does not test novel mechanism proposal; it only tests whether",
    "structured pre-result features can triage or rank candidate experiments",
    "better than deterministic priors.",
    "",
  ]
  return "\n".join(lines)

def run_cost_model(examples_path:pathlib.Path, out:pathlib.Path, *, backend:str="auto", seed:int=20260614) -> dict[str, Any]:
  examples = _read_jsonl(examples_path)
  train = [row for row in examples if row.get("split") == "train"]
  holdout = [row for row in examples if row.get("split") == "holdout"]
  if not train or not holdout: raise ValueError("examples must contain non-empty train and holdout splits")
  train_maps = [extract_feature_map(row) for row in train]
  holdout_maps = [extract_feature_map(row) for row in holdout]
  vectorizer = FeatureVectorizer().fit(train_maps)
  x_train, x_holdout = vectorizer.transform(train_maps), vectorizer.transform(holdout_maps)

  baseline_predictions = build_baseline_predictions(examples, seed=seed)
  baseline_metrics = {name: score_predictions(examples, preds) for name, preds in baseline_predictions.items()}
  requested = _requested_backends(backend)
  if backend == "xgboost" and not _backend_available("xgboost"):
    raise RuntimeError("xgboost backend requested but xgboost is not installed; install with `pip install xgboost` or use --backend centroid")

  model_predictions: dict[str, list[dict[str, Any]]] = {}
  model_metrics: dict[str, Any] = {}
  backend_details: dict[str, Any] = {"xgboost_available": _backend_available("xgboost"), "requested": backend, "ran": []}
  for name in requested:
    preds, detail = _fit_predict_backend(name, train, holdout, x_train, x_holdout, seed)
    backend_details["ran"].append(detail)
    model_predictions[detail["backend"]] = preds
    model_metrics[detail["backend"]] = score_predictions(examples, preds)

  feature_rows = []
  for row, fmap in zip(train + holdout, train_maps + holdout_maps):
    feature_rows.append({"id": row["id"], "split": row["split"], "features": fmap})
  prediction_rows = [pred for name in sorted(model_predictions) for pred in model_predictions[name]]
  ignored = vectorizer.ignored_holdout_categories(holdout_maps)
  model_best_name = None
  if model_metrics:
    model_best_name = sorted(model_metrics.items(), key=lambda item: (-item[1]["macro_f1"], -item[1]["accuracy"], item[0]))[0][0]
  summary = {
    "kind": "qk_flywheel_cost_model_eval",
    "examples": len(examples),
    "examples_path": str(examples_path),
    "seed": seed,
    "backend_request": backend,
    "backends": backend_details,
    "feature_policy": "pre_result_analytical_context_v0",
    "train_rows": len(train),
    "holdout_rows": len(holdout),
    "baselines": baseline_metrics,
    "models": model_metrics,
    "best_model": model_best_name,
    "conclusion": _conclusion(model_metrics, baseline_metrics),
    "features": {
      "feature_count": len(vectorizer.names),
      "numeric_feature_count": len(vectorizer.numeric),
      "categorical_feature_count": sum(len(values) for values in vectorizer.categorical.values()),
      "numeric_features": vectorizer.numeric,
      "categorical_features": vectorizer.categorical,
      "ignored_holdout_categories": ignored,
      "ignored_holdout_category_count": sum(len(values) for values in ignored.values()),
    },
    "leakage_audit": {
      "excluded_feature_sources": list(FORBIDDEN_FEATURE_SOURCES),
      "raw_ids_used_as_features": False,
      "target_or_result_fields_used": False,
      "feature_names_with_forbidden_tokens": [name for name in vectorizer.names if any(token in name for token in ("label", "reason", "retry", "evidence", "gain", "status", "candidate_gbs", "current_gbs"))],
    },
  }
  out.mkdir(parents=True, exist_ok=True)
  _jsonl(out / "features.jsonl", feature_rows)
  _jsonl(out / "predictions.jsonl", prediction_rows)
  (out / "feature-vocab.json").write_text(json.dumps({"feature_names": vectorizer.names, "numeric": vectorizer.numeric, "categorical": vectorizer.categorical}, indent=2, sort_keys=True))
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_markdown(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Train/evaluate a learned cost model for AMD decode flywheel triage")
  parser.add_argument("--examples", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--backend", choices=("auto", "all", "centroid", "xgboost"), default="auto")
  parser.add_argument("--seed", type=int, default=20260614)
  args = parser.parse_args()
  summary = run_cost_model(args.examples, args.out, backend=args.backend, seed=args.seed)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
