#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, pathlib, random
from collections import Counter, defaultdict
from typing import Any

from extra.qk_flywheel_dataset import LABELS, REASONS

USEFUL_LABELS = {"accept", "raw_accept_unconfirmed", "needs_rerun"}
LABEL_SCORE = {"accept": 1.0, "raw_accept_unconfirmed": 0.75, "needs_rerun": 0.55, "tie": 0.35, "diagnostic_only": 0.25, "reject": 0.0, "construction_blocked": 0.0}

from extra.llm_eval_common import read_jsonl as _read_jsonl

from extra.llm_eval_common import write_jsonl as _jsonl

def _majority(values:list[Any]) -> Any:
  if not values: return None
  return sorted(Counter(values).items(), key=lambda item: (-item[1], str(item[0])))[0][0]

def _train_holdout(examples:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  train = [row for row in examples if row.get("split") == "train"]
  holdout = [row for row in examples if row.get("split") == "holdout"]
  if not train or not holdout: raise ValueError("examples must contain non-empty train and holdout splits")
  return train, holdout

def _prediction(row:dict[str, Any], label:str, reason:str, retry:bool, confidence:float, baseline:str) -> dict[str, Any]:
  return {
    "id": row["id"],
    "baseline": baseline,
    "label": label,
    "reason": reason,
    "retry": retry,
    "confidence": round(float(confidence), 6),
    "score": LABEL_SCORE.get(label, 0.0),
  }

def build_baseline_predictions(examples:list[dict[str, Any]], *, seed:int=20260614) -> dict[str, list[dict[str, Any]]]:
  train, holdout = _train_holdout(examples)
  majority_label = _majority([row["label"] for row in train]) or "reject"
  majority_reason = _majority([row["reason"] for row in train if row["label"] == majority_label]) or "microbench_regression"
  majority_retry = bool(_majority([row["retry"] for row in train if row["label"] == majority_label]))
  label_counts = Counter(row["label"] for row in train)
  reason_counts = Counter(row["reason"] for row in train)
  rng = random.Random(seed)

  mech_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
  family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for row in train:
    mech_rows[row["mechanism"]].append(row)
    family_rows[row["family"]].append(row)

  def prior(rows:list[dict[str, Any]], fallback_name:str) -> tuple[str, str, bool, float]:
    if not rows: return majority_label, majority_reason, majority_retry, label_counts[majority_label] / len(train)
    label = _majority([row["label"] for row in rows]) or majority_label
    reason = _majority([row["reason"] for row in rows if row["label"] == label]) or majority_reason
    retry = bool(_majority([row["retry"] for row in rows if row["label"] == label]))
    confidence = Counter(row["label"] for row in rows)[label] / len(rows)
    return label, reason, retry, confidence

  labels = list(label_counts)
  label_weights = [label_counts[label] for label in labels]
  reasons = list(reason_counts)
  reason_weights = [reason_counts[reason] for reason in reasons]
  out: dict[str, list[dict[str, Any]]] = {"majority_label": [], "reject_all": [], "random_label": [], "mechanism_prior": [], "simple_family_heuristic": []}
  for row in holdout:
    out["majority_label"].append(_prediction(row, majority_label, majority_reason, majority_retry, label_counts[majority_label] / len(train), "majority_label"))
    out["reject_all"].append(_prediction(row, "reject", "microbench_regression", False, 0.75, "reject_all"))
    r_label = rng.choices(labels, weights=label_weights, k=1)[0]
    r_reason = rng.choices(reasons, weights=reason_weights, k=1)[0]
    out["random_label"].append(_prediction(row, r_label, r_reason, r_label in USEFUL_LABELS, label_counts[r_label] / len(train), "random_label"))
    label, reason, retry, confidence = prior(mech_rows.get(row["mechanism"], []), "mechanism_prior")
    out["mechanism_prior"].append(_prediction(row, label, reason, retry, confidence, "mechanism_prior"))
    family_train = family_rows.get(row["family"], [])
    if family_train:
      label, reason, retry, confidence = prior(family_train, "simple_family_heuristic")
    else:
      label, reason, retry, confidence = prior(mech_rows.get(row["mechanism"], []), "mechanism_prior")
    out["simple_family_heuristic"].append(_prediction(row, label, reason, retry, confidence, "simple_family_heuristic"))
  return out

def _macro_f1(targets:list[str], preds:list[str]) -> float:
  labels = sorted(set(targets) | set(preds) | set(LABELS))
  scores = []
  for label in labels:
    tp = sum(t == label and p == label for t, p in zip(targets, preds))
    fp = sum(t != label and p == label for t, p in zip(targets, preds))
    fn = sum(t == label and p != label for t, p in zip(targets, preds))
    if tp == fp == fn == 0: continue
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
  return 0.0 if not scores else sum(scores) / len(scores)

def _ranking_metrics(rows:list[dict[str, Any]], preds:list[dict[str, Any]]) -> dict[str, Any]:
  by_id = {row["id"]: row for row in rows}
  groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
  for pred in preds:
    row = by_id[pred["id"]]
    groups[f"{row['family']}:{row['model']}"].append((row, pred))
  p1 = p3 = ndcg = groups_seen = 0.0
  for items in groups.values():
    if len(items) < 2: continue
    ranked = sorted(items, key=lambda item: (-item[1].get("score", 0.0), item[0]["id"]))
    useful = [1.0 if row["label"] in USEFUL_LABELS else 0.0 for row, _ in ranked]
    p1 += useful[0]
    p3 += sum(useful[:3]) / min(3, len(useful))
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(useful))
    ideal = sorted(useful, reverse=True)
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))
    ndcg += 0.0 if idcg == 0 else dcg / idcg
    groups_seen += 1
  if groups_seen == 0: return {"groups": 0, "precision_at_1": None, "precision_at_3": None, "ndcg": None}
  return {"groups": int(groups_seen), "precision_at_1": p1 / groups_seen, "precision_at_3": p3 / groups_seen, "ndcg": ndcg / groups_seen}

def score_predictions(examples:list[dict[str, Any]], predictions:list[dict[str, Any]]) -> dict[str, Any]:
  _, holdout = _train_holdout(examples)
  by_id = {row["id"]: row for row in holdout}
  if {pred["id"] for pred in predictions} != set(by_id):
    raise ValueError("predictions must contain exactly the holdout ids")
  ordered = [next(pred for pred in predictions if pred["id"] == row["id"]) for row in holdout]
  targets = [row["label"] for row in holdout]
  pred_labels = [pred["label"] for pred in ordered]
  label_matches = [target == pred for target, pred in zip(targets, pred_labels)]
  reason_denom = sum(label_matches)
  reason_num = sum(pred["reason"] == row["reason"] for row, pred, ok in zip(holdout, ordered, label_matches) if ok)
  non_accept_rows = [row for row in holdout if row["label"] != "accept"]
  fp_accepts = sum(pred["label"] == "accept" and by_id[pred["id"]]["label"] != "accept" for pred in ordered)
  retry_tp = sum(pred["retry"] is True and by_id[pred["id"]]["retry"] is True for pred in ordered)
  retry_fp = sum(pred["retry"] is True and by_id[pred["id"]]["retry"] is False for pred in ordered)
  retry_fn = sum(pred["retry"] is False and by_id[pred["id"]]["retry"] is True for pred in ordered)
  return {
    "rows": len(holdout),
    "accuracy": sum(label_matches) / len(holdout),
    "macro_f1": _macro_f1(targets, pred_labels),
    "reason_accuracy_on_label_match": None if reason_denom == 0 else reason_num / reason_denom,
    "false_positive_accept_rate": 0.0 if not non_accept_rows else fp_accepts / len(non_accept_rows),
    "retry_precision": None if retry_tp + retry_fp == 0 else retry_tp / (retry_tp + retry_fp),
    "retry_recall": None if retry_tp + retry_fn == 0 else retry_tp / (retry_tp + retry_fn),
    "target_labels": dict(sorted(Counter(targets).items())),
    "pred_labels": dict(sorted(Counter(pred_labels).items())),
    "ranking": _ranking_metrics(holdout, ordered),
  }

def _parse_prediction_text(row:dict[str, Any], name:str) -> dict[str, Any]:
  text = str(row.get("text", "")).strip()
  try:
    data = json.loads(text)
  except json.JSONDecodeError:
    return _prediction({"id": row["id"]}, "invalid_output", "invalid_output", False, 0.0, name) | {"parse_ok": False, "text": text}
  if not isinstance(data, dict):
    return _prediction({"id": row["id"]}, "invalid_output", "invalid_output", False, 0.0, name) | {"parse_ok": False, "text": text}
  label = data.get("label")
  reason = data.get("reason")
  retry = data.get("retry")
  confidence = data.get("confidence", 0.5)
  if not isinstance(label, str) or not isinstance(reason, str) or not isinstance(retry, bool):
    return _prediction({"id": row["id"]}, "invalid_output", "invalid_output", False, 0.0, name) | {"parse_ok": False, "text": text, "parsed": data}
  if not isinstance(confidence, (int, float)): confidence = 0.5
  return _prediction({"id": row["id"]}, label, reason, retry, float(confidence), name) | {"parse_ok": True, "text": text, "parsed": data}

def load_rollout_predictions(spec:str) -> tuple[str, list[dict[str, Any]]]:
  if "=" not in spec: raise ValueError("--rollout must be NAME=PATH")
  name, raw_path = spec.split("=", 1)
  if not name: raise ValueError("--rollout name must be non-empty")
  path = pathlib.Path(raw_path)
  rollouts = _read_jsonl(path / "rollouts.jsonl" if path.is_dir() else path)
  return name, [_parse_prediction_text(row, name) for row in rollouts]

def evaluate_baselines(examples:list[dict[str, Any]], *, seed:int=20260614, rollout_specs:list[str]|None=None) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
  baseline_predictions = build_baseline_predictions(examples, seed=seed)
  metrics = {name: score_predictions(examples, preds) for name, preds in baseline_predictions.items()}
  model_metrics: dict[str, Any] = {}
  rollout_predictions: dict[str, list[dict[str, Any]]] = {}
  for spec in rollout_specs or []:
    name, preds = load_rollout_predictions(spec)
    rollout_predictions[name] = preds
    model_metrics[name] = score_predictions(examples, preds)
  holdout = [row for row in examples if row["split"] == "holdout"]
  conclusion = "baseline_ready"
  if len(holdout) < 8 or len({row["label"] for row in holdout}) < 2: conclusion = "dataset_insufficient"
  best_name, best = sorted(metrics.items(), key=lambda item: (-item[1]["macro_f1"], -item[1]["accuracy"], item[0]))[0]
  if model_metrics:
    best_model = sorted(model_metrics.items(), key=lambda item: (-item[1]["macro_f1"], -item[1]["accuracy"], item[0]))[0]
    mechanism_prior = metrics.get("mechanism_prior", {"macro_f1": 0.0, "false_positive_accept_rate": 1.0})
    conclusion = "shadow_ready" if best_model[1]["macro_f1"] > mechanism_prior["macro_f1"] and best_model[1]["false_positive_accept_rate"] <= 0.05 else "no_signal"
  return {
    "kind": "qk_flywheel_triage_baseline_eval",
    "seed": seed,
    "examples": len(examples),
    "train_rows": sum(row["split"] == "train" for row in examples),
    "holdout_rows": len(holdout),
    "baselines": metrics,
    "models": model_metrics,
    "best_baseline": best_name,
    "best_baseline_macro_f1": best["macro_f1"],
    "conclusion": conclusion,
  }, baseline_predictions | rollout_predictions

def _markdown(summary:dict[str, Any]) -> str:
  lines = [
    "# AMD Decode Flywheel Triage Baselines",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- examples: `{summary['examples']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- best baseline: `{summary['best_baseline']}`",
    "",
    "| baseline | accuracy | macro-F1 | false accept | retry precision | retry recall | p@1 | ndcg |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  def fmt(v:Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"
  for name, row in summary["baselines"].items():
    ranking = row["ranking"]
    lines.append(
      f"| `{name}` | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['false_positive_accept_rate'])} | "
      f"{fmt(row['retry_precision'])} | {fmt(row['retry_recall'])} | {fmt(ranking['precision_at_1'])} | {fmt(ranking['ndcg'])} |"
    )
  if summary.get("models"):
    lines += ["", "## Model Predictions", "", "| model | accuracy | macro-F1 | false accept | retry precision | retry recall | p@1 | ndcg |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary["models"].items():
      ranking = row["ranking"]
      lines.append(
        f"| `{name}` | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['false_positive_accept_rate'])} | "
        f"{fmt(row['retry_precision'])} | {fmt(row['retry_recall'])} | {fmt(ranking['precision_at_1'])} | {fmt(ranking['ndcg'])} |"
      )
  lines.append("")
  return "\n".join(lines)

def write_eval(examples_path:pathlib.Path, out:pathlib.Path, *, seed:int=20260614, rollout_specs:list[str]|None=None) -> dict[str, Any]:
  examples = _read_jsonl(examples_path)
  summary, predictions = evaluate_baselines(examples, seed=seed, rollout_specs=rollout_specs)
  out.mkdir(parents=True, exist_ok=True)
  all_preds = [pred for name in sorted(predictions) for pred in predictions[name]]
  _jsonl(out / "baseline-predictions.jsonl", all_preds)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_markdown(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Evaluate deterministic baselines for AMD decode flywheel triage")
  parser.add_argument("--examples", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--seed", type=int, default=20260614)
  parser.add_argument("--rollout", action="append", default=[], help="Optional model rollout predictions as NAME=DIR_OR_JSONL")
  args = parser.parse_args()
  summary = write_eval(args.examples, args.out, seed=args.seed, rollout_specs=args.rollout)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
