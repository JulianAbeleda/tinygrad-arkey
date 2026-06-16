#!/usr/bin/env python3
"""Prediction backends for the flywheel cost model.

Extracted verbatim from qk_flywheel_cost_model.py as a behavior-preserving
move (NFC). Owns the centroid and xgboost backends, the label policy, and the
backend selection logic. Operates on already-vectorized feature matrices.
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from typing import Any

import numpy as np

from extra.qk_flywheel_dataset import LABELS
from extra.qk_flywheel_triage_eval import LABEL_SCORE

RANK_RELEVANCE = {"accept": 4, "raw_accept_unconfirmed": 3, "needs_rerun": 2, "tie": 1, "diagnostic_only": 0, "reject": 0, "construction_blocked": 0}

def _majority(values:list[Any]) -> Any:
  return sorted(Counter(values).items(), key=lambda item: (-item[1], str(item[0])))[0][0]

def _label_policy(train:list[dict[str, Any]]) -> tuple[str, dict[str, tuple[str, bool]]]:
  majority_label = _majority([row["label"] for row in train])
  policy: dict[str, tuple[str, bool]] = {}
  for label in sorted({row["label"] for row in train}):
    rows = [row for row in train if row["label"] == label]
    policy[label] = (_majority([row["reason"] for row in rows]), bool(_majority([row["retry"] for row in rows])))
  return majority_label, policy

def _prediction(row:dict[str, Any], label:str, reason:str, retry:bool, confidence:float, score:float, model:str) -> dict[str, Any]:
  return {
    "id": row["id"],
    "model": model,
    "label": label,
    "reason": reason,
    "retry": bool(retry),
    "confidence": round(float(confidence), 6),
    "score": round(float(score), 6),
  }

def _centroid_predictions(train:list[dict[str, Any]], holdout:list[dict[str, Any]], x_train:np.ndarray, x_holdout:np.ndarray, seed:int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  del seed
  majority_label, policy = _label_policy(train)
  labels = sorted({row["label"] for row in train}, key=lambda label: LABELS.index(label) if label in LABELS else 999)
  centroids = {}
  norms = {}
  for label in labels:
    idx = [i for i, row in enumerate(train) if row["label"] == label]
    centroid = x_train[idx].mean(axis=0) if idx else np.zeros((x_train.shape[1],), dtype=np.float32)
    centroids[label] = centroid
    norms[label] = float(np.linalg.norm(centroid))
  predictions = []
  for row, vec in zip(holdout, x_holdout):
    vnorm = float(np.linalg.norm(vec))
    sims = []
    for label in labels:
      denom = vnorm * norms[label]
      sim = -1.0 if denom <= 0.0 else float(np.dot(vec, centroids[label]) / denom)
      sims.append((label, sim))
    if not sims:
      label, confidence, score = majority_label, 1.0, LABEL_SCORE.get(majority_label, 0.0)
    else:
      best_sim = max(score for _, score in sims)
      weights = np.exp(np.array([score - best_sim for _, score in sims], dtype=np.float64))
      weights = weights / max(float(weights.sum()), 1e-12)
      best_idx = int(np.argmax(weights))
      label = sims[best_idx][0]
      confidence = float(weights[best_idx])
      score = sum(float(weight) * LABEL_SCORE.get(sim_label, 0.0) for weight, (sim_label, _) in zip(weights, sims))
    reason, retry = policy.get(label, policy[majority_label])
    predictions.append(_prediction(row, label, reason, retry, confidence, score, "centroid_cost_model"))
  return predictions, {"backend": "centroid", "status": "ok", "label_classes": labels}

def _group_ids(rows:list[dict[str, Any]]) -> np.ndarray:
  groups = {key: idx for idx, key in enumerate(sorted({f"{row.get('family')}:{row.get('model')}" for row in rows}))}
  return np.array([groups[f"{row.get('family')}:{row.get('model')}"] for row in rows], dtype=np.int32)

def _xgboost_predictions(train:list[dict[str, Any]], holdout:list[dict[str, Any]], x_train:np.ndarray, x_holdout:np.ndarray, seed:int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  import xgboost as xgb
  majority_label, policy = _label_policy(train)
  labels = sorted({row["label"] for row in train}, key=lambda label: LABELS.index(label) if label in LABELS else 999)
  label_to_idx = {label: idx for idx, label in enumerate(labels)}
  y_label = np.array([label_to_idx[row["label"]] for row in train], dtype=np.int32)
  y_score = np.array([LABEL_SCORE.get(row["label"], 0.0) for row in train], dtype=np.float32)
  y_relevance = np.array([RANK_RELEVANCE.get(row["label"], 0) for row in train], dtype=np.float32)
  dtrain_label = xgb.DMatrix(x_train, label=y_label, feature_names=[f"f{idx}" for idx in range(x_train.shape[1])])
  dholdout = xgb.DMatrix(x_holdout, feature_names=[f"f{idx}" for idx in range(x_holdout.shape[1])])
  clf = xgb.train(
    {
      "objective": "multi:softprob", "num_class": len(labels), "eval_metric": "mlogloss",
      "tree_method": "hist", "max_depth": 2, "eta": 0.2, "lambda": 4.0,
      "seed": seed, "nthread": 1, "verbosity": 0,
    },
    dtrain_label,
    num_boost_round=32,
  )
  probs = np.asarray(clf.predict(dholdout), dtype=np.float64)
  if probs.ndim == 1: probs = probs.reshape((-1, len(labels)))
  rank_status = "ranker"
  try:
    qid = _group_ids(train)
    order = np.argsort(qid, kind="stable")
    group_sizes = [int(np.sum(qid[order] == group_id)) for group_id in sorted(set(qid.tolist()))]
    dtrain_rank = xgb.DMatrix(x_train[order], label=y_relevance[order], feature_names=[f"f{idx}" for idx in range(x_train.shape[1])])
    dtrain_rank.set_group(group_sizes)
    ranker = xgb.train(
      {
        "objective": "rank:ndcg", "eval_metric": "ndcg", "tree_method": "hist",
        "max_depth": 2, "eta": 0.2, "lambda": 4.0, "seed": seed, "nthread": 1,
        "verbosity": 0,
      },
      dtrain_rank,
      num_boost_round=32,
    )
    rank_scores = np.asarray(ranker.predict(dholdout), dtype=np.float64)
  except Exception as exc:
    rank_status = f"regressor_fallback:{type(exc).__name__}"
    dtrain_score = xgb.DMatrix(x_train, label=y_score, feature_names=[f"f{idx}" for idx in range(x_train.shape[1])])
    reg = xgb.train(
      {
        "objective": "reg:squarederror", "tree_method": "hist", "max_depth": 2,
        "eta": 0.2, "lambda": 4.0, "seed": seed, "nthread": 1, "verbosity": 0,
      },
      dtrain_score,
      num_boost_round=32,
    )
    rank_scores = np.asarray(reg.predict(dholdout), dtype=np.float64)

  predictions = []
  for row, prob, rank_score in zip(holdout, probs, rank_scores):
    idx = int(np.argmax(prob)) if len(prob) else 0
    label = labels[idx] if idx < len(labels) else majority_label
    reason, retry = policy.get(label, policy[majority_label])
    predictions.append(_prediction(row, label, reason, retry, float(prob[idx]) if len(prob) else 0.0, float(rank_score), "xgboost_cost_model"))
  return predictions, {"backend": "xgboost", "status": "ok", "label_classes": labels, "rank_score_model": rank_status, "xgboost_version": getattr(xgb, "__version__", "unknown")}

def _backend_available(name:str) -> bool:
  return name == "centroid" or importlib.util.find_spec(name) is not None

def _fit_predict_backend(backend:str, train:list[dict[str, Any]], holdout:list[dict[str, Any]], x_train:np.ndarray, x_holdout:np.ndarray, seed:int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  if backend == "centroid": return _centroid_predictions(train, holdout, x_train, x_holdout, seed)
  if backend == "xgboost": return _xgboost_predictions(train, holdout, x_train, x_holdout, seed)
  raise ValueError(f"unknown backend {backend!r}")

def _requested_backends(request:str) -> list[str]:
  if request == "centroid": return ["centroid"]
  if request == "xgboost": return ["xgboost"]
  if request == "all": return ["centroid", "xgboost"] if _backend_available("xgboost") else ["centroid"]
  if request == "auto": return ["xgboost"] if _backend_available("xgboost") else ["centroid"]
  raise ValueError(f"unknown backend request {request!r}")
