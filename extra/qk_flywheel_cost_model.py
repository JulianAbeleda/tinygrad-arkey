#!/usr/bin/env python3
"""Learned cost model for AMD decode flywheel triage.

Single module owning the whole cost-model surface: leak-free pre-result feature
extraction + vectorizer, the centroid and xgboost prediction backends, backend
dispatch, and the eval/report driver. The two backends are distinct algorithms
(irreducible dual-backend dispatch); the feature policy and holdout are the
load-bearing pieces.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, pathlib, re
from typing import Any

import numpy as np

from extra.qk_flywheel_dataset import LABELS, parse_load_width_words, parse_opts
from extra.qk_flywheel_triage_eval import LABEL_SCORE, _majority, build_baseline_predictions, score_predictions

from extra.llm_eval_common import read_jsonl as _read_jsonl

from extra.llm_eval_common import write_jsonl as _jsonl

# --- Feature policy + extraction ----------------------------------------------------
FORBIDDEN_FEATURE_SOURCES = (
  "id", "candidate_id", "label", "reason", "retry", "evidence", "source_files",
  "split", "family_order", "status", "gain", "gain_pct", "candidate_gbs",
  "current_gbs", "decision", "correctness_ok", "ab_match_result",
)

SAFE_TOP_LEVEL_CATEGORICAL = ("row_kind", "family", "role", "format", "mechanism", "prediction_stage")
SAFE_CONTEXT_CATEGORICAL = ("mode", "reference", "reference_mode", "model_size")
SAFE_SCHEDULE_CATEGORICAL = ("activation_cache", "codegen_mode", "family", "format", "name", "reduction_mode", "role", "semantic_object")

def _model_size_b(value:Any) -> float:
  text = str(value or "")
  m = re.search(r"(\d+)\s*[bB]", text)
  return float(m.group(1)) if m else 0.0

def _tensor_block(value:Any) -> float:
  m = re.search(r"blk[.-](\d+)", str(value or ""))
  return float(m.group(1)) if m else 0.0

def _as_float(value:Any, default:float=0.0) -> float:
  if value is None: return default
  if isinstance(value, bool): return 1.0 if value else 0.0
  if isinstance(value, (int, float)) and math.isfinite(float(value)): return float(value)
  return default

def _add_num(out:dict[str, Any], name:str, value:Any, default:float=0.0) -> None:
  out[name] = _as_float(value, default)

def _add_cat(out:dict[str, Any], name:str, value:Any) -> None:
  if value is None: return
  text = str(value)
  if text: out[name] = text

def _add_flat_group(out:dict[str, Any], prefix:str, data:Any) -> None:
  if not isinstance(data, dict): return
  for key, value in sorted(data.items()):
    if key in ("outcome", "label", "reason", "retry", "evidence", "source_files"): continue
    name = f"{prefix}_{key}"
    if isinstance(value, str):
      _add_cat(out, name, value)
    elif isinstance(value, bool):
      _add_num(out, name, value)
    elif isinstance(value, (int, float)):
      _add_num(out, name, value)
    elif isinstance(value, list):
      _add_num(out, f"{name}_count", len(value))

def _parse_load_width(value:Any) -> float:
  return float(parse_load_width_words(value))

def _opts_features(opts:Any) -> dict[str, float]:
  return {key: float(val) for key, val in parse_opts(opts).items()}

def _add_opts(out:dict[str, Any], prefix:str, opts:Any) -> None:
  for key, value in _opts_features(opts).items(): out[f"{prefix}_{key}"] = value

def _text_flags(*values:Any) -> dict[str, float]:
  text = " ".join(str(v or "") for v in values).lower().replace("-", "_")
  return {
    "flag_direct_out": float("direct_out" in text),
    "flag_row_upcast": float("row_upcast" in text or "upcast" in text),
    "flag_reduce_unroll": float("reduce_unroll" in text or "unroll" in text),
    "flag_two_dim_local": float("two_dim_local" in text),
    "flag_packed_load": float("packed_load" in text or "packed_word" in text),
    "flag_vector_load": float("vector_load" in text),
    "flag_tile_custom": float("tile_custom" in text),
    "flag_qk_block_dot": float("qk_block_dot" in text),
    "flag_split_k": float("split_k" in text),
  }

def extract_feature_map(row:dict[str, Any]) -> dict[str, Any]:
  """Return leak-free pre-result features for a kernel-triage row."""
  out: dict[str, Any] = {}
  is_v1 = row.get("schema_version") == "kernel_triage_v1"
  top_level = () if is_v1 else SAFE_TOP_LEVEL_CATEGORICAL
  for key in top_level: _add_cat(out, key, row.get(key))
  if not is_v1: _add_cat(out, "model_size_cat", f"{int(_model_size_b(row.get('model')))}B" if _model_size_b(row.get("model")) else "unknown")
  _add_num(out, "model_size_b", _model_size_b(row.get("model")))
  _add_num(out, "tensor_block", _tensor_block(row.get("tensor")))

  ctx = row.get("pre_result_context") or {}
  if not isinstance(ctx, dict): ctx = {}
  if not is_v1:
    for key in SAFE_CONTEXT_CATEGORICAL: _add_cat(out, f"context_{key}", ctx.get(key))
  for key in ("ab_match", "candidate_stable", "reference_stable", "full_decode_supported", "full_decode_ready", "source_ok", "wide_loads"):
    _add_num(out, f"context_{key}", ctx.get(key))
  for key in ("parts", "runs"):
    _add_num(out, f"context_{key}", ctx.get(key))
  _add_num(out, "context_meaningful_threshold_pct", ctx.get("meaningful_gain_pct"))
  _add_num(out, "context_required_threshold_pct", ctx.get("required_gain_pct"))

  shape = ctx.get("shape")
  if isinstance(shape, dict):
    _add_num(out, "shape_rows", shape.get("rows"))
    _add_num(out, "shape_k", shape.get("k"))
    _add_num(out, "shape_parts", shape.get("parts"))
  elif isinstance(shape, list):
    for idx, value in enumerate(shape[:4]): _add_num(out, f"shape_dim{idx}", value)

  load_width = ctx.get("load_width") if isinstance(ctx.get("load_width"), dict) else {}
  if load_width:
    _add_num(out, "load_width_inferred_words", _parse_load_width(load_width.get("inferred")))
    _add_num(out, "load_width_kernel_words", _parse_load_width(load_width.get("kernel")))
    _add_num(out, "load_width_report_words", _parse_load_width(load_width.get("report")))

  schedule = ctx.get("schedule") or {}
  if not isinstance(schedule, dict): schedule = {}
  if not is_v1:
    for key in SAFE_SCHEDULE_CATEGORICAL: _add_cat(out, f"schedule_{key}", schedule.get(key))
  for key in ("full_decode_supported", "group_unroll", "k_tile_blocks", "lane_width", "parts", "row_tile"):
    _add_num(out, f"schedule_{key}", schedule.get(key))
  _add_num(out, "schedule_requires_count", len(schedule.get("requires") or []) if isinstance(schedule.get("requires"), list) else 0)
  _add_opts(out, "schedule_opts", schedule.get("opts"))

  _add_opts(out, "context_opts", ctx.get("opts"))
  change = ctx.get("change") or {}
  if isinstance(change, dict):
    _add_cat(out, "change_format", change.get("format"))
    for side in ("from", "to"):
      side_row = change.get(side) or {}
      if not isinstance(side_row, dict): side_row = {}
      _add_cat(out, f"change_{side}_family", side_row.get("family"))
      _add_num(out, f"change_{side}_parts", side_row.get("parts"))
      _add_opts(out, f"change_{side}_opts", side_row.get("opts"))
    out["change_delta_parts"] = out.get("change_to_parts", 0.0) - out.get("change_from_parts", 0.0)
    out["change_delta_local0"] = out.get("change_to_opts_local0_arg", 0.0) - out.get("change_from_opts_local0_arg", 0.0)
    out["change_delta_local1"] = out.get("change_to_opts_local1_arg", 0.0) - out.get("change_from_opts_local1_arg", 0.0)

  text_bits = _text_flags(
    row.get("candidate_id"), ctx.get("candidate"), ctx.get("candidate_id"), ctx.get("mode"),
    schedule.get("name"), schedule.get("codegen_mode"), schedule.get("reduction_mode"),
    row.get("mechanism"),
  )
  out.update(text_bits)

  candidate_record = row.get("candidate_record") or {}
  if isinstance(candidate_record, dict):
    _add_cat(out, "candidate_schema_version", candidate_record.get("schema_version"))
    _add_num(out, "candidate_frozen_before_outcome", candidate_record.get("frozen_before_outcome"))
    _add_flat_group(out, "v1_static", candidate_record.get("static_features"))
    _add_flat_group(out, "v1_uop", candidate_record.get("uop_features"))
    _add_flat_group(out, "v1_profile", candidate_record.get("profile_features"))

  parts = max(out.get("schedule_parts", 0.0), out.get("context_parts", 0.0), out.get("shape_parts", 0.0), out.get("change_to_parts", 0.0), 1.0)
  row_tile = max(out.get("schedule_row_tile", 0.0), out.get("schedule_opts_local0_arg", 0.0), out.get("change_to_opts_local0_arg", 0.0), 1.0)
  lane_width = max(out.get("schedule_lane_width", 0.0), out.get("load_width_inferred_words", 0.0), out.get("load_width_kernel_words", 0.0), 1.0)
  group_unroll = max(out.get("schedule_group_unroll", 0.0), out.get("schedule_opts_unroll_arg", 0.0), out.get("change_to_opts_unroll_arg", 0.0), 1.0)
  local_axes = max(out.get("schedule_opts_local_count", 0.0), out.get("context_opts_local_count", 0.0), out.get("change_to_opts_local_count", 0.0))
  opts_count = out.get("schedule_opts_count", 0.0) + out.get("context_opts_count", 0.0) + out.get("change_to_opts_count", 0.0)
  requires = out.get("schedule_requires_count", 0.0)

  out["ana_data_reuse_shared_proxy"] = math.log1p(row_tile * parts)
  out["ana_register_reuse_proxy"] = math.log1p(row_tile * lane_width)
  out["ana_ilp_proxy"] = math.log1p(lane_width * group_unroll)
  out["ana_warp_concurrency_proxy"] = math.log1p(max(row_tile / 32.0, 0.0) * parts)
  out["ana_load_width_proxy"] = math.log1p(lane_width)
  out["ana_load_imbalance_proxy"] = math.log1p(abs(parts - 1.0) + local_axes)
  out["ana_schedule_complexity_proxy"] = math.log1p(opts_count + requires + float(group_unroll > 1.0) + float(out.get("schedule_k_tile_blocks", 0.0) > 0.0))
  return {key: out[key] for key in sorted(out)}

class FeatureVectorizer:
  def __init__(self) -> None:
    self.numeric: list[str] = []
    self.categorical: dict[str, list[str]] = {}
    self.names: list[str] = []

  def fit(self, maps:list[dict[str, Any]]) -> "FeatureVectorizer":
    numeric, cats = set(), {}
    for fmap in maps:
      for key, value in fmap.items():
        if isinstance(value, str): cats.setdefault(key, set()).add(value)
        else: numeric.add(key)
    self.numeric = sorted(numeric)
    self.categorical = {key: sorted(values) for key, values in sorted(cats.items())}
    self.names = list(self.numeric)
    for key, values in self.categorical.items():
      self.names += [f"{key}={value}" for value in values]
    return self

  def transform_one(self, fmap:dict[str, Any]) -> np.ndarray:
    values = []
    for key in self.numeric:
      value = fmap.get(key, 0.0)
      values.append(_as_float(value))
    for key, allowed in self.categorical.items():
      value = fmap.get(key)
      values += [1.0 if value == option else 0.0 for option in allowed]
    return np.array(values, dtype=np.float32)

  def transform(self, maps:list[dict[str, Any]]) -> np.ndarray:
    if not self.names: return np.zeros((len(maps), 0), dtype=np.float32)
    return np.stack([self.transform_one(fmap) for fmap in maps])

  def ignored_holdout_categories(self, maps:list[dict[str, Any]]) -> dict[str, list[str]]:
    ignored: dict[str, set[str]] = {}
    for fmap in maps:
      for key, value in fmap.items():
        if isinstance(value, str) and key in self.categorical and value not in self.categorical[key]:
          ignored.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in sorted(ignored.items())}

# --- Prediction backends ------------------------------------------------------------
RANK_RELEVANCE = {"accept": 4, "raw_accept_unconfirmed": 3, "needs_rerun": 2, "tie": 1, "diagnostic_only": 0, "reject": 0, "construction_blocked": 0}

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

def _portable_features(fmap:dict[str, Any]) -> dict[str, Any]:
  # Round float features to 6 decimals before serialization so the artifact is
  # byte-identical across platforms. Unrounded values include transcendental
  # results (e.g. log1p proxies) whose last ULP differs between macOS libm and
  # Linux glibc; that drift breaks the cross-machine golden lock without changing
  # any prediction (predictions are already rounded the same way). The in-memory
  # map fed to the vectorizer is left untouched.
  return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fmap.items()}

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
    feature_rows.append({"id": row["id"], "split": row["split"], "features": _portable_features(fmap)})
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
