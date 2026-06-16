#!/usr/bin/env python3
"""Leak-free pre-result feature extraction for the flywheel cost model.

Extracted verbatim from qk_flywheel_cost_model.py as a behavior-preserving
move (NFC). Owns the feature policy: the forbidden/safe field lists, the
per-field extraction helpers, extract_feature_map, and FeatureVectorizer.
"""
from __future__ import annotations

import math, re
from typing import Any

import numpy as np

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
  text = str(value or "").lower()
  m = re.search(r"x(\d+)", text)
  if m: return float(m.group(1))
  m = re.search(r"(\d+)", text)
  return float(m.group(1)) if m else 0.0

def _opts_features(opts:Any) -> dict[str, float]:
  out = {"count": 0.0, "local_count": 0.0, "local0_arg": 0.0, "local1_arg": 0.0, "upcast_count": 0.0, "upcast_arg": 0.0, "unroll_count": 0.0, "unroll_arg": 0.0}
  if not isinstance(opts, list): return out
  out["count"] = float(len(opts))
  for raw in opts:
    text = str(raw)
    m = re.search(r"LOCAL:(\d+):(\d+)", text)
    if not m: m = re.search(r"OptOps\.LOCAL.*axis=(\d+).*arg=(\d+)", text)
    if m:
      axis, arg = int(m.group(1)), float(m.group(2))
      out["local_count"] += 1.0
      if axis == 0: out["local0_arg"] = max(out["local0_arg"], arg)
      if axis == 1: out["local1_arg"] = max(out["local1_arg"], arg)
    m = re.search(r"UPCAST:(\d+):(\d+)", text)
    if m:
      out["upcast_count"] += 1.0
      out["upcast_arg"] = max(out["upcast_arg"], float(m.group(2)))
    m = re.search(r"UNROLL:(\d+):(\d+)", text)
    if m:
      out["unroll_count"] += 1.0
      out["unroll_arg"] = max(out["unroll_arg"], float(m.group(2)))
  return out

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
