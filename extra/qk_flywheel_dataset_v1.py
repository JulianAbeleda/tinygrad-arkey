#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, pathlib, re
from collections import Counter
from typing import Any

from extra import qk_flywheel_dataset as v0

V1_MECHANISMS = tuple(sorted(set(v0.MECHANISMS) | {"row_upcast", "reduce_unroll", "two_dim_local"}))
SCHEDULE_NAME_ALIASES = {
  "direct_out_tensor": "direct_out",
  "direct-out-tensor": "direct_out",
  "direct_out": "direct_out",
  "direct-out": "direct_out",
  "row_upcast2": "row_upcast",
  "row-upcast2": "row_upcast",
  "reduce_unroll4": "reduce_unroll",
  "reduce-unroll4": "reduce_unroll",
  "two_dim_local4": "two_dim_local",
  "two-dim-local4": "two_dim_local",
}

def _slug(value:Any) -> str:
  return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

def _norm_schedule_name(value:Any) -> str:
  text = _slug(value)
  return SCHEDULE_NAME_ALIASES.get(text, text or "unknown")

def _norm_schedule_family(value:Any) -> str:
  text = _slug(value)
  if text.startswith(("q4_k_packed", "q6_k_packed")): return "packed_quant_gemv"
  return text or "unknown"

def _norm_semantic_object(value:Any) -> str:
  text = _slug(value)
  if text in ("packed_quant_gemv_schedule", "packed_quant_gemv_codegen"): return "packed_quant_gemv"
  return text or "unknown"

def _norm_mechanism(row:dict[str, Any]) -> str:
  ctx = row.get("pre_result_context") or {}
  schedule = ctx.get("schedule") if isinstance(ctx, dict) else None
  schedule_name = _norm_schedule_name((schedule or {}).get("name") if isinstance(schedule, dict) else None)
  text = " ".join(str(v or "") for v in (
    row.get("mechanism"), row.get("id"), row.get("candidate_id"),
    ctx.get("candidate") if isinstance(ctx, dict) else None,
    ctx.get("candidate_id") if isinstance(ctx, dict) else None,
    ctx.get("mode") if isinstance(ctx, dict) else None,
    schedule_name,
  )).lower().replace("-", "_")
  if "row_upcast" in text: return "row_upcast"
  if "reduce_unroll" in text or "unroll4" in text: return "reduce_unroll"
  if "two_dim_local" in text: return "two_dim_local"
  if "packed_load" in text or "packed_word" in text: return "packed_word_lane_unroll"
  if "vector_load" in text: return "vector_load" if row.get("family") != "threeway_load" else "wide_load_only"
  if "qk_block_dot" in text: return "qk_block_dot"
  if "direct_out" in text: return "direct_output"
  if "tile_custom" in text: return "tile_custom"
  if "row_group" in text: return "row_grouping"
  if "parts_local_policy" in text or " local" in f" {text}" or "_local" in text or "parts" in text: return "parts_local_policy"
  mech = str(row.get("mechanism") or "unknown")
  return mech if mech in V1_MECHANISMS else "unknown"

def _opts_features(opts:Any) -> dict[str, Any]:
  out = {"opt_count": 0, "local_axes": 0, "local0": 0, "local1": 0, "upcast_axes": 0, "upcast": 0, "unroll_axes": 0, "unroll": 0}
  if not isinstance(opts, list): return out
  out["opt_count"] = len(opts)
  for raw in opts:
    text = str(raw)
    m = re.search(r"LOCAL:(\d+):(\d+)", text)
    if not m: m = re.search(r"OptOps\.LOCAL.*axis=(\d+).*arg=(\d+)", text)
    if m:
      axis, arg = int(m.group(1)), int(m.group(2))
      out["local_axes"] += 1
      if axis == 0: out["local0"] = max(out["local0"], arg)
      if axis == 1: out["local1"] = max(out["local1"], arg)
    m = re.search(r"UPCAST:(\d+):(\d+)", text)
    if m:
      out["upcast_axes"] += 1
      out["upcast"] = max(out["upcast"], int(m.group(2)))
    m = re.search(r"UNROLL:(\d+):(\d+)", text)
    if m:
      out["unroll_axes"] += 1
      out["unroll"] = max(out["unroll"], int(m.group(2)))
  return out

def _shape_features(value:Any) -> dict[str, Any]:
  if isinstance(value, dict):
    return {
      "shape_rows": int(value.get("rows") or 0),
      "shape_k": int(value.get("k") or 0),
      "shape_parts": int(value.get("parts") or 0),
    }
  if isinstance(value, list):
    return {f"shape_dim{idx}": int(item) if isinstance(item, int) else 0 for idx, item in enumerate(value[:4])}
  return {}

def _load_width_words(value:Any) -> int:
  text = str(value or "").lower()
  m = re.search(r"x(\d+)", text)
  if m: return int(m.group(1))
  m = re.search(r"(\d+)", text)
  return int(m.group(1)) if m else 0

def _normalize_pre_result_context(ctx:dict[str, Any]) -> dict[str, Any]:
  out = copy.deepcopy(ctx)
  schedule = out.get("schedule")
  if isinstance(schedule, dict):
    schedule.setdefault("raw_name", schedule.get("name"))
    schedule.setdefault("raw_family", schedule.get("family"))
    schedule.setdefault("raw_semantic_object", schedule.get("semantic_object"))
    schedule["name"] = _norm_schedule_name(schedule.get("name"))
    schedule["family"] = _norm_schedule_family(schedule.get("family"))
    schedule["semantic_object"] = _norm_semantic_object(schedule.get("semantic_object"))
  return out

def _static_features(row:dict[str, Any], normalized_ctx:dict[str, Any], mechanism:str) -> dict[str, Any]:
  schedule = normalized_ctx.get("schedule") if isinstance(normalized_ctx.get("schedule"), dict) else {}
  load_width = normalized_ctx.get("load_width") if isinstance(normalized_ctx.get("load_width"), dict) else {}
  opts = _opts_features(schedule.get("opts"))
  ctx_opts = _opts_features(normalized_ctx.get("opts"))
  features: dict[str, Any] = {
    "mechanism": mechanism,
    "row_kind": row.get("row_kind"),
    "role": row.get("role"),
    "format": row.get("format"),
    "prediction_stage": row.get("prediction_stage"),
    "model": row.get("model"),
    "tensor_block": _tensor_block(row.get("tensor")),
    "full_decode_supported": bool(normalized_ctx.get("full_decode_supported") or schedule.get("full_decode_supported")),
    "full_decode_ready": bool(normalized_ctx.get("full_decode_ready")),
    "source_ok": bool(normalized_ctx.get("source_ok")),
    "wide_loads": bool(normalized_ctx.get("wide_loads")),
    "schedule_name": schedule.get("name") or "none",
    "schedule_family": schedule.get("family") or "none",
    "schedule_codegen_mode": schedule.get("codegen_mode") or "none",
    "schedule_reduction_mode": schedule.get("reduction_mode") or "none",
    "schedule_semantic_object": schedule.get("semantic_object") or "none",
    "schedule_parts": int(schedule.get("parts") or normalized_ctx.get("parts") or 0),
    "schedule_row_tile": int(schedule.get("row_tile") or 0),
    "schedule_lane_width": int(schedule.get("lane_width") or 0),
    "schedule_group_unroll": int(schedule.get("group_unroll") or 0),
    "schedule_k_tile_blocks": int(schedule.get("k_tile_blocks") or 0),
    "schedule_requires_count": len(schedule.get("requires") or []) if isinstance(schedule.get("requires"), list) else 0,
    "context_parts": int(normalized_ctx.get("parts") or 0),
    "context_runs": int(normalized_ctx.get("runs") or 0),
    "load_width_inferred_words": _load_width_words(load_width.get("inferred") if load_width else None),
    "load_width_kernel_words": _load_width_words(load_width.get("kernel") if load_width else None),
    "load_width_report_words": _load_width_words(load_width.get("report") if load_width else None),
  }
  features.update({f"schedule_{key}": value for key, value in opts.items()})
  features.update({f"context_{key}": value for key, value in ctx_opts.items()})
  features.update(_shape_features(normalized_ctx.get("shape")))
  candidate_text = " ".join(str(v or "") for v in (row.get("candidate_id"), normalized_ctx.get("candidate"), normalized_ctx.get("candidate_id"), normalized_ctx.get("mode"), schedule.get("name"))).lower()
  for token in ("direct_out", "row_upcast", "reduce_unroll", "two_dim_local", "packed_load", "vector_load", "tile_custom", "qk_block_dot"):
    features[f"has_{token}"] = token in candidate_text
  return features

def _uop_features(static:dict[str, Any]) -> dict[str, Any]:
  lane_width = max(int(static.get("schedule_lane_width") or 0), int(static.get("load_width_inferred_words") or 0), int(static.get("load_width_kernel_words") or 0), 1)
  row_tile = max(int(static.get("schedule_row_tile") or 0), int(static.get("schedule_local0") or 0), 1)
  parts = max(int(static.get("schedule_parts") or 0), int(static.get("context_parts") or 0), 1)
  group_unroll = max(int(static.get("schedule_group_unroll") or 0), int(static.get("schedule_unroll") or 0), 1)
  vector_load_words = max(lane_width, int(static.get("load_width_report_words") or 0), 1)
  return {
    "uop_available": False,
    "estimated_global_load_words": row_tile * parts * vector_load_words,
    "estimated_global_store_words": row_tile * parts,
    "estimated_vector_load_width_words": vector_load_words,
    "estimated_scalar_loads": int(vector_load_words <= 1),
    "estimated_vector_loads": int(vector_load_words > 1),
    "estimated_local_axes": int(static.get("schedule_local_axes") or 0) + int(static.get("context_local_axes") or 0),
    "estimated_loop_axes": int(parts > 1) + int(group_unroll > 1) + int(row_tile > 1),
    "estimated_arithmetic_intensity_proxy": (row_tile * group_unroll) / max(row_tile * parts * vector_load_words, 1),
    "estimated_memory_pressure_proxy": row_tile * parts * vector_load_words,
  }

def _tensor_block(value:Any) -> int:
  m = re.search(r"blk[.-](\d+)", str(value or ""))
  return int(m.group(1)) if m else 0

def _profile_features(row:dict[str, Any]) -> dict[str, Any]|None:
  stage = str(row.get("prediction_stage") or "")
  if stage not in ("after_policy_before_full_decode", "after_microbench_before_full_decode"): return None
  return {
    "microbench_summary_available": False,
    "policy_path_available": bool((row.get("pre_result_context") or {}).get("policy")),
  }

def normalize_row(row:dict[str, Any]) -> dict[str, Any]:
  out = copy.deepcopy(row)
  old_mechanism = str(out.get("mechanism", "unknown"))
  normalized_ctx = _normalize_pre_result_context(out.get("pre_result_context") or {})
  out["pre_result_context"] = normalized_ctx
  mechanism = _norm_mechanism(out)
  out["schema_version"] = "kernel_triage_v1"
  out["mechanism_v0"] = old_mechanism
  out["mechanism"] = mechanism
  static = _static_features(out, normalized_ctx, mechanism)
  out["candidate_record"] = {
    "schema_version": "candidate_outcome_v1",
    "candidate_id": out["candidate_id"],
    "prediction_stage": out["prediction_stage"],
    "mechanism": mechanism,
    "static_features": static,
    "uop_features": _uop_features(static),
    "profile_features": _profile_features(out),
    "frozen_before_outcome": True,
    "outcome": {
      "label": out["label"],
      "reason": out["reason"],
      "retry": out["retry"],
      "source_files": out["source_files"],
    },
  }
  return out

def build_examples(repo:pathlib.Path) -> list[dict[str, Any]]:
  return [normalize_row(row) for row in v0.build_examples(repo)]

def validate_examples(rows:list[dict[str, Any]]) -> dict[str, Any]:
  base = v0.validate_examples(rows)
  errors = []
  for row in rows:
    record = row.get("candidate_record")
    if row.get("schema_version") != "kernel_triage_v1": errors.append(f"{row.get('id')}: missing v1 schema_version")
    if not isinstance(record, dict): errors.append(f"{row.get('id')}: missing candidate_record")
    else:
      if record.get("outcome", {}).get("label") != row.get("label"): errors.append(f"{row.get('id')}: outcome label mismatch")
      if not record.get("frozen_before_outcome"): errors.append(f"{row.get('id')}: candidate record not frozen")
      for key in ("static_features", "uop_features"):
        if not isinstance(record.get(key), dict): errors.append(f"{row.get('id')}: missing {key}")
  if errors: raise ValueError("; ".join(errors[:8]))
  base["schema_version"] = "kernel_triage_v1"
  base["mechanisms"] = dict(sorted(Counter(row["mechanism"] for row in rows).items()))
  base["unknown_mechanism_rows"] = sum(row["mechanism"] == "unknown" for row in rows)
  base["mechanism_changes_from_v0"] = sum(row.get("mechanism_v0") != row.get("mechanism") for row in rows)
  return base

def _prompt(row:dict[str, Any]) -> dict[str, Any]:
  record = copy.deepcopy(row["candidate_record"])
  record.pop("outcome", None)
  context = {
    "candidate_id": row["candidate_id"],
    "row_kind": row["row_kind"],
    "model": row["model"],
    "tensor": row["tensor"],
    "role": row["role"],
    "format": row["format"],
    "mechanism": row["mechanism"],
    "prediction_stage": row["prediction_stage"],
    "candidate_record": record,
  }
  question = (
    "/no_think\n"
    "Return only compact JSON with exactly these keys: label, reason, retry. "
    "Do not include analysis, markdown, or extra text. "
    f"Allowed labels: {', '.join(v0.LABELS)}. Allowed reasons: {', '.join(v0.REASONS)}. "
    "Decide from the frozen pre-result kernel candidate record. Context: "
    + json.dumps(context, sort_keys=True, separators=(",", ":"))
  )
  return {
    "id": row["id"],
    "prompt": question,
    "expected_json": {"label": row["label"], "reason": row["reason"], "retry": row["retry"]},
    "tags": ["qk_flywheel", "kernel_triage", "v1", row["split"], row["family"], row["mechanism"], row["label"]],
    "max_tokens": 64,
    "split": row["split"],
    "family": row["family"],
    "mechanism": row["mechanism"],
  }

def _jsonl(path:pathlib.Path, rows:list[dict[str, Any]]) -> None:
  with path.open("w") as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")

def _summary(rows:list[dict[str, Any]], prompts:list[dict[str, Any]]) -> dict[str, Any]:
  integrity = validate_examples(rows)
  train_rows = [row for row in rows if row["split"] == "train"]
  holdout_rows = [row for row in rows if row["split"] == "holdout"]
  return {
    "kind": "qk_flywheel_kernel_triage_dataset_v1",
    "rows": len(rows),
    "prompts": len(prompts),
    "split_policy": "family_split_v0_preserved",
    "train_rows": len(train_rows),
    "holdout_rows": len(holdout_rows),
    "feature_schema": "candidate_outcome_v1",
    "integrity": integrity,
    "files": {
      "examples": "examples.jsonl",
      "prompts": "prompts.jsonl",
      "train_prompts": "prompts-train.jsonl",
      "holdout_prompts": "prompts-holdout.jsonl",
      "summary": "summary.json",
      "readme": "README.md",
    },
  }

def _readme(summary:dict[str, Any]) -> str:
  lines = [
    "# AMD Decode Flywheel Kernel Triage Dataset v1",
    "",
    "This Phase 3D artifact preserves the v0 family split while adding normalized",
    "mechanisms and a frozen candidate-outcome feature schema for future cost",
    "model training.",
    "",
    f"- rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- feature schema: `{summary['feature_schema']}`",
    f"- unknown mechanism rows: `{summary['integrity']['unknown_mechanism_rows']}`",
    f"- mechanism changes from v0: `{summary['integrity']['mechanism_changes_from_v0']}`",
    "",
    "## Mechanisms",
    "",
    "| mechanism | rows |",
    "|---|---:|",
  ]
  for mech, count in summary["integrity"]["mechanisms"].items():
    lines.append(f"| `{mech}` | {count} |")
  lines.append("")
  return "\n".join(lines)

def write_dataset(repo:pathlib.Path, out:pathlib.Path) -> dict[str, Any]:
  rows = build_examples(repo)
  prompts = [_prompt(row) for row in rows]
  summary = _summary(rows, prompts)
  out.mkdir(parents=True, exist_ok=True)
  _jsonl(out / "examples.jsonl", rows)
  _jsonl(out / "prompts.jsonl", prompts)
  _jsonl(out / "prompts-train.jsonl", [row for row in prompts if row["split"] == "train"])
  _jsonl(out / "prompts-holdout.jsonl", [row for row in prompts if row["split"] == "holdout"])
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_readme(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Build AMD decode flywheel kernel-history triage dataset v1")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()
  summary = write_dataset(args.repo, args.out)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
