#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.llm_eval_common import load_json as _load_json
from extra.qk_experiment_matrix import _fmt

BUCKET_ORDER = (
  "q4k_primitive_gemv",
  "q6k_primitive_gemv",
  "q4k_primitive_reduction",
  "fallback_quant_fused",
  "attention_misc",
  "norm_sampling_misc",
  "copy",
  "other_amd",
  "residual_overhead",
)

def _model_from_dir(path:pathlib.Path) -> str:
  name = path.name.lower()
  if name in ("8b", "14b", "32b"): return name.upper()
  raise ValueError(f"{path}: expected model directory named 8b, 14b, or 32b")

def _mode_map(report:list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  out = {}
  for row in report:
    mode = row.get("mode")
    if not isinstance(mode, str): raise ValueError("profile row missing mode")
    out[mode] = row
  return out

def _dominant_buckets(summary:dict[str, Any], *, pct_key:str, include_residual:bool=True) -> list[dict[str, Any]]:
  buckets = summary.get("bucket_ms_tok") or {}
  pcts = summary.get(pct_key) or {}
  rows = []
  for bucket in BUCKET_ORDER:
    if bucket == "residual_overhead" and not include_residual: continue
    rows.append({"bucket": bucket, "ms_tok": buckets.get(bucket, 0.0), "pct": pcts.get(bucket, 0.0)})
  return sorted(rows, key=lambda row: row["ms_tok"], reverse=True)

def summarize_model(path:pathlib.Path) -> dict[str, Any]:
  model = _model_from_dir(path)
  profile_path = path / "profile-report.json"
  if not profile_path.exists():
    return {
      "model": model,
      "path": str(path),
      "status": "profile_missing",
      "reason": f"{profile_path} is missing; run DEBUG=2 generated/explicit profile before optimizing this model.",
    }
  report = _load_json(profile_path)
  if not isinstance(report, list): raise ValueError(f"{profile_path}: expected list")
  modes = _mode_map(report)
  generated_batched = modes.get("QK_GENERATED_POLICY batched")
  generated_named = modes.get("QK_GENERATED_POLICY named")
  explicit_batched = modes.get("Q4K+Q6K_PRIMITIVE=1 batched")
  explicit_named = modes.get("Q4K+Q6K_PRIMITIVE=1 named")
  if generated_batched is None or generated_named is None:
    raise ValueError(f"{profile_path}: expected generated batched and generated named rows")

  gb = generated_batched["summary"]
  gn = generated_named["summary"]
  gn_buckets = gn.get("bucket_ms_tok") or {}
  top_named = _dominant_buckets(gn, pct_key="bucket_pct_amd", include_residual=False)
  top_batched = _dominant_buckets(gb, pct_key="bucket_pct_wall")
  return {
    "model": model,
    "path": str(path),
    "profile_report": str(profile_path),
    "status": "profiled",
    "throughput_truth": {
      "mode": generated_batched["mode"],
      "tok_s": gb["tok_s"],
      "wall_ms_tok": gb["wall_ms_tok"],
      "amd_kernel_ms_tok": gb["amd_kernel_ms_tok"],
      "residual_ms_tok": gb["residual_ms_tok"],
      "top_batched_wall_buckets": top_batched[:5],
    },
    "attribution_named": {
      "mode": generated_named["mode"],
      "amd_kernel_ms_tok": gn["amd_kernel_ms_tok"],
      "wall_ms_tok": gn["wall_ms_tok"],
      "residual_ms_tok": gn["residual_ms_tok"],
      "dominant_amd_buckets": top_named[:8],
      "qk_gemv_ms_tok": gn_buckets.get("q4k_primitive_gemv", 0.0) + gn_buckets.get("q6k_primitive_gemv", 0.0),
      "qk_reduction_ms_tok": gn_buckets.get("q4k_primitive_reduction", 0.0),
      "fallback_quant_ms_tok": gn_buckets.get("fallback_quant_fused", 0.0),
    },
    "explicit_reference": {
      "batched_tok_s": None if explicit_batched is None else explicit_batched["summary"]["tok_s"],
      "named_amd_kernel_ms_tok": None if explicit_named is None else explicit_named["summary"]["amd_kernel_ms_tok"],
    },
    "next_decision": _next_decision(top_named),
  }

def _next_decision(top_named:list[dict[str, Any]]) -> str:
  dominant = next((row for row in top_named if row["bucket"] != "residual_overhead"), None)
  if dominant is None: return "needs_profile_review"
  bucket = dominant["bucket"]
  if bucket in ("q4k_primitive_gemv", "q6k_primitive_gemv"):
    return "qk_semantic_schedule_or_codegen"
  if bucket == "q4k_primitive_reduction":
    return "reduce_or_fuse_partial_reductions"
  if bucket == "fallback_quant_fused":
    return "extend_qk_coverage_or_replace_fallback"
  if bucket in ("attention_misc", "norm_sampling_misc"):
    return "pause_qk_and_profile_non_qk"
  return "inspect_other_amd_kernels"

def build_gap_profile(model_dirs:list[pathlib.Path]) -> dict[str, Any]:
  models = [summarize_model(path) for path in model_dirs]
  profiled = [row for row in models if row["status"] == "profiled"]
  missing = [row["model"] for row in models if row["status"] != "profiled"]
  return {
    "kind": "qk_gap_profile",
    "scope": "Ansor-transition bottleneck attribution over committed DEBUG=2 profile reports",
    "models": models,
    "summary": {
      "models": len(models),
      "profiled": len(profiled),
      "missing_profile": missing,
      "next_decisions": {row["model"]: row.get("next_decision") for row in models if row["status"] == "profiled"},
    },
  }

def gap_profile_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK Gap Profile",
    "",
    "Bottleneck attribution for the Ansor-transition loop. `batched` rows are",
    "the throughput truth; `named` rows are attribution-only because graph",
    "batching is disabled for readable kernel names.",
    "",
    "## Summary",
    "",
    f"- profiled models: `{report['summary']['profiled']}/{report['summary']['models']}`",
    f"- missing profiles: `{', '.join(report['summary']['missing_profile']) or 'none'}`",
    "",
    "| model | status | generated tok/s | batched wall ms/tok | named QK GEMV ms/tok | named reduction ms/tok | fallback ms/tok | next decision |",
    "|---|---|---:|---:|---:|---:|---:|---|",
  ]
  for row in report["models"]:
    if row["status"] != "profiled":
      lines.append(f"| `{row['model']}` | `{row['status']}` | n/a | n/a | n/a | n/a | n/a | `{row['reason']}` |")
      continue
    truth, attr = row["throughput_truth"], row["attribution_named"]
    lines.append(
      f"| `{row['model']}` | `profiled` | {_fmt(truth['tok_s'])} | {_fmt(truth['wall_ms_tok'])} | "
      f"{_fmt(attr['qk_gemv_ms_tok'])} | {_fmt(attr['qk_reduction_ms_tok'])} | "
      f"{_fmt(attr['fallback_quant_ms_tok'])} | `{row['next_decision']}` |"
    )
  lines += ["", "## Dominant Named AMD Buckets", ""]
  for row in report["models"]:
    if row["status"] != "profiled": continue
    lines += [f"### {row['model']}", "", "| bucket | ms/tok | % named AMD |", "|---|---:|---:|"]
    for bucket in row["attribution_named"]["dominant_amd_buckets"]:
      if bucket["ms_tok"] <= 0: continue
      lines.append(f"| `{bucket['bucket']}` | {_fmt(bucket['ms_tok'])} | {_fmt(bucket['pct'])} |")
    lines.append("")
  return "\n".join(lines)

def write_gap_profile(report:dict[str, Any], json_path:pathlib.Path, md_path:pathlib.Path) -> None:
  json_path.parent.mkdir(parents=True, exist_ok=True)
  md_path.parent.mkdir(parents=True, exist_ok=True)
  json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
  md_path.write_text(gap_profile_markdown(report))

def main() -> int:
  parser = argparse.ArgumentParser(description="Summarize current QK gap attribution from committed profile reports")
  parser.add_argument("model_dirs", nargs="+", type=pathlib.Path)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  args = parser.parse_args()
  report = build_gap_profile([p.expanduser() for p in args.model_dirs])
  write_gap_profile(report, args.json, args.md)
  print(gap_profile_markdown(report))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
