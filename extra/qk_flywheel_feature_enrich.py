#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, pathlib, re
from collections import Counter
from typing import Any

from extra import qk_flywheel_dataset_v1 as v1

FEATURE_SCHEMA = "candidate_outcome_v1_featured"

def _read_json(path:pathlib.Path) -> dict[str, Any] | None:
  if not path.exists(): return None
  data = json.loads(path.read_text())
  return data if isinstance(data, dict) else None

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

def _repo_path(repo:pathlib.Path, value:Any) -> pathlib.Path:
  path = pathlib.Path(str(value))
  return path if path.is_absolute() else repo / path

def _portable(repo:pathlib.Path, path:pathlib.Path) -> str:
  try:
    return str(path.resolve().relative_to(repo.resolve()))
  except ValueError:
    return str(path)

def _load_width_words(value:Any) -> int:
  text = str(value or "").lower()
  m = re.search(r"x(\d+)", text)
  if m: return int(m.group(1))
  m = re.search(r"(\d+)", text)
  return int(m.group(1)) if m else 0

def _mode_candidates(row:dict[str, Any]) -> list[str]:
  ctx = row.get("pre_result_context") or {}
  record = row.get("candidate_record") or {}
  static = record.get("static_features") if isinstance(record, dict) else {}
  raw = [
    ctx.get("mode") if isinstance(ctx, dict) else None,
    (static or {}).get("schedule_name") if isinstance(static, dict) else None,
    row.get("mechanism"),
  ]
  mech = str(row.get("mechanism") or "")
  if mech == "packed_word_lane_unroll": raw += ["packed_load", "packed_load_partial"]
  if mech in ("vector_load", "wide_load_only"): raw += ["vector_load", "unknown"]
  if mech == "tile_custom": raw += ["tile_custom", "tile_custom_partial", "custom_uint4", "packed_tile_custom_q4_dot"]
  if mech == "qk_block_dot": raw += ["qk_block_dot"]
  if mech == "direct_output": raw += ["direct_out"]
  out = []
  for item in raw:
    text = str(item or "").strip()
    if text and text not in out: out.append(text)
  return out

def _select_named(rows:dict[str, Any], row:dict[str, Any]) -> dict[str, Any] | None:
  for mode in _mode_candidates(row):
    if mode in rows and isinstance(rows[mode], dict): return rows[mode]
  return None

def _select_load_width_row(report:dict[str, Any], row:dict[str, Any]) -> dict[str, Any] | None:
  rows = [r for r in report.get("rows", []) if isinstance(r, dict)]
  if not rows: return None
  modes = set(_mode_candidates(row))
  for item in rows:
    if str(item.get("mode") or "") in modes: return item
  ctx = row.get("pre_result_context") or {}
  if isinstance(ctx, dict) and ctx.get("mode"): return None
  non_baseline = [item for item in rows if "baseline" not in str(item.get("mode") or "")]
  if len(non_baseline) == 1: return non_baseline[0]
  return None

def _load_width_report_paths(source:pathlib.Path) -> list[pathlib.Path]:
  parent = source.parent
  candidates = [
    parent / "load-width" / "report.json",
    parent / "source" / "load-width-report.json",
  ]
  if source.name in ("analysis.json", "diagnostic.json", "verdict.json", "microbench.json"):
    candidates += [
      parent / "load-width" / "report.json",
      parent / "source" / "load-width-report.json",
    ]
  return [path for idx, path in enumerate(candidates) if path not in candidates[:idx]]

def _add_load_width_features(features:dict[str, Any], report:dict[str, Any], item:dict[str, Any]) -> None:
  counts = item.get("pattern_counts") or {}
  if not isinstance(counts, dict): counts = {}
  kernels = item.get("kernels") if isinstance(item.get("kernels"), list) else []
  features["load_width_report_available"] = True
  features["source_kernel_count"] = len(kernels)
  features["source_load_width_words"] = _load_width_words(item.get("load_width_inferred"))
  features["source_contains_packed_dot"] = bool(item.get("contains_packed_dot"))
  summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
  features["source_has_vector_load_shape"] = bool(summary.get("has_vector_load_evidence"))
  features["source_has_packed_load_kernel"] = bool(summary.get("has_packed_load_kernel"))
  for key in ("uint_or_u32", "ushort_or_u16", "uchar_or_u8", "ulong_or_u64", "vector_u32x2", "vector_u32x4", "amd_vdot4"):
    value = counts.get(key, 0)
    if isinstance(value, (int, float)): features[f"source_pattern_{key}"] = int(value)
  features["source_has_vector_type"] = bool(features.get("source_pattern_vector_u32x2", 0) or features.get("source_pattern_vector_u32x4", 0) or features["source_load_width_words"] > 1)

def _add_compile_features(features:dict[str, Any], mode:dict[str, Any]) -> None:
  features["compile_report_available"] = True
  for key in ("instruction_count", "memory_instruction_count", "source_lines", "workgroup_size", "global_load_b128", "global_load_b64", "global_load_b32"):
    value = mode.get(key)
    if isinstance(value, (int, float)): features[f"compile_{key}"] = int(value)
  features["compile_source_has_vector_type"] = bool(mode.get("source_has_vector_type"))
  features["compile_source_has_tg_uint4_load"] = bool(mode.get("source_has_tg_uint4_load"))
  local_counts = mode.get("local_counts") if isinstance(mode.get("local_counts"), dict) else {}
  group_counts = mode.get("group_counts") if isinstance(mode.get("group_counts"), dict) else {}
  features["compile_local_lane_count"] = sum(int(v) for v in local_counts.values() if isinstance(v, (int, float)))
  features["compile_group_axis_count"] = sum(int(v) for v in group_counts.values() if isinstance(v, (int, float)))
  inst = mode.get("instruction_counts") if isinstance(mode.get("instruction_counts"), dict) else {}
  mem = mode.get("memory_instruction_counts") if isinstance(mode.get("memory_instruction_counts"), dict) else {}
  for key in ("global_load_b128", "global_load_b64", "global_load_b32", "global_store_b32", "v_fma_f32", "v_fma_mix_f32", "v_bfe_u32", "v_cvt_f32_ubyte0_e32"):
    value = inst.get(key, mem.get(key, 0))
    if isinstance(value, (int, float)): features[f"compile_inst_{key}"] = int(value)

def _feature_sources_for(source:pathlib.Path, data:dict[str, Any], repo:pathlib.Path) -> list[pathlib.Path]:
  out = []
  for path in _load_width_report_paths(source):
    if path.exists(): out.append(path)
  raw = data.get("source_compile_gate")
  if raw:
    path = _repo_path(repo, raw)
    if path.exists(): out.append(path)
  return out

def _extract_from_json(row:dict[str, Any], repo:pathlib.Path, source:pathlib.Path, data:dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
  features: dict[str, Any] = {}
  used: list[str] = []

  modes = data.get("modes")
  if isinstance(modes, dict):
    selected = _select_named(modes, row)
    if selected:
      _add_compile_features(features, selected)
      used.append(_portable(repo, source))

  for extra_path in _feature_sources_for(source, data, repo):
    extra = _read_json(extra_path)
    if not extra: continue
    if extra.get("kind") == "qk_load_width_report":
      selected = _select_load_width_row(extra, row)
      if selected:
        _add_load_width_features(features, extra, selected)
        used.append(_portable(repo, extra_path))
    elif isinstance(extra.get("modes"), dict):
      selected = _select_named(extra["modes"], row)
      if selected:
        _add_compile_features(features, selected)
        used.append(_portable(repo, extra_path))

  if data.get("kind") == "qk_load_width_report":
    selected = _select_load_width_row(data, row)
    if selected:
      _add_load_width_features(features, data, selected)
      used.append(_portable(repo, source))

  return features, sorted(set(used))

def extracted_uop_features(row:dict[str, Any], repo:pathlib.Path) -> tuple[dict[str, Any], list[str]]:
  features: dict[str, Any] = {}
  sources: list[str] = []
  for raw in row.get("source_files") or []:
    source = _repo_path(repo, raw)
    data = _read_json(source)
    if not data: continue
    new_features, used = _extract_from_json(row, repo, source, data)
    features.update(new_features)
    sources += used
  if features:
    features["uop_available"] = True
    features["real_feature_source_count"] = len(set(sources))
  else:
    features["uop_available"] = False
    features["real_feature_source_count"] = 0
  return features, sorted(set(sources))

def enrich_row(row:dict[str, Any], repo:pathlib.Path) -> dict[str, Any]:
  out = copy.deepcopy(row)
  record = out.get("candidate_record")
  if not isinstance(record, dict): raise ValueError(f"{out.get('id')}: missing candidate_record")
  uop = copy.deepcopy(record.get("uop_features") or {})
  extracted, sources = extracted_uop_features(out, repo)
  if extracted.get("uop_available"):
    uop.update(extracted)
  else:
    uop["uop_available"] = bool(uop.get("uop_available", False))
    uop["real_feature_source_count"] = 0
  record["schema_version"] = FEATURE_SCHEMA
  record["uop_features"] = uop
  out["candidate_record"] = record
  out["feature_schema"] = FEATURE_SCHEMA
  out["feature_extraction"] = {
    "real_uop_available": bool(uop.get("uop_available")),
    "real_feature_sources": sources,
    "real_feature_source_count": len(sources),
  }
  return out

def _summary(rows:list[dict[str, Any]], prompts:list[dict[str, Any]]) -> dict[str, Any]:
  integrity = v1.validate_examples(rows)
  train = [row for row in rows if row["split"] == "train"]
  holdout = [row for row in rows if row["split"] == "holdout"]
  real_rows = [row for row in rows if row.get("feature_extraction", {}).get("real_uop_available")]
  real_by_split = Counter(row["split"] for row in real_rows)
  real_by_mech = Counter(row["mechanism"] for row in real_rows)
  source_counts = Counter(src for row in rows for src in row.get("feature_extraction", {}).get("real_feature_sources", []))
  return {
    "kind": "qk_flywheel_kernel_triage_dataset_v1_featured",
    "rows": len(rows),
    "prompts": len(prompts),
    "split_policy": "family_split_v0_preserved",
    "train_rows": len(train),
    "holdout_rows": len(holdout),
    "feature_schema": FEATURE_SCHEMA,
    "integrity": integrity,
    "real_feature_coverage": {
      "uop_available_rows": len(real_rows),
      "uop_available_train_rows": real_by_split.get("train", 0),
      "uop_available_holdout_rows": real_by_split.get("holdout", 0),
      "uop_available_by_mechanism": dict(sorted(real_by_mech.items())),
      "feature_source_count": len(source_counts),
      "top_feature_sources": dict(source_counts.most_common(12)),
    },
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
  cov = summary["real_feature_coverage"]
  lines = [
    "# AMD Decode Flywheel Kernel Triage Dataset v1 Featured",
    "",
    "This Phase 3E artifact preserves the Phase 3D rows and split while adding",
    "real source/compile evidence where committed artifacts expose it. It does",
    "not add synthetic outcomes or move holdout rows into train.",
    "",
    f"- rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- feature schema: `{summary['feature_schema']}`",
    f"- real UOp/source rows: `{cov['uop_available_rows']}`",
    f"- real UOp/source train rows: `{cov['uop_available_train_rows']}`",
    f"- real UOp/source holdout rows: `{cov['uop_available_holdout_rows']}`",
    "",
    "## Real Feature Coverage By Mechanism",
    "",
    "| mechanism | rows |",
    "|---|---:|",
  ]
  for mech, count in cov["uop_available_by_mechanism"].items():
    lines.append(f"| `{mech}` | {count} |")
  lines += ["", "## Top Feature Sources", "", "| source | rows |", "|---|---:|"]
  for source, count in cov["top_feature_sources"].items():
    lines.append(f"| `{source}` | {count} |")
  lines.append("")
  return "\n".join(lines)

def write_featured_dataset(repo:pathlib.Path, out:pathlib.Path, examples_path:pathlib.Path|None=None) -> dict[str, Any]:
  repo = repo.resolve()
  rows = _read_jsonl(examples_path) if examples_path else v1.build_examples(repo)
  enriched = [enrich_row(row, repo) for row in rows]
  prompts = [v1._prompt(row) for row in enriched]
  summary = _summary(enriched, prompts)
  out.mkdir(parents=True, exist_ok=True)
  _jsonl(out / "examples.jsonl", enriched)
  _jsonl(out / "prompts.jsonl", prompts)
  _jsonl(out / "prompts-train.jsonl", [row for row in prompts if row["split"] == "train"])
  _jsonl(out / "prompts-holdout.jsonl", [row for row in prompts if row["split"] == "holdout"])
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  (out / "README.md").write_text(_readme(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Add real source/compile features to AMD decode flywheel v1 dataset")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  parser.add_argument("--examples", type=pathlib.Path)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()
  summary = write_featured_dataset(args.repo, args.out, args.examples)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
