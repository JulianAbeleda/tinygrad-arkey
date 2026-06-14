#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from collections import Counter, defaultdict
from typing import Any

from extra.qk_flywheel_cost_model import FORBIDDEN_FEATURE_SOURCES, extract_feature_map
from extra.qk_flywheel_dataset import LABELS, MECHANISMS

TARGET_MIN_LABEL_ROWS = 5
TARGET_MIN_HOLDOUT_MECHANISM_ROWS = 5
FORBIDDEN_FEATURE_NAME_TOKENS = ("label", "reason", "retry", "evidence", "gain", "status", "candidate_gbs", "current_gbs", "decision", "correctness")

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

def _counts(rows:list[dict[str, Any]], key:str) -> dict[str, int]:
  return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))

def _split(examples:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  train = [row for row in examples if row.get("split") == "train"]
  holdout = [row for row in examples if row.get("split") == "holdout"]
  if not train or not holdout: raise ValueError("examples must contain non-empty train and holdout splits")
  return train, holdout

def _categorical_values(maps:list[dict[str, Any]]) -> dict[str, set[str]]:
  values: dict[str, set[str]] = defaultdict(set)
  for fmap in maps:
    for key, value in fmap.items():
      if isinstance(value, str): values[key].add(value)
  return values

def _presence(maps:list[dict[str, Any]]) -> dict[str, int]:
  out: Counter[str] = Counter()
  for fmap in maps:
    out.update(fmap.keys())
  return dict(sorted(out.items()))

def _is_nonzero_number(value:Any) -> bool:
  return isinstance(value, (int, float)) and abs(float(value)) > 1e-9

def _has_group(fmap:dict[str, Any], prefix:str) -> bool:
  return any(key.startswith(prefix) and (isinstance(value, str) or _is_nonzero_number(value)) for key, value in fmap.items())

def _row_audit(row:dict[str, Any], fmap:dict[str, Any], train_families:set[str], train_mechanisms:set[str], train_labels:set[str]) -> dict[str, Any]:
  numeric_nonzero = sorted(key for key, value in fmap.items() if _is_nonzero_number(value))
  categorical = sorted(key for key, value in fmap.items() if isinstance(value, str))
  groups = {
    "schedule": _has_group(fmap, "schedule_"),
    "change": _has_group(fmap, "change_"),
    "shape": _has_group(fmap, "shape_"),
    "load_width": _has_group(fmap, "load_width_"),
    "context_opts": _has_group(fmap, "context_opts_"),
    "v1_static": _has_group(fmap, "v1_static_"),
    "v1_uop": _has_group(fmap, "v1_uop_"),
    "v1_profile": _has_group(fmap, "v1_profile_"),
    "analytical": _has_group(fmap, "ana_"),
  }
  reasons = []
  if row.get("prediction_stage") == "after_full_decode":
    reasons.append("post_full_decode_training_row")
  if not any(groups[name] for name in ("schedule", "change", "shape", "load_width", "context_opts", "v1_static", "v1_uop", "v1_profile")):
    reasons.append("no_structural_kernel_detail")
  if row.get("mechanism") == "unknown":
    reasons.append("unknown_mechanism")
  if row.get("split") == "holdout" and "family" in fmap and row.get("family") not in train_families:
    reasons.append("family_unseen_in_train")
  if row.get("split") == "holdout" and row.get("mechanism") not in train_mechanisms:
    reasons.append("mechanism_unseen_in_train")
  if row.get("split") == "holdout" and row.get("label") not in train_labels:
    reasons.append("label_unseen_in_train")
  return {
    "id": row["id"],
    "split": row["split"],
    "family": row["family"],
    "mechanism": row["mechanism"],
    "label": row["label"],
    "prediction_stage": row["prediction_stage"],
    "categorical_feature_count": len(categorical),
    "numeric_nonzero_feature_count": len(numeric_nonzero),
    "groups": groups,
    "weak": bool(reasons),
    "weak_reasons": reasons,
  }

def _categorical_coverage(train_maps:list[dict[str, Any]], holdout_maps:list[dict[str, Any]]) -> dict[str, Any]:
  train_values = _categorical_values(train_maps)
  holdout_values = _categorical_values(holdout_maps)
  coverage = {}
  for key in sorted(set(train_values) | set(holdout_values)):
    train_seen = train_values.get(key, set())
    holdout_seen = holdout_values.get(key, set())
    unseen = holdout_seen - train_seen
    coverage[key] = {
      "train_values": sorted(train_seen),
      "holdout_values": sorted(holdout_seen),
      "unseen_holdout_values": sorted(unseen),
      "unseen_holdout_value_count": len(unseen),
      "covered": not unseen,
    }
  return coverage

def _target_rows(train_counts:dict[str, int], holdout_counts:dict[str, int], target:int, universe:list[str]|None=None) -> dict[str, dict[str, int]]:
  keys = sorted((set(universe or []) | set(train_counts) | set(holdout_counts)))
  out = {}
  for key in keys:
    train = int(train_counts.get(key, 0))
    holdout = int(holdout_counts.get(key, 0))
    if holdout > 0 or train < target:
      out[key] = {"train_rows": train, "holdout_rows": holdout, "target_train_rows": target, "needed_train_rows": max(0, target - train)}
  return out

def _stage_viability(train:list[dict[str, Any]], holdout:list[dict[str, Any]]) -> dict[str, Any]:
  all_rows = train + holdout
  by_stage = {}
  for stage in sorted({str(row.get("prediction_stage", "unknown")) for row in all_rows}):
    rows = [row for row in all_rows if row.get("prediction_stage") == stage]
    by_stage[stage] = {
      "rows": len(rows),
      "train_rows": sum(row.get("split") == "train" for row in rows),
      "holdout_rows": sum(row.get("split") == "holdout" for row in rows),
      "labels": _counts(rows, "label"),
      "allowed_feature_scope": _allowed_scope(stage),
    }
  post_full_decode_train = [row["id"] for row in train if row.get("prediction_stage") == "after_full_decode"]
  return {
    "by_stage": by_stage,
    "post_full_decode_train_rows": len(post_full_decode_train),
    "post_full_decode_train_ids": post_full_decode_train,
    "note": "Rows at after_full_decode are useful historical outcomes but weak training signal for pre-outcome triage.",
  }

def _allowed_scope(stage:str) -> str:
  if stage in ("after_static_before_microbench", "after_compile_before_microbench"):
    return "static_compile_features_only"
  if stage in ("after_policy_before_full_decode", "after_microbench_before_full_decode"):
    return "static_compile_plus_frozen_microbench_summary"
  if stage == "after_full_decode":
    return "post_outcome_baseline_context_only"
  return "unknown"

def _recommendations(summary:dict[str, Any]) -> list[dict[str, Any]]:
  recs = []
  missing_labels = [label for label, row in summary["targets"]["labels"].items() if row["needed_train_rows"] > 0 and row["holdout_rows"] > 0]
  if missing_labels:
    recs.append({
      "priority": 1,
      "kind": "collect_label_coverage",
      "recommendation": "Add train rows for labels that appear in holdout but are absent or undercovered in train.",
      "targets": missing_labels,
    })
  unknown_mechanism = summary["targets"]["holdout_mechanisms"].get("unknown", {})
  if unknown_mechanism.get("holdout_rows", 0) > 0:
    recs.append({
      "priority": 2,
      "kind": "normalize_unknown_mechanisms",
      "recommendation": "Map unknown holdout mechanisms to first-class mechanism names before treating them as learnable classes.",
      "targets": {"unknown_holdout_rows": unknown_mechanism["holdout_rows"]},
    })
  missing_mechanisms = [mechanism for mechanism, row in summary["targets"]["holdout_mechanisms"].items() if mechanism != "unknown" and row["needed_train_rows"] > 0 and row["holdout_rows"] > 0]
  if missing_mechanisms:
    recs.append({
      "priority": 3,
      "kind": "collect_mechanism_coverage",
      "recommendation": "Add targeted candidates for holdout mechanisms with fewer than five train rows.",
      "targets": missing_mechanisms,
    })
  if summary["coverage"]["categorical"]["unseen_holdout_value_total"] > 0:
    recs.append({
      "priority": 4,
      "kind": "reduce_unseen_categorical_gap",
      "recommendation": "Add rows or normalize feature extraction so holdout families/mechanisms/schedule names are represented before model training.",
      "targets": summary["coverage"]["categorical"]["top_unseen_features"],
    })
  if summary["row_quality"]["weak_row_count"] > 0:
    recs.append({
      "priority": 5,
      "kind": "improve_feature_extraction",
      "recommendation": "Add first-class tinygrad/UOp/profile features for weak rows instead of relying on top-level labels and candidate names.",
      "targets": summary["row_quality"]["top_weak_reasons"],
    })
  return recs

def _markdown(summary:dict[str, Any]) -> str:
  phase = "Phase 3D" if "kernel-triage-v1" in str(summary.get("examples_path", "")) else "Phase 3C"
  description = (
    "extends the data and feature audit over the normalized v1 schema."
    if phase == "Phase 3D" else
    "scopes the data and feature gaps that blocked the learned cost-model triage result."
  )
  lines = [
    "# AMD Decode Flywheel Feature Coverage Audit",
    "",
    f"This {phase} artifact {description} It does not train a model.",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- unseen holdout categorical values: `{summary['coverage']['categorical']['unseen_holdout_value_total']}`",
    f"- weak rows: `{summary['row_quality']['weak_row_count']}`",
    f"- post-full-decode train rows: `{summary['stage_viability']['post_full_decode_train_rows']}`",
    "",
    "## Highest Priority Targets",
    "",
  ]
  for rec in summary["recommendations"]:
    lines.append(f"- P{rec['priority']} `{rec['kind']}`: {rec['recommendation']}")
  lines += [
    "",
    "## Label Targets",
    "",
    "| label | train | holdout | needed train rows |",
    "|---|---:|---:|---:|",
  ]
  for label, row in summary["targets"]["labels"].items():
    if row["holdout_rows"] or row["needed_train_rows"]:
      lines.append(f"| `{label}` | {row['train_rows']} | {row['holdout_rows']} | {row['needed_train_rows']} |")
  lines += [
    "",
    "## Holdout Mechanism Targets",
    "",
    "| mechanism | train | holdout | needed train rows |",
    "|---|---:|---:|---:|",
  ]
  for mechanism, row in summary["targets"]["holdout_mechanisms"].items():
    if row["holdout_rows"]:
      lines.append(f"| `{mechanism}` | {row['train_rows']} | {row['holdout_rows']} | {row['needed_train_rows']} |")
  lines += [
    "",
    "## Top Unseen Categorical Features",
    "",
    "| feature | unseen holdout values |",
    "|---|---|",
  ]
  for item in summary["coverage"]["categorical"]["top_unseen_features"]:
    lines.append(f"| `{item['feature']}` | `{', '.join(item['values'])}` |")
  lines += [
    "",
    "## Top Weak Reasons",
    "",
    "| reason | rows |",
    "|---|---:|",
  ]
  for reason, count in summary["row_quality"]["top_weak_reasons"].items():
    lines.append(f"| `{reason}` | {count} |")
  lines.append("")
  return "\n".join(lines)

def run_feature_audit(examples_path:pathlib.Path, out:pathlib.Path) -> dict[str, Any]:
  examples = _read_jsonl(examples_path)
  train, holdout = _split(examples)
  train_maps = [extract_feature_map(row) for row in train]
  holdout_maps = [extract_feature_map(row) for row in holdout]
  train_families = {row["family"] for row in train}
  train_mechanisms = {row["mechanism"] for row in train}
  train_labels = {row["label"] for row in train}
  row_audits = [_row_audit(row, fmap, train_families, train_mechanisms, train_labels) for row, fmap in zip(train + holdout, train_maps + holdout_maps)]
  weak_reasons = Counter(reason for audit in row_audits for reason in audit["weak_reasons"])
  categorical = _categorical_coverage(train_maps, holdout_maps)
  unseen_features = [
    {"feature": key, "values": row["unseen_holdout_values"], "count": row["unseen_holdout_value_count"]}
    for key, row in categorical.items() if row["unseen_holdout_values"]
  ]
  unseen_features = sorted(unseen_features, key=lambda row: (-row["count"], row["feature"]))
  forbidden_names = sorted({key for fmap in train_maps + holdout_maps for key in fmap if any(token in key for token in FORBIDDEN_FEATURE_NAME_TOKENS)})
  mechanism_targets = _target_rows(_counts(train, "mechanism"), _counts(holdout, "mechanism"), TARGET_MIN_HOLDOUT_MECHANISM_ROWS, list(MECHANISMS))
  if "unknown" in mechanism_targets:
    mechanism_targets["unknown"]["target_train_rows"] = 0
    mechanism_targets["unknown"]["needed_train_rows"] = 0
  summary = {
    "kind": "qk_flywheel_feature_coverage_audit",
    "examples_path": str(examples_path),
    "examples": len(examples),
    "train_rows": len(train),
    "holdout_rows": len(holdout),
    "conclusion": "needs_data_and_feature_expansion",
    "split_counts": {"train": len(train), "holdout": len(holdout)},
    "labels": {"train": _counts(train, "label"), "holdout": _counts(holdout, "label")},
    "mechanisms": {"train": _counts(train, "mechanism"), "holdout": _counts(holdout, "mechanism")},
    "families": {"train": _counts(train, "family"), "holdout": _counts(holdout, "family")},
    "prediction_stages": {"train": _counts(train, "prediction_stage"), "holdout": _counts(holdout, "prediction_stage")},
    "coverage": {
      "categorical": {
        "by_feature": categorical,
        "unseen_holdout_value_total": sum(row["unseen_holdout_value_count"] for row in categorical.values()),
        "top_unseen_features": unseen_features[:12],
      },
      "feature_presence": {
        "train": _presence(train_maps),
        "holdout": _presence(holdout_maps),
      },
    },
    "targets": {
      "labels": _target_rows(_counts(train, "label"), _counts(holdout, "label"), TARGET_MIN_LABEL_ROWS, list(LABELS)),
      "holdout_mechanisms": mechanism_targets,
    },
    "row_quality": {
      "weak_row_count": sum(audit["weak"] for audit in row_audits),
      "top_weak_reasons": dict(sorted(weak_reasons.items(), key=lambda item: (-item[1], item[0]))),
    },
    "stage_viability": _stage_viability(train, holdout),
    "leakage_audit": {
      "excluded_feature_sources": list(FORBIDDEN_FEATURE_SOURCES),
      "feature_names_with_forbidden_tokens": forbidden_names,
      "target_or_result_fields_used": bool(forbidden_names),
    },
  }
  summary["recommendations"] = _recommendations(summary)
  out.mkdir(parents=True, exist_ok=True)
  _jsonl(out / "row-audit.jsonl", row_audits)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_markdown(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Audit feature/data coverage for AMD decode flywheel cost-model triage")
  parser.add_argument("--examples", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()
  summary = run_feature_audit(args.examples, args.out)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
