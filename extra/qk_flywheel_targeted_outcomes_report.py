#!/usr/bin/env python3
"""Summary and README rendering for the Phase 3F targeted-outcomes batch.

Extracted verbatim from qk_flywheel_targeted_outcomes.py as a behavior-preserving
move (NFC). Owns the targeted/plus summary dicts and their markdown READMEs.
TARGETED_FAMILY / TARGETED_FAMILY_ORDER live here as the single source of truth
because the summaries embed them; the row-builder module imports them from here.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from extra import qk_flywheel_dataset_v1 as v1

TARGETED_FAMILY = "targeted_outcomes_v1"
TARGETED_FAMILY_ORDER = 11

def _targeted_summary(rows:list[dict[str, Any]], excluded:list[dict[str, Any]], *, base_count:int, plus_count:int) -> dict[str, Any]:
  real_rows = [row for row in rows if (row.get("feature_extraction") or {}).get("real_uop_available")]
  return {
    "kind": "qk_flywheel_targeted_outcomes_v1",
    "phase": "Phase 3F",
    "conclusion": "partial_real_outcome_batch_cost_model_rerun_still_gated",
    "rows": len(rows),
    "base_rows": base_count,
    "plus_rows": plus_count,
    "split": "train",
    "family": TARGETED_FAMILY,
    "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
    "mechanisms": dict(sorted(Counter(row["mechanism"] for row in rows).items())),
    "prediction_stages": dict(sorted(Counter(row["prediction_stage"] for row in rows).items())),
    "source_files": dict(Counter(src for row in rows for src in row.get("source_files", [])).most_common()),
    "real_feature_rows": len(real_rows),
    "excluded_sources": excluded,
    "rules": [
      "uses only committed real probe/compile/source diagnostic artifacts",
      "does not duplicate any existing v1-featured row id",
      "does not move family-split holdout rows into train",
      "does not use design-only contracts as train labels",
      "does not authorize a Phase 3B cost-model rerun as a decision point unless the plus audit clears the coverage gate",
    ],
  }

def _plus_summary(rows:list[dict[str, Any]], targeted:list[dict[str, Any]], prompts:list[dict[str, Any]], targeted_summary:dict[str, Any]) -> dict[str, Any]:
  integrity = v1.validate_examples(rows)
  train = [row for row in rows if row["split"] == "train"]
  holdout = [row for row in rows if row["split"] == "holdout"]
  real_rows = [row for row in rows if (row.get("feature_extraction") or {}).get("real_uop_available")]
  real_by_split = Counter(row["split"] for row in real_rows)
  real_by_mech = Counter(row["mechanism"] for row in real_rows)
  return {
    "kind": "qk_flywheel_kernel_triage_dataset_v1_featured_plus",
    "phase": "Phase 3F",
    "rows": len(rows),
    "prompts": len(prompts),
    "split_policy": "family_split_v0_preserved_plus_post_phase3e_train_batch",
    "train_rows": len(train),
    "holdout_rows": len(holdout),
    "feature_schema": "candidate_outcome_v1_featured",
    "targeted_rows_added": len(targeted),
    "targeted_family": TARGETED_FAMILY,
    "targeted_summary": {
      "labels": targeted_summary["labels"],
      "mechanisms": targeted_summary["mechanisms"],
      "excluded_sources": targeted_summary["excluded_sources"],
    },
    "integrity": integrity,
    "real_feature_coverage": {
      "uop_available_rows": len(real_rows),
      "uop_available_train_rows": real_by_split.get("train", 0),
      "uop_available_holdout_rows": real_by_split.get("holdout", 0),
      "uop_available_by_mechanism": dict(sorted(real_by_mech.items())),
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

def _targeted_readme(summary:dict[str, Any]) -> str:
  lines = [
    "# AMD Decode Flywheel Targeted Outcomes v1",
    "",
    "This Phase 3F artifact converts unused committed real probe/source",
    "diagnostics into a small post-Phase-3E train batch. It is deliberately",
    "partial: no holdout row is moved into train, no synthetic outcome is added,",
    "and design-only contracts remain excluded.",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- targeted train rows: `{summary['rows']}`",
    f"- base rows: `{summary['base_rows']}`",
    f"- plus rows: `{summary['plus_rows']}`",
    f"- real feature rows: `{summary['real_feature_rows']}`",
    "",
    "## Mechanisms",
    "",
    "| mechanism | rows |",
    "|---|---:|",
  ]
  for mechanism, count in summary["mechanisms"].items():
    lines.append(f"| `{mechanism}` | {count} |")
  lines += ["", "## Labels", "", "| label | rows |", "|---|---:|"]
  for label, count in summary["labels"].items():
    lines.append(f"| `{label}` | {count} |")
  lines += ["", "## Excluded Sources", "", "| source | rows | reason |", "|---|---:|---|"]
  for item in summary["excluded_sources"]:
    lines.append(f"| `{item['source_file']}` | {item['rows']} | {item['reason']} |")
  lines += ["", "## Rules", ""]
  for rule in summary["rules"]:
    lines.append(f"- {rule}")
  lines.append("")
  return "\n".join(lines)

def _plus_readme(summary:dict[str, Any]) -> str:
  lines = [
    "# AMD Decode Flywheel Kernel Triage Dataset v1 Featured Plus",
    "",
    "This Phase 3F dataset appends the targeted-outcomes train batch to the",
    "Phase 3E featured dataset while preserving the original family-split",
    "holdout. It is an intermediate coverage artifact, not a cost-model win.",
    "",
    f"- rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- targeted rows added: `{summary['targeted_rows_added']}`",
    f"- split policy: `{summary['split_policy']}`",
    f"- real UOp/source rows: `{summary['real_feature_coverage']['uop_available_rows']}`",
    "",
    "## Targeted Mechanisms",
    "",
    "| mechanism | rows |",
    "|---|---:|",
  ]
  for mechanism, count in summary["targeted_summary"]["mechanisms"].items():
    lines.append(f"| `{mechanism}` | {count} |")
  lines += ["", "## Targeted Labels", "", "| label | rows |", "|---|---:|"]
  for label, count in summary["targeted_summary"]["labels"].items():
    lines.append(f"| `{label}` | {count} |")
  lines.append("")
  return "\n".join(lines)
