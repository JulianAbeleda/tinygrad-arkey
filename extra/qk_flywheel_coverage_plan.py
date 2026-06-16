#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

MECHANISM_RECIPES = {
  "packed_word_lane_unroll": {
    "batch": "packed-load lane-unroll microbench",
    "recipe": "Run packed_word_lane_unroll on additional Q4_K tensors/roles; require generated-source load-width report before timing.",
    "candidate_stage": "after_static_before_microbench",
  },
  "qk_block_dot": {
    "batch": "QK_BLOCK_DOT compile+dominant-shape microbench",
    "recipe": "Repeat the block-local semantic op compile gate on more dominant Q4_K tensors, then microbench only compile-shape passes.",
    "candidate_stage": "after_compile_before_microbench",
  },
  "reduce_unroll": {
    "batch": "semantic schedule v1",
    "recipe": "Run reduce_unroll schedule candidates on fresh tensors/models as a post-Phase-3E train batch.",
    "candidate_stage": "after_static_before_microbench",
  },
  "row_upcast": {
    "batch": "semantic schedule v1",
    "recipe": "Run row_upcast schedule candidates on fresh tensors/models as a post-Phase-3E train batch.",
    "candidate_stage": "after_static_before_microbench",
  },
  "two_dim_local": {
    "batch": "semantic schedule v1",
    "recipe": "Run two_dim_local schedule candidates on fresh tensors/models as a post-Phase-3E train batch.",
    "candidate_stage": "after_static_before_microbench",
  },
  "vector_load": {
    "batch": "vector-load construction probe",
    "recipe": "Collect construction outcomes for schedulable vector-load candidates, with source-shape evidence and no full decode unless gates pass.",
    "candidate_stage": "after_static_before_microbench",
  },
  "wide_load_only": {
    "batch": "three-way load diagnostic continuation",
    "recipe": "Run wide-load-only/vector-load controls on additional tensors to bound the branch without runtime integration.",
    "candidate_stage": "after_compile_before_microbench",
  },
}

LABEL_RECIPES = {
  "construction_blocked": "Natural outcome from failed construction/static candidates; record exact verifier/shape failure.",
  "diagnostic_only": "Compile/source/counter evidence rows that authorize or reject the next gate without being promotion candidates.",
  "raw_accept_unconfirmed": "Only record when a repeated microbench clears its predeclared bar but no full-decode confirmation exists yet; do not force this label.",
}

from extra.llm_eval_common import read_json_object as _read_json

def build_coverage_plan(audit:dict[str, Any], audit_path:pathlib.Path) -> dict[str, Any]:
  audit_text = str(audit_path)
  phase = "Phase 3F Plus" if "featured-plus" in audit_text else "Phase 3E"
  mechanism_targets = audit.get("targets", {}).get("holdout_mechanisms", {})
  label_targets = audit.get("targets", {}).get("labels", {})
  missing_mechanisms = {
    mechanism: row for mechanism, row in sorted(mechanism_targets.items())
    if mechanism != "unknown" and int(row.get("holdout_rows", 0)) > 0 and int(row.get("needed_train_rows", 0)) > 0
  }
  missing_labels = {
    label: row for label, row in sorted(label_targets.items())
    if int(row.get("holdout_rows", 0)) > 0 and int(row.get("needed_train_rows", 0)) > 0
  }
  mechanism_batches = []
  for mechanism, target in missing_mechanisms.items():
    recipe = MECHANISM_RECIPES.get(mechanism, {
      "batch": "targeted candidate batch",
      "recipe": "Collect real candidate outcomes for this mechanism before rerunning the cost model.",
      "candidate_stage": "after_static_before_microbench",
    })
    mechanism_batches.append({
      "mechanism": mechanism,
      "needed_train_rows": int(target["needed_train_rows"]),
      "current_train_rows": int(target["train_rows"]),
      "holdout_rows": int(target["holdout_rows"]),
      **recipe,
    })
  label_batches = []
  for label, target in missing_labels.items():
    label_batches.append({
      "label": label,
      "needed_train_rows": int(target["needed_train_rows"]),
      "current_train_rows": int(target["train_rows"]),
      "holdout_rows": int(target["holdout_rows"]),
      "recipe": LABEL_RECIPES.get(label, "Collect natural outcomes for this label; do not synthesize labels."),
    })
  min_mechanism_rows = sum(row["needed_train_rows"] for row in mechanism_batches)
  min_label_rows = sum(row["needed_train_rows"] for row in label_batches)
  unseen_categorical = int(((audit.get("coverage") or {}).get("categorical") or {}).get("unseen_holdout_value_total", 0))
  # The rerun gate is data-driven: it clears once no holdout mechanism or label is short on train
  # coverage and no holdout categorical value is still unseen in train. Earlier plans hardcoded this
  # to False because the featured dataset added no new outcomes; the Phase 3G batch closes those gaps.
  rerun_blockers = []
  if mechanism_batches: rerun_blockers.append("missing mechanism train coverage")
  if label_batches: rerun_blockers.append("missing or thin label train coverage")
  if unseen_categorical > 0: rerun_blockers.append("unseen holdout categorical value in train")
  rerun_phase3b_allowed = not rerun_blockers
  return {
    "kind": "qk_flywheel_phase3f_plus_coverage_plan" if phase == "Phase 3F Plus" else "qk_flywheel_phase3e_coverage_plan",
    "phase": phase,
    "source_audit": str(audit_path),
    "conclusion": "coverage_gate_cleared_cost_model_rerun_allowed" if rerun_phase3b_allowed else "collect_targeted_outcomes_before_cost_model_rerun",
    "rerun_phase3b_allowed": rerun_phase3b_allowed,
    "rerun_blockers": rerun_blockers,
    "unseen_holdout_categorical_values": unseen_categorical,
    "minimum_new_mechanism_rows": min_mechanism_rows,
    "minimum_new_label_rows": min_label_rows,
    "mechanism_batches": mechanism_batches,
    "label_batches": label_batches,
    "notes": [
      "Rows must be real candidate outcomes, not duplicated holdout rows or synthetic labels.",
      "The existing family-split holdout remains valid; new post-Phase-3E outcomes should form a dated train/rolling-shadow batch.",
      "Full decode remains gated by the normal static, correctness, microbench, and confirmation rules.",
    ],
  }

def _readme(plan:dict[str, Any]) -> str:
  phase = plan.get("phase", "Phase 3E")
  lines = [
    f"# AMD Decode Flywheel {phase} Coverage Plan",
    "",
    "This artifact turns the current featured audit into a concrete",
    "data-collection batch. It does not add training examples by itself.",
    "",
    f"- conclusion: `{plan['conclusion']}`",
    f"- rerun Phase 3B allowed: `{plan['rerun_phase3b_allowed']}`",
    f"- minimum mechanism rows: `{plan['minimum_new_mechanism_rows']}`",
    f"- minimum label rows: `{plan['minimum_new_label_rows']}`",
    "",
    "## Mechanism Batches",
    "",
    "| mechanism | needed | batch | stage |",
    "|---|---:|---|---|",
  ]
  for row in plan["mechanism_batches"]:
    lines.append(f"| `{row['mechanism']}` | {row['needed_train_rows']} | {row['batch']} | `{row['candidate_stage']}` |")
  lines += ["", "## Label Batches", "", "| label | needed | note |", "|---|---:|---|"]
  for row in plan["label_batches"]:
    lines.append(f"| `{row['label']}` | {row['needed_train_rows']} | {row['recipe']} |")
  lines += ["", "## Rules", ""]
  for note in plan["notes"]:
    lines.append(f"- {note}")
  lines.append("")
  return "\n".join(lines)

def write_coverage_plan(audit_path:pathlib.Path, out:pathlib.Path) -> dict[str, Any]:
  audit = _read_json(audit_path)
  plan = build_coverage_plan(audit, audit_path)
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
  (out / "README.md").write_text(_readme(plan))
  return plan

def main() -> int:
  parser = argparse.ArgumentParser(description="Build Phase 3E flywheel cost-model coverage plan")
  parser.add_argument("--audit", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()
  plan = write_coverage_plan(args.audit, args.out)
  print(json.dumps(plan, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
