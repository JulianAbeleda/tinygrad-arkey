#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from collections import Counter
from typing import Any

from extra.llm_eval_common import score_prompt
from extra.qk_flywheel_dataset import LABELS, REASONS

from extra.llm_eval_common import read_jsonl as _read_jsonl

from extra.llm_eval_common import write_jsonl as _jsonl

def _completion(row:dict[str, Any]) -> str:
  return json.dumps({"label": row["label"], "reason": row["reason"], "retry": row["retry"]}, separators=(",", ":"))

def _prompt_by_id(prompts:list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  out = {}
  for row in prompts:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id: raise ValueError("prompt row missing id")
    if row_id in out: raise ValueError(f"duplicate prompt id {row_id}")
    out[row_id] = row
  return out

def _sft_row(example:dict[str, Any], prompt:dict[str, Any], split:str, ordinal:int, repeat:int=0) -> dict[str, Any]:
  completion = _completion(example)
  expected_json = {"label": example["label"], "reason": example["reason"], "retry": example["retry"]}
  row = {
    "id": f"qk_triage_{split}_{ordinal:04d}" if repeat == 0 else f"qk_triage_{split}_{ordinal:04d}_rep{repeat:02d}",
    "source_id": example["id"],
    "split": split,
    "category": "qk_kernel_triage",
    "prompt": prompt["prompt"],
    "completion": completion,
    "expected_json": expected_json,
    "tags": ["qk_flywheel", "kernel_triage", split, example["family"], example["mechanism"], example["label"]],
    "max_tokens": 64,
    "family": example["family"],
    "mechanism": example["mechanism"],
    "label": example["label"],
    "reason": example["reason"],
    "retry": example["retry"],
    "source_split": example["split"],
    "source_files": example.get("source_files", []),
  }
  score = score_prompt({"id": row["id"], "expected_json": expected_json}, completion)
  if score.get("status") != "pass": raise ValueError(f"{row['id']}: canonical completion failed strict JSON scorer")
  return row

def build_sft_rows(examples:list[dict[str, Any]], prompts:list[dict[str, Any]], *, oversample_min_per_label:int=0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  prompts_by_id = _prompt_by_id(prompts)
  if set(prompts_by_id) != {row["id"] for row in examples}: raise ValueError("examples/prompts id mismatch")
  train_examples = [row for row in examples if row.get("split") == "train"]
  holdout_examples = [row for row in examples if row.get("split") == "holdout"]
  if not train_examples or not holdout_examples: raise ValueError("expected non-empty train and holdout examples")
  for row in examples:
    if row.get("label") not in LABELS: raise ValueError(f"{row.get('id')}: unknown label {row.get('label')}")
    if row.get("reason") not in REASONS: raise ValueError(f"{row.get('id')}: unknown reason {row.get('reason')}")

  rows = []
  train_rows = []
  eval_rows = []
  for idx, example in enumerate(train_examples, 1):
    row = _sft_row(example, prompts_by_id[example["id"]], "train", idx)
    rows.append(row)
    train_rows.append(row)
  for idx, example in enumerate(holdout_examples, 1):
    row = _sft_row(example, prompts_by_id[example["id"]], "eval", idx)
    rows.append(row)
    eval_rows.append(row)

  oversampled = []
  if oversample_min_per_label > 0:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows: by_label.setdefault(row["label"], []).append(row)
    next_ord = len(train_rows) + 1
    for label, label_rows in sorted(by_label.items()):
      deficit = max(0, oversample_min_per_label - len(label_rows))
      for i in range(deficit):
        source = label_rows[i % len(label_rows)]
        clone = dict(source)
        clone["id"] = f"qk_triage_train_{next_ord:04d}_oversample_{label}_{i + 1:02d}"
        clone["oversampled_from"] = source["id"]
        rows.insert(len(train_rows) + len(oversampled), clone)
        oversampled.append(clone)
        next_ord += 1

  train_ids = {row["source_id"] for row in rows if row["split"] == "train"}
  eval_ids = {row["source_id"] for row in rows if row["split"] == "eval"}
  if train_ids & eval_ids: raise ValueError(f"holdout contamination: {sorted(train_ids & eval_ids)[:3]}")
  summary = {
    "kind": "qk_flywheel_triage_sft_dataset",
    "rows": len(rows),
    "train_rows": sum(row["split"] == "train" for row in rows),
    "eval_rows": sum(row["split"] == "eval" for row in rows),
    "source_train_rows": len(train_examples),
    "source_holdout_rows": len(holdout_examples),
    "oversampled_rows": len(oversampled),
    "oversample_min_per_label": oversample_min_per_label,
    "schema_support_rows": 0,
    "train_label_counts": dict(sorted(Counter(row["label"] for row in rows if row["split"] == "train").items())),
    "eval_label_counts": dict(sorted(Counter(row["label"] for row in rows if row["split"] == "eval").items())),
    "train_reason_counts": dict(sorted(Counter(row["reason"] for row in rows if row["split"] == "train").items())),
    "eval_reason_counts": dict(sorted(Counter(row["reason"] for row in rows if row["split"] == "eval").items())),
    "holdout_ids_in_train": 0,
    "files": {
      "adapter_input": "adapter-input.jsonl",
      "train": "train.jsonl",
      "holdout_prompts": "holdout-prompts.jsonl",
      "summary": "summary.json",
      "readme": "README.md",
    },
  }
  return rows, holdout_prompt_rows(holdout_examples, prompts_by_id), summary

def holdout_prompt_rows(holdout_examples:list[dict[str, Any]], prompts_by_id:dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  rows = []
  for example in holdout_examples:
    prompt = dict(prompts_by_id[example["id"]])
    prompt["expected_json"] = {"label": example["label"], "reason": example["reason"], "retry": example["retry"]}
    prompt["max_tokens"] = 64
    rows.append(prompt)
  return rows

def write_dataset(examples_path:pathlib.Path, prompts_path:pathlib.Path, out:pathlib.Path, *, oversample_min_per_label:int=0) -> dict[str, Any]:
  examples = _read_jsonl(examples_path)
  prompts = _read_jsonl(prompts_path)
  rows, holdout_prompts, summary = build_sft_rows(examples, prompts, oversample_min_per_label=oversample_min_per_label)
  out.mkdir(parents=True, exist_ok=True)
  _jsonl(out / "adapter-input.jsonl", rows)
  _jsonl(out / "train.jsonl", [row for row in rows if row["split"] == "train"])
  _jsonl(out / "holdout-prompts.jsonl", holdout_prompts)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_readme(summary))
  return summary

def _readme(summary:dict[str, Any]) -> str:
  lines = [
    "# QK Flywheel Triage SFT Dataset",
    "",
    "This Phase 3.1 artifact converts the Phase 1 kernel-history triage dataset",
    "into strict JSON SFT rows for adapter training. Holdout rows are included",
    "only as `split=eval` rows for teacher-forced diagnostics and as rollout",
    "prompts; they are not optimized as train rows.",
    "",
    f"- rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- eval/holdout rows: `{summary['eval_rows']}`",
    f"- oversampled rows: `{summary['oversampled_rows']}`",
    f"- schema-support rows: `{summary['schema_support_rows']}`",
    f"- holdout ids in train: `{summary['holdout_ids_in_train']}`",
    "",
    "## Train Labels",
    "",
    "| label | rows |",
    "|---|---:|",
  ]
  for label, count in summary["train_label_counts"].items(): lines.append(f"| `{label}` | {count} |")
  lines += ["", "## Eval Labels", "", "| label | rows |", "|---|---:|"]
  for label, count in summary["eval_label_counts"].items(): lines.append(f"| `{label}` | {count} |")
  lines.append("")
  return "\n".join(lines)

def main() -> int:
  parser = argparse.ArgumentParser(description="Export Phase 3 QK flywheel triage SFT rows")
  parser.add_argument("--examples", type=pathlib.Path, required=True)
  parser.add_argument("--prompts", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--oversample-min-per-label", type=int, default=0)
  args = parser.parse_args()
  summary = write_dataset(args.examples, args.prompts, args.out, oversample_min_per_label=args.oversample_min_per_label)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
