#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from collections import defaultdict
from typing import Any

from extra.llm_adapter_json_data_v4 import COMPILER_CONCEPTS, MAX_TOKENS, _completion, _prompt
from extra.llm_eval_common import read_prompt_jsonl, score_prompt

DATASET_KIND = "llm_adapter_json_dataset_v4_1_compiler"
CATEGORY = "compiler"
DEFAULT_TRAIN_ROWS = 68
DEFAULT_EVAL_ROWS = 34
STABLE_KEY_RE = re.compile(r"^qk_[a-z0-9_]+$")
NUMERIC_SUFFIX_RE = re.compile(r"_\d{3,}$")

TRAIN_TEMPLATES = (
  "Stable key: `{key}`. In the tinygrad GPU glossary, this key means it {definition}. Return the stable key exactly.",
  "For a tinygrad GPU glossary lookup, `{key}` is the stable key for the idea that {definition}. Return that stable key.",
  "The stable compiler glossary key `{key}` names the concept that {definition}. Return the key exactly as shown.",
  "Use the tinygrad GPU stable key `{key}` when a concept {definition}. Return only that stable key.",
  "A compiler note says `{key}` is the stable key whose definition is: it {definition}. Return the stable key.",
  "In the quantized-kernel glossary, stable key `{key}` means it {definition}. Return the key.",
)

EVAL_TEMPLATES = (
  "A tinygrad GPU glossary entry uses stable key `{key}` for a concept that {definition}. Return that stable key exactly.",
  "Which stable compiler glossary key should be returned for the concept that {definition}? The key is `{key}`.",
  "Given stable key `{key}` and meaning `it {definition}`, return the tinygrad GPU glossary key.",
  "The concept that {definition} is recorded under stable key `{key}`. Return only the stable key.",
  "For held-out compiler lookup, `{key}` is the stable key. It {definition}. Return the stable key.",
)

def stable_key(concept:str) -> str:
  return f"qk_{concept}"

def _normalized(answer:Any) -> Any:
  return answer.strip() if isinstance(answer, str) else answer

def _template(split:str, idx:int) -> str:
  templates = TRAIN_TEMPLATES if split == "train" else EVAL_TEMPLATES
  cycle = (idx - 1) // len(COMPILER_CONCEPTS)
  return templates[cycle % len(templates)]

def _row(split:str, idx:int, concept:str, definition:str) -> dict[str, Any]:
  answer = stable_key(concept)
  question = _template(split, idx).format(key=answer, definition=definition)
  template_id = f"compiler_stable_key_{concept}"
  row_id = f"json_v4_1_{split}_{CATEGORY}_{idx:03d}"
  return {
    "id": row_id,
    "source_id": row_id,
    "split": split,
    "category": CATEGORY,
    "prompt": _prompt(question),
    "completion": _completion(answer),
    "expected_json": {"answer": answer},
    "normalized_answer": _normalized(answer),
    "tags": ["json_answer", CATEGORY, split, "v4_1_compiler", "stable_key"],
    "max_tokens": MAX_TOKENS,
    "template_id": template_id,
    "template_instance": f"{split}:{template_id}:{question}",
    "concept": concept,
    "definition": definition,
    "stable_key": answer,
  }

def _compiler_rows(split:str, count:int) -> list[dict[str, Any]]:
  rows = []
  for i in range(1, count + 1):
    concept, definition = COMPILER_CONCEPTS[(i - 1) % len(COMPILER_CONCEPTS)]
    rows.append(_row(split, i, concept, definition))
  return rows

def build_rows(*, train_rows:int=DEFAULT_TRAIN_ROWS, eval_rows:int=DEFAULT_EVAL_ROWS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  if train_rows <= 0 or eval_rows <= 0:
    raise ValueError("train/eval counts must be positive")
  train_sft_rows = _compiler_rows("train", train_rows)
  eval_sft_rows = _compiler_rows("eval", eval_rows)
  sft_rows = train_sft_rows + eval_sft_rows
  prompt_rows = [
    {k: row[k] for k in ("id", "source_id", "split", "category", "prompt", "expected_json", "normalized_answer", "tags", "max_tokens", "template_id", "template_instance")}
    for row in eval_sft_rows
  ]
  return sft_rows, prompt_rows

def _json_key(value:Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))

def _ensure_unique(rows:list[dict[str, Any]], key:str, label:str, errors:list[str]) -> None:
  seen: set[Any] = set()
  for row in rows:
    value = row.get(key)
    if value in seen:
      errors.append(f"duplicate {label} {value!r}")
      return
    seen.add(value)

def _stable_answer_errors(row:dict[str, Any]) -> list[str]:
  row_id = row.get("id")
  answer = row.get("expected_json", {}).get("answer")
  errors: list[str] = []
  if not isinstance(answer, str):
    return [f"{row_id}: compiler answer must be a string"]
  if not STABLE_KEY_RE.fullmatch(answer):
    errors.append(f"{row_id}: compiler answer is not a stable qk_* key: {answer!r}")
  if answer.startswith(("train_", "eval_")):
    errors.append(f"{row_id}: compiler answer must not include split prefix: {answer!r}")
  if NUMERIC_SUFFIX_RE.search(answer):
    errors.append(f"{row_id}: compiler answer must not include row numeric suffix: {answer!r}")
  concept = row.get("concept")
  if isinstance(concept, str) and answer != stable_key(concept):
    errors.append(f"{row_id}: compiler answer {answer!r} does not match concept {concept!r}")
  stable = row.get("stable_key")
  if isinstance(stable, str) and stable != answer:
    errors.append(f"{row_id}: stable_key metadata differs from expected answer")
  return errors

def validate_dataset(sft_rows:list[dict[str, Any]], eval_rows:list[dict[str, Any]]) -> dict[str, Any]:
  errors: list[str] = []
  _ensure_unique(sft_rows, "id", "sft id", errors)
  _ensure_unique(eval_rows, "id", "eval id", errors)
  sft_by_id = {row.get("id"): row for row in sft_rows}
  train_sft_rows = [row for row in sft_rows if row.get("split") == "train"]
  eval_sft_rows = [row for row in sft_rows if row.get("split") == "eval"]
  if len(train_sft_rows) + len(eval_sft_rows) != len(sft_rows):
    errors.append("all SFT rows must have split=train or split=eval")
  if {row.get("id") for row in eval_sft_rows} != {row.get("id") for row in eval_rows}:
    errors.append("eval-prompts ids must match split=eval SFT rows")

  for row in sft_rows:
    for key in ("id", "source_id", "split", "category", "prompt", "completion", "expected_json", "normalized_answer", "tags", "max_tokens", "template_id", "template_instance"):
      if key not in row: errors.append(f"{row.get('id', '<missing>')}: missing {key}")
    if row.get("category") != CATEGORY: errors.append(f"{row.get('id')}: category must be compiler")
    if row.get("normalized_answer") != row.get("expected_json", {}).get("answer"):
      errors.append(f"{row.get('id')}: normalized_answer must match expected answer")
    errors += _stable_answer_errors(row)
    try:
      if json.loads(row.get("completion", "{}")) != row.get("expected_json"):
        errors.append(f"{row.get('id')}: completion does not match expected_json")
    except json.JSONDecodeError:
      errors.append(f"{row.get('id')}: completion is not JSON")

  for row in eval_rows:
    for key in ("id", "source_id", "split", "category", "prompt", "expected_json", "normalized_answer", "tags", "max_tokens", "template_id", "template_instance"):
      if key not in row: errors.append(f"{row.get('id', '<missing>')}: missing eval {key}")
    if row.get("split") != "eval": errors.append(f"{row.get('id')}: eval row split must be eval")
    if row.get("category") != CATEGORY: errors.append(f"{row.get('id')}: eval category must be compiler")
    if not isinstance(row.get("expected_json"), dict) or set(row.get("expected_json", {})) != {"answer"}:
      errors.append(f"{row.get('id')}: expected_json must be exactly one answer key")
    pseudo_sft = dict(sft_by_id.get(row.get("id"), {}), **row)
    errors += _stable_answer_errors(pseudo_sft)
    score = score_prompt(row, _completion(row.get("expected_json", {}).get("answer")))
    if score.get("status") != "pass": errors.append(f"{row.get('id')}: scorer rejected canonical completion")
    sft = sft_by_id.get(row.get("id"))
    if sft is not None and sft.get("expected_json") != row.get("expected_json"):
      errors.append(f"{row.get('id')}: eval expected_json differs from SFT expected_json")

  train_prompts = {row["prompt"] for row in train_sft_rows if "prompt" in row}
  eval_prompts = {row["prompt"] for row in eval_rows if "prompt" in row}
  train_answers = {_json_key(row.get("normalized_answer")) for row in train_sft_rows}
  eval_answers = {_json_key(row.get("normalized_answer")) for row in eval_rows}
  train_instances = {row["template_instance"] for row in train_sft_rows if "template_instance" in row}
  eval_instances = {row["template_instance"] for row in eval_rows if "template_instance" in row}
  prompt_overlap = sorted(train_prompts & eval_prompts)
  answer_overlap = sorted(train_answers & eval_answers)
  instance_overlap = sorted(train_instances & eval_instances)
  if prompt_overlap: errors.append(f"train/eval prompt overlap: {prompt_overlap[:3]}")
  if instance_overlap: errors.append(f"train/eval template_instance overlap: {instance_overlap[:3]}")

  category_rows: dict[str, dict[str, int]] = {CATEGORY: {"train_rows": 0, "eval_rows": 0}}
  for row in train_sft_rows: category_rows[CATEGORY]["train_rows"] += int(row.get("category") == CATEGORY)
  for row in eval_rows: category_rows[CATEGORY]["eval_rows"] += int(row.get("category") == CATEGORY)
  if errors:
    raise ValueError("; ".join(errors[:8]))
  return {
    "scorer_compatible": True,
    "eval_prompts_match_sft_eval_rows": True,
    "train_eval_prompt_overlap": len(prompt_overlap),
    "train_eval_answer_overlap": len(answer_overlap),
    "answer_overlap_allowed": True,
    "train_eval_template_instance_overlap": len(instance_overlap),
    "stable_compiler_answers": True,
    "compiler_answers_with_numeric_suffix": 0,
    "category_rows": category_rows,
  }

def _write_jsonl(path:pathlib.Path, rows:list[dict[str, Any]]) -> None:
  with path.open("w") as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")

def _read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def validate_dataset_dir(out:pathlib.Path) -> dict[str, Any]:
  sft_rows = _read_jsonl(out / "sft.jsonl")
  eval_rows = read_prompt_jsonl(out / "eval-prompts.jsonl")
  return validate_dataset(sft_rows, eval_rows)

def _readme(summary:dict[str, Any]) -> str:
  lines = [
    "# Adapter JSON Dataset V4.1 Compiler",
    "",
    "This compiler-only artifact isolates the Phase 4.2 prompt/data redesign.",
    "The expected answers are stable concept keys such as `qk_gemv`, not",
    "row-specific keys such as `train_qk_gemv_005`.",
    "",
    f"- SFT rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- held-out eval rows: `{summary['eval_rows']}`",
    "- category: `compiler`",
    "- schema: `{\"answer\": \"qk_<concept>\"}`",
    "- disjointness: train/eval prompts and template instances are checked",
    "- answer overlap: intentional, because the stable concept key is the task target",
    "",
    "## Stable Keys",
    "",
    "| concept | stable key | definition |",
    "|---|---|---|",
  ]
  for concept in summary["compiler_concepts"]:
    lines.append(f"| `{concept['concept']}` | `{concept['stable_key']}` | {concept['definition']} |")
  lines.append("")
  return "\n".join(lines)

def write_dataset(out:pathlib.Path, *, train_rows:int=DEFAULT_TRAIN_ROWS, eval_rows:int=DEFAULT_EVAL_ROWS) -> dict[str, Any]:
  sft_rows, prompt_rows = build_rows(train_rows=train_rows, eval_rows=eval_rows)
  integrity = validate_dataset(sft_rows, prompt_rows)
  category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train_rows": 0, "eval_rows": 0})
  for row in sft_rows:
    category_counts[row["category"]]["train_rows" if row["split"] == "train" else "eval_rows"] += 1
  summary = {
    "kind": DATASET_KIND,
    "version": "v4.1-compiler",
    "rows": len(sft_rows),
    "train_rows": sum(row["split"] == "train" for row in sft_rows),
    "eval_rows": len(prompt_rows),
    "categories": {CATEGORY: category_counts[CATEGORY]},
    "schema": {"answer": "stable_qk_key_string"},
    "compiler_concepts": [{"concept": concept, "stable_key": stable_key(concept), "definition": definition} for concept, definition in COMPILER_CONCEPTS],
    "files": {"sft": "sft.jsonl", "eval_prompts": "eval-prompts.jsonl"},
    "integrity": integrity,
    "note": "Compiler-only V4.1 strict-JSON data with stable concept-key answers for Phase 4.2.",
  }
  out.mkdir(parents=True, exist_ok=True)
  _write_jsonl(out / "sft.jsonl", sft_rows)
  _write_jsonl(out / "eval-prompts.jsonl", prompt_rows)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_readme(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Build strict JSON V4.1 compiler-only adapter SFT/eval data")
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
  parser.add_argument("--eval-rows", type=int, default=DEFAULT_EVAL_ROWS)
  parser.add_argument("--validate-only", action="store_true")
  args = parser.parse_args()
  if args.validate_only:
    summary = validate_dataset_dir(args.out)
  else:
    summary = write_dataset(args.out, train_rows=args.train_rows, eval_rows=args.eval_rows)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
