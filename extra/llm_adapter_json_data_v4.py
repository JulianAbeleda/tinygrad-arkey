#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from collections import defaultdict
from typing import Any

from extra.llm_eval_common import read_prompt_jsonl, score_prompt

CATEGORIES = ("arithmetic", "fact", "code", "compiler", "string", "categorization")
DEFAULT_TRAIN_PER_CATEGORY = 68
DEFAULT_EVAL_PER_CATEGORY = 34
MAX_TOKENS = 24

ELEMENTS = [
  ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"), ("Beryllium", "Be"), ("Boron", "B"), ("Carbon", "C"),
  ("Nitrogen", "N"), ("Oxygen", "O"), ("Fluorine", "F"), ("Neon", "Ne"), ("Sodium", "Na"), ("Magnesium", "Mg"),
  ("Aluminum", "Al"), ("Silicon", "Si"), ("Phosphorus", "P"), ("Sulfur", "S"), ("Chlorine", "Cl"), ("Argon", "Ar"),
  ("Potassium", "K"), ("Calcium", "Ca"), ("Scandium", "Sc"), ("Titanium", "Ti"), ("Vanadium", "V"), ("Chromium", "Cr"),
  ("Manganese", "Mn"), ("Iron", "Fe"), ("Cobalt", "Co"), ("Nickel", "Ni"), ("Copper", "Cu"), ("Zinc", "Zn"),
  ("Gallium", "Ga"), ("Germanium", "Ge"), ("Arsenic", "As"), ("Selenium", "Se"), ("Bromine", "Br"), ("Krypton", "Kr"),
  ("Rubidium", "Rb"), ("Strontium", "Sr"), ("Yttrium", "Y"), ("Zirconium", "Zr"), ("Niobium", "Nb"), ("Molybdenum", "Mo"),
  ("Technetium", "Tc"), ("Ruthenium", "Ru"), ("Rhodium", "Rh"), ("Palladium", "Pd"), ("Silver", "Ag"), ("Cadmium", "Cd"),
  ("Indium", "In"), ("Tin", "Sn"), ("Antimony", "Sb"), ("Tellurium", "Te"), ("Iodine", "I"), ("Xenon", "Xe"),
  ("Cesium", "Cs"), ("Barium", "Ba"), ("Lanthanum", "La"), ("Cerium", "Ce"), ("Praseodymium", "Pr"), ("Neodymium", "Nd"),
  ("Promethium", "Pm"), ("Samarium", "Sm"), ("Europium", "Eu"), ("Gadolinium", "Gd"), ("Terbium", "Tb"), ("Dysprosium", "Dy"),
  ("Holmium", "Ho"), ("Erbium", "Er"), ("Thulium", "Tm"), ("Ytterbium", "Yb"), ("Lutetium", "Lu"), ("Hafnium", "Hf"),
  ("Tantalum", "Ta"), ("Tungsten", "W"), ("Rhenium", "Re"), ("Osmium", "Os"), ("Iridium", "Ir"), ("Platinum", "Pt"),
  ("Gold", "Au"), ("Mercury", "Hg"), ("Thallium", "Tl"), ("Lead", "Pb"), ("Bismuth", "Bi"), ("Polonium", "Po"),
  ("Astatine", "At"), ("Radon", "Rn"), ("Francium", "Fr"), ("Radium", "Ra"), ("Actinium", "Ac"), ("Thorium", "Th"),
  ("Protactinium", "Pa"), ("Uranium", "U"), ("Neptunium", "Np"), ("Plutonium", "Pu"), ("Americium", "Am"), ("Curium", "Cm"),
  ("Berkelium", "Bk"), ("Californium", "Cf"), ("Einsteinium", "Es"), ("Fermium", "Fm"), ("Mendelevium", "Md"), ("Nobelium", "No"),
  ("Lawrencium", "Lr"), ("Rutherfordium", "Rf"), ("Dubnium", "Db"), ("Seaborgium", "Sg"), ("Bohrium", "Bh"), ("Hassium", "Hs"),
  ("Meitnerium", "Mt"), ("Darmstadtium", "Ds"), ("Roentgenium", "Rg"), ("Copernicium", "Cn"), ("Nihonium", "Nh"), ("Flerovium", "Fl"),
  ("Moscovium", "Mc"), ("Livermorium", "Lv"), ("Tennessine", "Ts"), ("Oganesson", "Og"),
]

COMPILER_CONCEPTS = [
  ("wide_load", "loads multiple adjacent packed words at once"),
  ("coalesced_read", "maps neighboring lanes to neighboring addresses"),
  ("wavefront", "names the SIMD execution group on AMD GPUs"),
  ("dequant", "converts packed quantized values toward floating point values"),
  ("gemv", "multiplies a matrix by one vector"),
  ("q4_block", "stores a group of four-bit quantized weights"),
  ("q6_block", "stores a group of six-bit quantized weights"),
  ("uop", "names tinygrad's internal operation node"),
  ("beam", "searches schedule choices in tinygrad"),
  ("policy", "records which lowering choice to use for a tensor family"),
  ("suffix_cache", "stores the frozen prefix hidden state before adapter blocks"),
  ("json_axis", "separates parse, schema, type, and value scoring"),
]

def _completion(answer:Any) -> str:
  return json.dumps({"answer": answer}, separators=(",", ":"))

def _prompt(question:str) -> str:
  return f'Return only compact JSON with exactly one key "answer". No prose. Question: {question}'

def _normalized(answer:Any) -> Any:
  return answer.strip() if isinstance(answer, str) else answer

def _row(split:str, category:str, idx:int, template_id:str, question:str, answer:Any) -> dict[str, Any]:
  row_id = f"json_v4_{split}_{category}_{idx:03d}"
  prompt = _prompt(question)
  normalized_answer = _normalized(answer)
  return {
    "id": row_id,
    "source_id": row_id,
    "split": split,
    "category": category,
    "prompt": prompt,
    "completion": _completion(answer),
    "expected_json": {"answer": answer},
    "normalized_answer": normalized_answer,
    "tags": ["json_answer", category, split],
    "max_tokens": MAX_TOKENS,
    "template_id": template_id,
    "template_instance": f"{template_id}:{question}",
  }

def _arithmetic_rows(split:str, count:int) -> list[dict[str, Any]]:
  base = 1000 if split == "train" else 7000
  rows = []
  for i in range(1, count + 1):
    a, b = base + 11 * i, base // 2 + 7 * i + 3
    if i % 3 == 0:
      question, answer, template_id = f"What is {a} - {b}?", a - b, "arithmetic_subtract"
    elif i % 3 == 1:
      question, answer, template_id = f"What is {a} + {b}?", a + b, "arithmetic_add"
    else:
      left, right = 20 + i, 3 + (i % 9)
      question, answer, template_id = f"What is {left} * {right}?", left * right + (0 if split == "train" else 5000), "arithmetic_multiply_offset"
      question = f"What is ({left} * {right}) + {0 if split == 'train' else 5000}?"
    rows.append(_row(split, "arithmetic", i, template_id, question, answer))
  return rows

def _fact_rows(split:str, count:int) -> list[dict[str, Any]]:
  pool = ELEMENTS[:count] if split == "train" else ELEMENTS[-count:]
  rows = []
  for i, (name, symbol) in enumerate(pool, 1):
    rows.append(_row(split, "fact", i, "chemical_symbol", f"What is the chemical symbol for {name}?", symbol))
  return rows

def _code_rows(split:str, count:int) -> list[dict[str, Any]]:
  rows = []
  for i in range(1, count + 1):
    ident = f"{split}_json_handler_{i:03d}"
    question = f"What Python-style identifier is shown between backticks: `{ident}`?"
    rows.append(_row(split, "code", i, "identifier_copy", question, ident))
  return rows

def _compiler_rows(split:str, count:int) -> list[dict[str, Any]]:
  rows = []
  for i in range(1, count + 1):
    concept, definition = COMPILER_CONCEPTS[(i - 1) % len(COMPILER_CONCEPTS)]
    key = f"{split}_qk_{concept}_{i:03d}"
    question = f"In this tinygrad GPU glossary, `{key}` means it {definition}. What glossary key names that concept?"
    rows.append(_row(split, "compiler", i, f"compiler_glossary_{concept}", question, key))
  return rows

def _string_rows(split:str, count:int) -> list[dict[str, Any]]:
  rows = []
  for i in range(1, count + 1):
    token = f"{split}qk{i:03d}ab"
    if i % 3 == 0:
      question, answer, template_id = f"Reverse the string `{token}`.", token[::-1], "string_reverse"
    elif i % 3 == 1:
      question, answer, template_id = f"Return the uppercase form of `{token}`.", token.upper(), "string_upper"
    else:
      phrase = f"{split} alpha{i:03d} beta{i:03d}"
      answer = f"{split[0]}ab{i:03d}"
      question, template_id = f"Take the first letters of each word in `{phrase}`.", "string_initials"
    rows.append(_row(split, "string", i, template_id, question, answer))
  return rows

def _categorization_rows(split:str, count:int) -> list[dict[str, Any]]:
  rows = []
  base = 2000 if split == "train" else 9000
  for i in range(1, count + 1):
    number = base + i
    even_label, odd_label = f"{split}_even_bucket_{i:03d}", f"{split}_odd_bucket_{i:03d}"
    answer = even_label if number % 2 == 0 else odd_label
    question = f"Classify {number} as even or odd. Return `{even_label}` for even and `{odd_label}` for odd."
    rows.append(_row(split, "categorization", i, "binary_even_odd_label", question, answer))
  return rows

def build_rows(*, train_per_category:int=DEFAULT_TRAIN_PER_CATEGORY, eval_per_category:int=DEFAULT_EVAL_PER_CATEGORY) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  if train_per_category <= 0 or eval_per_category <= 0:
    raise ValueError("train/eval counts must be positive")
  if train_per_category + eval_per_category > len(ELEMENTS):
    raise ValueError(f"fact category needs {train_per_category + eval_per_category} unique elements, only {len(ELEMENTS)} available")
  builders = (_arithmetic_rows, _fact_rows, _code_rows, _compiler_rows, _string_rows, _categorization_rows)
  train_rows, eval_sft_rows = [], []
  for builder in builders:
    train_rows += builder("train", train_per_category)
    eval_sft_rows += builder("eval", eval_per_category)
  sft_rows = train_rows + eval_sft_rows
  eval_rows = [
    {k: row[k] for k in ("id", "source_id", "split", "category", "prompt", "expected_json", "normalized_answer", "tags", "max_tokens", "template_id", "template_instance")}
    for row in eval_sft_rows
  ]
  return sft_rows, eval_rows

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

def validate_dataset(sft_rows:list[dict[str, Any]], eval_rows:list[dict[str, Any]]) -> dict[str, Any]:
  errors: list[str] = []
  _ensure_unique(sft_rows, "id", "sft id", errors)
  _ensure_unique(eval_rows, "id", "eval id", errors)
  sft_by_id = {row.get("id"): row for row in sft_rows}
  train_rows = [row for row in sft_rows if row.get("split") == "train"]
  eval_sft_rows = [row for row in sft_rows if row.get("split") == "eval"]
  if len(train_rows) + len(eval_sft_rows) != len(sft_rows): errors.append("all SFT rows must have split=train or split=eval")
  if {row.get("id") for row in eval_sft_rows} != {row.get("id") for row in eval_rows}:
    errors.append("eval-prompts ids must match split=eval SFT rows")
  for row in sft_rows:
    for key in ("id", "source_id", "split", "category", "prompt", "completion", "expected_json", "normalized_answer", "tags", "max_tokens", "template_id", "template_instance"):
      if key not in row: errors.append(f"{row.get('id', '<missing>')}: missing {key}")
    try:
      if json.loads(row.get("completion", "{}")) != row.get("expected_json"):
        errors.append(f"{row.get('id')}: completion does not match expected_json")
    except json.JSONDecodeError:
      errors.append(f"{row.get('id')}: completion is not JSON")
  for row in eval_rows:
    for key in ("id", "source_id", "split", "category", "prompt", "expected_json", "normalized_answer", "tags", "max_tokens", "template_id", "template_instance"):
      if key not in row: errors.append(f"{row.get('id', '<missing>')}: missing eval {key}")
    if row.get("split") != "eval": errors.append(f"{row.get('id')}: eval row split must be eval")
    if not isinstance(row.get("expected_json"), dict) or set(row.get("expected_json", {})) != {"answer"}:
      errors.append(f"{row.get('id')}: expected_json must be exactly one answer key")
    score = score_prompt(row, _completion(row.get("expected_json", {}).get("answer")))
    if score.get("status") != "pass": errors.append(f"{row.get('id')}: scorer rejected canonical completion")
    sft = sft_by_id.get(row.get("id"))
    if sft is not None and sft.get("expected_json") != row.get("expected_json"):
      errors.append(f"{row.get('id')}: eval expected_json differs from SFT expected_json")

  train_prompts = {row["prompt"] for row in train_rows if "prompt" in row}
  eval_prompts = {row["prompt"] for row in eval_rows if "prompt" in row}
  train_answers = {_json_key(row.get("normalized_answer")) for row in train_rows}
  eval_answers = {_json_key(row.get("normalized_answer")) for row in eval_rows}
  train_instances = {row["template_instance"] for row in train_rows if "template_instance" in row}
  eval_instances = {row["template_instance"] for row in eval_rows if "template_instance" in row}
  prompt_overlap = sorted(train_prompts & eval_prompts)
  answer_overlap = sorted(train_answers & eval_answers)
  instance_overlap = sorted(train_instances & eval_instances)
  if prompt_overlap: errors.append(f"train/eval prompt overlap: {prompt_overlap[:3]}")
  if answer_overlap: errors.append(f"train/eval answer overlap: {answer_overlap[:3]}")
  if instance_overlap: errors.append(f"train/eval template_instance overlap: {instance_overlap[:3]}")
  category_rows: dict[str, dict[str, int]] = {category: {"train_rows": 0, "eval_rows": 0} for category in CATEGORIES}
  for row in train_rows:
    if row.get("category") in category_rows: category_rows[row["category"]]["train_rows"] += 1
    else: errors.append(f"{row.get('id')}: unknown category {row.get('category')!r}")
  for row in eval_rows:
    if row.get("category") in category_rows: category_rows[row["category"]]["eval_rows"] += 1
    else: errors.append(f"{row.get('id')}: unknown eval category {row.get('category')!r}")
  if errors:
    raise ValueError("; ".join(errors[:8]))
  return {
    "scorer_compatible": True,
    "eval_prompts_match_sft_eval_rows": True,
    "train_eval_prompt_overlap": len(prompt_overlap),
    "train_eval_answer_overlap": len(answer_overlap),
    "train_eval_template_instance_overlap": len(instance_overlap),
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
    "# Adapter JSON Dataset V4",
    "",
    "This is the first strict-JSON generation eval large enough to act as a",
    "promotion gate. It keeps the task deterministic: each prompt expects compact",
    "JSON with exactly one key, `answer`, and every answer is scored by the",
    "multi-axis JSON scorer.",
    "",
    f"- SFT rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- held-out eval rows: `{summary['eval_rows']}`",
    "- categories: `arithmetic`, `fact`, `code`, `compiler`, `string`, `categorization`",
    "- schema: `{\"answer\": ...}` with strings and integers",
    "- disjointness: train/eval prompts, answers, and template instances are mechanically checked",
    "",
    "The categorization prompts use binary-choice labels rather than raw JSON",
    "booleans so the train/eval answer sets can remain disjoint.",
    "",
    "## Category Balance",
    "",
    "| category | train | eval |",
    "|---|---:|---:|",
  ]
  for category, row in summary["categories"].items():
    lines.append(f"| `{category}` | {row['train_rows']} | {row['eval_rows']} |")
  lines.append("")
  return "\n".join(lines)

def write_dataset(out:pathlib.Path, *, train_per_category:int=DEFAULT_TRAIN_PER_CATEGORY, eval_per_category:int=DEFAULT_EVAL_PER_CATEGORY) -> dict[str, Any]:
  sft_rows, eval_rows = build_rows(train_per_category=train_per_category, eval_per_category=eval_per_category)
  integrity = validate_dataset(sft_rows, eval_rows)
  category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train_rows": 0, "eval_rows": 0})
  for row in sft_rows:
    category_counts[row["category"]]["train_rows" if row["split"] == "train" else "eval_rows"] += 1
  summary = {
    "kind": "llm_adapter_json_dataset_v4",
    "rows": len(sft_rows),
    "train_rows": sum(row["split"] == "train" for row in sft_rows),
    "eval_rows": len(eval_rows),
    "train_per_category": train_per_category,
    "eval_per_category": eval_per_category,
    "categories": {category: category_counts[category] for category in CATEGORIES},
    "schema": {"answer": "string_or_integer"},
    "files": {"sft": "sft.jsonl", "eval_prompts": "eval-prompts.jsonl"},
    "integrity": integrity,
    "note": "Strict JSON-answer V4 data for generation-gated adapter/objective evaluation.",
  }
  out.mkdir(parents=True, exist_ok=True)
  _write_jsonl(out / "sft.jsonl", sft_rows)
  _write_jsonl(out / "eval-prompts.jsonl", eval_rows)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_readme(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Build strict JSON V4 adapter SFT/eval data")
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--train-per-category", type=int, default=DEFAULT_TRAIN_PER_CATEGORY)
  parser.add_argument("--eval-per-category", type=int, default=DEFAULT_EVAL_PER_CATEGORY)
  parser.add_argument("--validate-only", action="store_true")
  args = parser.parse_args()
  if args.validate_only:
    summary = validate_dataset_dir(args.out)
  else:
    summary = write_dataset(args.out, train_per_category=args.train_per_category, eval_per_category=args.eval_per_category)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
