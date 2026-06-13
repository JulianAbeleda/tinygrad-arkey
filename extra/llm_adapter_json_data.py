#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib

TASKS = [
  ("json_math_001", "What is 14 + 28?", "42", ["json_answer", "math"]),
  ("json_fact_002", "What is the capital of France?", "Paris", ["json_answer", "fact"]),
  ("json_code_003", "What Python function returns the length of a list?", "len", ["json_answer", "code"]),
  ("json_fact_004", "Which planet is called the Red Planet?", "Mars", ["json_answer", "fact"]),
  ("json_pattern_005", "Give the next number: 3, 6, 12, 24,", "48", ["json_answer", "pattern"]),
  ("json_compiler_006", "What does GPU stand for?", "graphics processing unit", ["json_answer", "compiler"]),
  ("json_sort_007", "Sort these numbers ascending: 3, 1, 2.", "1,2,3", ["json_answer", "reasoning"]),
  ("json_fact_008", "What is the chemical formula for water?", "H2O", ["json_answer", "fact"]),
  ("json_reason_009", "Which word comes first alphabetically: banana, apple, cherry?", "apple", ["json_answer", "reasoning"]),
  ("json_fact_010", "What color do red and blue paint usually make?", "purple", ["json_answer", "fact"]),
  ("json_math_011", "What is 100 - 37?", "63", ["json_answer", "math"]),
  ("json_transform_012", "Return the lowercase of DOG.", "dog", ["json_answer", "transform"]),
  ("json_compiler_013", "What does JIT stand for?", "just in time", ["json_answer", "compiler"]),
  ("json_code_014", "What Python data type stores key-value pairs?", "dict", ["json_answer", "code"]),
  ("json_math_015", "What is half of 50?", "25", ["json_answer", "math"]),
  ("json_fact_016", "Name a mammal that lays eggs.", "platypus", ["json_answer", "fact"]),
  ("json_compiler_017", "What does memory bandwidth measure?", "data transfer rate", ["json_answer", "compiler"]),
  ("json_reason_018", "Which is heavier: 1 kilogram or 1 gram?", "1 kilogram", ["json_answer", "reasoning"]),
  ("json_code_019", "What Python keyword loads a module?", "import", ["json_answer", "code"]),
  ("json_compiler_020", "What does quantization usually do to model weights?", "reduce precision", ["json_answer", "compiler"]),
  ("json_math_021", "What is 12 * 8?", "96", ["json_answer", "math"]),
  ("json_fact_022", "What is the largest ocean on Earth?", "Pacific", ["json_answer", "fact"]),
  ("json_transform_023", "What is the final word in: load store compute?", "compute", ["json_answer", "transform"]),
  ("json_reason_024", "What is the opposite direction of north?", "south", ["json_answer", "reasoning"]),
  ("json_math_025", "What is 10 percent of 80?", "8", ["json_answer", "math"]),
  ("json_compiler_026", "What does GEMV multiply?", "matrix and vector", ["json_answer", "compiler"]),
  ("json_fact_027", "Who wrote Hamlet?", "Shakespeare", ["json_answer", "fact"]),
  ("json_math_028", "What is 144 / 12?", "12", ["json_answer", "math"]),
  ("json_transform_029", "Return the uppercase of cat.", "CAT", ["json_answer", "transform"]),
  ("json_compiler_030", "What does dequantization convert quantized values toward?", "floating point", ["json_answer", "compiler"]),
  ("json_math_031", "What is 11 squared?", "121", ["json_answer", "math"]),
  ("json_reason_032", "Two days after Monday is which day?", "Wednesday", ["json_answer", "reasoning"]),
  ("json_code_033", "Python list indexing starts at what number?", "0", ["json_answer", "code"]),
  ("json_reason_034", "What is the first month of the year?", "January", ["json_answer", "reasoning"]),
  ("json_math_035", "What is 2 + 12?", "14", ["json_answer", "math"]),
  ("json_compiler_036", "What kind of object does tinygrad mainly operate on?", "tensor", ["json_answer", "compiler"]),
  ("json_math_037", "Which number is smaller: 19 or 27?", "19", ["json_answer", "math"]),
  ("json_fact_038", "What is the currency of the United States?", "dollar", ["json_answer", "fact"]),
  ("json_math_039", "What is -3 + 10?", "7", ["json_answer", "math"]),
  ("json_compiler_040", "In GPU programming, what is a kernel?", "program", ["json_answer", "compiler"]),
  ("json_math_041", "What is 5 - 3?", "2", ["json_answer", "math"]),
  ("json_fact_042", "Egypt is on which continent?", "Africa", ["json_answer", "fact"]),
  ("json_code_043", "What operator tests equality in Python?", "==", ["json_answer", "code"]),
  ("json_code_044", "What is the true boolean value in Python?", "True", ["json_answer", "code"]),
  ("json_math_045", "What is the average of 4, 6, and 8?", "6", ["json_answer", "math"]),
  ("json_reason_046", "Which is warmest: ice, liquid water, or steam?", "steam", ["json_answer", "reasoning"]),
  ("json_code_047", "What function writes text to standard output in Python?", "print", ["json_answer", "code"]),
  ("json_math_048", "How many vowels are in the word cat?", "1", ["json_answer", "math"]),
  ("json_reason_049", "What is the last letter of tinygrad?", "d", ["json_answer", "reasoning"]),
  ("json_math_050", "What is 15 + 0?", "15", ["json_answer", "math"]),
  ("json_code_051", "What list method adds one item to the end?", "append", ["json_answer", "code"]),
  ("json_transform_052", "Take the first letters of Alpha Beta.", "AB", ["json_answer", "transform"]),
  ("json_fact_053", "What is the capital of Japan?", "Tokyo", ["json_answer", "fact"]),
  ("json_fact_054", "Mixing yellow and blue paint usually makes what color?", "green", ["json_answer", "fact"]),
  ("json_code_055", "What character starts a single-line comment in Python?", "#", ["json_answer", "code"]),
  ("json_compiler_056", "What does a compiler optimization pass try to improve?", "performance", ["json_answer", "compiler"]),
  ("json_code_057", "Which keyword starts a for-loop in Python?", "for", ["json_answer", "code"]),
  ("json_reason_058", "Is 2 an even number?", "yes", ["json_answer", "reasoning"]),
  ("json_sort_059", "What is the sorted order of c, a, b?", "a,b,c", ["json_answer", "reasoning"]),
  ("json_compiler_060", "What does CPU stand for?", "central processing unit", ["json_answer", "compiler"]),
]

def _completion(answer:str) -> str:
  return json.dumps({"answer": answer}, separators=(",", ":"))

def _prompt(question:str) -> str:
  return f'Return only compact JSON with exactly one key "answer". No prose. Question: {question}'

def write_dataset(out:pathlib.Path, *, eval_every:int=5, limit:int=len(TASKS)) -> dict:
  if eval_every < 2: raise ValueError("--eval-every must be >= 2")
  tasks = TASKS[:limit]
  if len(tasks) < eval_every: raise ValueError("--limit too small for requested split")
  out.mkdir(parents=True, exist_ok=True)
  sft_rows, eval_rows = [], []
  for idx, (row_id, question, answer, tags) in enumerate(tasks, 1):
    split = "eval" if idx % eval_every == 0 else "train"
    prompt = _prompt(question)
    completion = _completion(answer)
    sft_rows.append({
      "id": row_id, "source_id": row_id, "split": split,
      "prompt": prompt, "completion": completion, "tags": tags + [split],
    })
    if split == "eval":
      eval_rows.append({
        "id": row_id, "prompt": prompt, "tags": tags + [split],
        "max_tokens": 24, "expected_json": {"answer": answer},
      })
  with (out / "sft.jsonl").open("w") as f:
    for row in sft_rows: f.write(json.dumps(row, sort_keys=True) + "\n")
  with (out / "eval-prompts.jsonl").open("w") as f:
    for row in eval_rows: f.write(json.dumps(row, sort_keys=True) + "\n")
  summary = {
    "kind": "llm_adapter_json_dataset",
    "rows": len(sft_rows),
    "train_rows": sum(row["split"] == "train" for row in sft_rows),
    "eval_rows": len(eval_rows),
    "eval_every": eval_every,
    "schema": {"answer": "string"},
    "files": {"sft": "sft.jsonl", "eval_prompts": "eval-prompts.jsonl"},
    "note": "Human-authored strict JSON-answer data for output-LoRA behavior-change eval.",
  }
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(
    "# Adapter JSON Dataset V3\n\n"
    "This dataset tests a narrow real behavior target: return only compact JSON\n"
    "with exactly one key, `answer`, and the expected string value. It is designed\n"
    "for automatic held-out scoring and is not a broad capability benchmark.\n\n"
    f"- rows: `{summary['rows']}`\n"
    f"- train rows: `{summary['train_rows']}`\n"
    f"- eval rows: `{summary['eval_rows']}`\n"
    "- schema: `{\"answer\":\"...\"}`\n"
  )
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Build human-authored strict JSON-answer adapter data")
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--eval-every", type=int, default=5)
  parser.add_argument("--limit", type=int, default=len(TASKS))
  args = parser.parse_args()
  print(json.dumps(write_dataset(args.out, eval_every=args.eval_every, limit=args.limit), indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
