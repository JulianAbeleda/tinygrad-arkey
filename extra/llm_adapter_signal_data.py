#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib

PROMPTS = [
  "What is 14 + 28?",
  "Name the capital of France.",
  "Write the Python function that returns a list length.",
  "Which planet is known as the Red Planet?",
  "Give the next number: 3, 6, 12, 24,",
  "What does GPU stand for?",
  "Sort these numbers ascending: 3, 1, 2.",
  "What is the chemical formula for water?",
  "Which word comes first alphabetically: banana, apple, cherry?",
  "What color do red and blue paint usually make?",
  "What is 100 - 37?",
  "What is the lowercase of DOG?",
  "What is JIT short for in compiler/runtime systems?",
  "What data type stores key-value pairs in Python?",
  "What is half of 50?",
  "Name a mammal that lays eggs.",
  "What is memory bandwidth a measure of?",
  "Which is heavier: 1 kilogram or 1 gram?",
  "What Python keyword loads a module?",
  "What does quantization usually do to model weights?",
  "What is 12 * 8?",
  "Which ocean is the largest on Earth?",
  "What is the final word in: load store compute?",
  "What is the opposite direction of north?",
  "What is 10 percent of 80?",
  "What does GEMV multiply?",
  "Who wrote Hamlet?",
  "What is 144 / 12?",
  "Return the uppercase of cat.",
  "What does dequantization convert quantized values toward?",
  "What is 11 squared?",
  "Two days after Monday is which day?",
  "Python list indexing starts at what number?",
  "What is the first month of the year?",
  "What is 2 + 12?",
  "What kind of object does tinygrad mainly operate on?",
  "Which number is smaller: 19 or 27?",
  "What is the currency of the United States?",
  "What is -3 + 10?",
  "In GPU programming, what is a kernel?",
  "What is 5 - 3?",
  "Egypt is on which continent?",
  "What operator tests equality in Python?",
  "What is the true boolean value in Python?",
  "What is the average of 4, 6, and 8?",
  "Which is warmest: ice, liquid water, or steam?",
  "What function writes text to standard output in Python?",
  "How many vowels are in the word cat?",
  "What is the last letter of tinygrad?",
  "What is 15 + 0?",
  "What list method adds one item to the end?",
  "Take the first letters of Alpha Beta.",
  "What is the capital of Japan?",
  "Mixing yellow and blue paint usually makes what color?",
  "What character starts a single-line comment in Python?",
  "What does a compiler optimization pass try to improve?",
  "Which keyword starts a for-loop in Python?",
  "Is 2 an even number?",
  "What is the sorted order of c, a, b?",
  "What does CPU stand for?",
]

def _row(idx:int, prompt:str, target:str, eval_every:int) -> dict:
  split = "eval" if idx % eval_every == 0 else "train"
  row_id = f"sentinel_{idx:03d}"
  return {
    "id": row_id,
    "source_id": row_id,
    "split": split,
    "prompt": prompt + "\nReply with the learned sentinel.",
    "completion": target,
    "tags": ["sentinel_override", split],
  }

def _rollout_row(row:dict, target:str) -> dict:
  return {
    "id": row["id"],
    "prompt": row["prompt"],
    "tags": row["tags"],
    "max_tokens": 1,
    "expected_exact": target,
  }

def write_dataset(out:pathlib.Path, *, target:str, eval_every:int, limit:int) -> dict:
  if eval_every < 2: raise ValueError("--eval-every must be >= 2")
  prompts = PROMPTS[:limit]
  if len(prompts) < eval_every: raise ValueError("--limit too small for requested split")
  rows = [_row(idx + 1, prompt, target, eval_every) for idx, prompt in enumerate(prompts)]
  train_rows, eval_rows = [row for row in rows if row["split"] == "train"], [row for row in rows if row["split"] == "eval"]
  out.mkdir(parents=True, exist_ok=True)
  with (out / "sft.jsonl").open("w") as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")
  with (out / "eval-prompts.jsonl").open("w") as f:
    for row in eval_rows: f.write(json.dumps(_rollout_row(row, target), sort_keys=True) + "\n")
  summary = {
    "kind": "llm_adapter_signal_dataset",
    "target": target,
    "rows": len(rows),
    "train_rows": len(train_rows),
    "eval_rows": len(eval_rows),
    "eval_every": eval_every,
    "files": {"sft": "sft.jsonl", "eval_prompts": "eval-prompts.jsonl"},
    "note": "Synthetic sentinel-override data: useful for proving adapter behavior change, not model capability.",
  }
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(
    "# Adapter Signal Dataset V2\n\n"
    "This dataset gives the output-LoRA path a real supervised signal: ordinary\n"
    f"question prompts must answer with the one-token sentinel `{target}`. The base\n"
    "model should fail the held-out exact-match rollout; a trained adapter should\n"
    "learn the override. This is a behavior-change plumbing gate, not a capability\n"
    "benchmark.\n\n"
    f"- rows: `{len(rows)}`\n"
    f"- train rows: `{len(train_rows)}`\n"
    f"- eval rows: `{len(eval_rows)}`\n"
    f"- target: `{target}`\n"
  )
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Build a small real-signal adapter SFT/eval dataset")
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--target", default="OK")
  parser.add_argument("--eval-every", type=int, default=5)
  parser.add_argument("--limit", type=int, default=len(PROMPTS))
  args = parser.parse_args()
  summary = write_dataset(args.out, target=args.target, eval_every=args.eval_every, limit=args.limit)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
