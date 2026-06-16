#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, time
from collections import defaultdict
from typing import Any

from extra.llm_adapter_json_data_v4 import _completion
from extra.llm_eval_common import build_prompt_ids, quality_summary, score_prompt
from extra.llm_sft_smoke_train import load_sft_rows

from extra.llm_eval_common import write_jsonl as _jsonl
from extra.qk_modes import prompt_format_choices

def _read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  if not path.exists(): return []
  rows = []
  seen: set[str] = set()
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    if not raw.strip(): continue
    try:
      row = json.loads(raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id: raise ValueError(f"{path}:{lineno}: missing string id")
    if row_id in seen: raise ValueError(f"{path}:{lineno}: duplicate id {row_id!r}")
    seen.add(row_id)
    rows.append(row)
  return rows

def _split_source_rows(rows:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  train_rows = [row for row in rows if row.get("split") == "train"]
  eval_rows = [row for row in rows if row.get("split") == "eval"]
  if not train_rows or not eval_rows: raise ValueError("source SFT rows must contain non-empty split=train and split=eval rows")
  if {row["source_id"] for row in train_rows} & {row["source_id"] for row in eval_rows}:
    raise ValueError("train/eval source_id overlap")
  return train_rows, eval_rows

def _prompt_row(row:dict[str, Any], default_max_tokens:int=24) -> dict[str, Any]:
  expected = row.get("expected_json")
  if not isinstance(expected, dict) or set(expected) != {"answer"}:
    raise ValueError(f"{row.get('id')}: expected_json must be exactly one answer key")
  return {
    "id": row["id"],
    "prompt": row["prompt"],
    "tags": row.get("tags", []),
    "max_tokens": row.get("max_tokens", default_max_tokens),
    "expected_json": expected,
  }

def _is_near_miss(score:dict[str, Any]) -> bool:
  axes = score.get("json_axes", {}).get("axes", {})
  return all(axes.get(axis) is True for axis in ("parse_valid", "no_extra_text", "schema_ok", "type_ok")) and axes.get("value_correct") is False

def _sample_plan(train_rows:list[dict[str, Any]], include_categories:list[str]) -> list[tuple[int, dict[str, Any]]]:
  if not include_categories: return list(enumerate(train_rows))
  requested = set(include_categories)
  known = {row.get("category") for row in train_rows}
  missing = sorted(category for category in requested if category not in known)
  if missing: raise ValueError(f"--sample-categories contains unknown categories: {missing}")
  plan = [(idx, row) for idx, row in enumerate(train_rows) if row.get("category") in requested]
  if not plan: raise ValueError("--sample-categories matched no train rows")
  return plan

def _sft_train_row(sample:dict[str, Any], rank:int) -> dict[str, Any]:
  return {
    "id": f"{sample['source_id']}:rs{rank:02d}",
    "source_id": sample["source_id"],
    "split": "train",
    "category": sample.get("category"),
    "prompt": sample["prompt"],
    "completion": sample["text"].strip(),
    "expected_json": sample["expected_json"],
    "normalized_answer": sample.get("normalized_answer"),
    "tags": sorted(set((sample.get("tags") or []) + ["rs_accepted", "train"])),
    "max_tokens": sample.get("max_tokens", 24),
    "sample_id": sample["id"],
    "sample_temperature": sample["temperature"],
    "sample_seed": sample["seed"],
  }

def _sft_eval_row(row:dict[str, Any]) -> dict[str, Any]:
  return {
    "id": row["id"],
    "source_id": row["source_id"],
    "split": "eval",
    "category": row.get("category"),
    "prompt": row["prompt"],
    "completion": row.get("completion") or _completion(row["expected_json"]["answer"]),
    "expected_json": row["expected_json"],
    "normalized_answer": row.get("normalized_answer"),
    "tags": row.get("tags", []),
    "max_tokens": row.get("max_tokens", 24),
  }

def build_rejection_dataset(samples:list[dict[str, Any]], train_rows:list[dict[str, Any]], eval_rows:list[dict[str, Any]],
                            *, max_accepted_per_source:int=1) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  if max_accepted_per_source < 1: raise ValueError("max_accepted_per_source must be >= 1")
  train_ids = {row["id"] for row in train_rows}
  eval_ids = {row["id"] for row in eval_rows}
  if train_ids & eval_ids: raise ValueError("train/eval id overlap")
  bad_samples = [sample["id"] for sample in samples if sample.get("source_id") not in train_ids]
  if bad_samples: raise ValueError(f"samples from non-train source ids: {bad_samples[:5]}")

  accepted = [sample for sample in samples if sample.get("score", {}).get("passed") is True]
  near_miss = [sample for sample in samples if _is_near_miss(sample.get("score", {}))]
  by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for sample in accepted: by_source[sample["source_id"]].append(sample)
  selected_train: list[dict[str, Any]] = []
  for source_id in sorted(by_source):
    ranked = sorted(by_source[source_id], key=lambda row: (row.get("generated", 10**9), row.get("sample_idx", 10**9), row["id"]))
    for rank, sample in enumerate(ranked[:max_accepted_per_source], 1):
      selected_train.append(_sft_train_row(sample, rank))
  sft_rows = selected_train + [_sft_eval_row(row) for row in eval_rows]

  source_by_id = {row["id"]: row for row in train_rows}
  attempts_by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "accepted_attempts": 0, "selected_train_rows": 0, "near_miss": 0})
  axes_by_category: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"passed": 0, "scored": 0}))
  attempts_by_temperature: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "accepted_attempts": 0, "near_miss": 0})
  for sample in samples:
    category = source_by_id[sample["source_id"]].get("category", "unknown")
    attempts_by_category[category]["attempts"] += 1
    if sample in accepted: attempts_by_category[category]["accepted_attempts"] += 1
    if sample in near_miss: attempts_by_category[category]["near_miss"] += 1
    for axis, passed in sample.get("score", {}).get("json_axes", {}).get("axes", {}).items():
      axes_by_category[category][axis]["scored"] += 1
      if passed is True: axes_by_category[category][axis]["passed"] += 1
    temp = str(sample.get("temperature"))
    attempts_by_temperature[temp]["attempts"] += 1
    if sample in accepted: attempts_by_temperature[temp]["accepted_attempts"] += 1
    if sample in near_miss: attempts_by_temperature[temp]["near_miss"] += 1
  for row in selected_train:
    attempts_by_category[row.get("category", "unknown")]["selected_train_rows"] += 1
  summary = {
    "kind": "llm_json_rejection_sample_summary",
    "train_source_rows": len(train_rows),
    "eval_rows": len(eval_rows),
    "attempts": len(samples),
    "accepted_attempts": len(accepted),
    "near_miss": len(near_miss),
    "selected_train_rows": len(selected_train),
    "max_accepted_per_source": max_accepted_per_source,
    "sft_rows": len(sft_rows),
    "quality": quality_summary(samples),
    "categories": {k: v for k, v in sorted(attempts_by_category.items())},
    "category_axes": {category: {axis: counts for axis, counts in sorted(axes.items())} for category, axes in sorted(axes_by_category.items())},
    "temperatures_summary": {k: v for k, v in sorted(attempts_by_temperature.items(), key=lambda item: float(item[0]))},
    "integrity": {
      "samples_from_train_only": True,
      "eval_rows_from_gold_source": True,
      "train_eval_source_overlap": 0,
    },
  }
  return accepted, near_miss, sft_rows, summary

def summary_markdown(summary:dict[str, Any]) -> str:
  lines = [
    "# JSON Rejection-Sampling Data",
    "",
    "This artifact samples completions from the selected adapter on source",
    "train prompts, keeps strict JSON passes as SFT rows, and carries the",
    "source eval rows only for trainer diagnostics. Held-out promotion should",
    "use the separate rollout gate for the matching source dataset.",
    "",
    "## Summary",
    "",
    f"- attempts: `{summary['attempts']}`",
    f"- accepted attempts: `{summary['accepted_attempts']}`",
    f"- selected train rows: `{summary['selected_train_rows']}`",
    f"- eval rows: `{summary['eval_rows']}`",
    f"- strict pass: `{summary['quality']['passed']}/{summary['quality']['scored']}`",
    "",
    "| category | attempts | accepted attempts | selected train | near miss |",
    "|---|---:|---:|---:|---:|",
  ]
  for category, row in summary["categories"].items():
    lines.append(f"| `{category}` | {row['attempts']} | {row['accepted_attempts']} | {row['selected_train_rows']} | {row['near_miss']} |")
  if summary.get("sample_categories"):
    lines += [
      "",
      f"- sampled categories this run: `{', '.join(summary['sample_categories'])}`",
      f"- sampled train rows this run: `{summary.get('sample_train_rows')}`",
    ]
  if summary.get("category_axes"):
    lines += ["", "## Category JSON Axes", "", "| category | parse | schema | type | value | strict |", "|---|---:|---:|---:|---:|---:|"]
    for category, axes in summary["category_axes"].items():
      def cell(axis:str) -> str:
        row = axes.get(axis, {"passed": 0, "scored": 0})
        return f"{row['passed']}/{row['scored']}"
      lines.append(f"| `{category}` | {cell('parse_valid')} | {cell('schema_ok')} | {cell('type_ok')} | {cell('value_correct')} | {cell('strict_pass')} |")
  if summary.get("temperatures_summary"):
    lines += ["", "## Temperature Summary", "", "| temperature | attempts | accepted | near miss |", "|---|---:|---:|---:|"]
    for temperature, row in summary["temperatures_summary"].items():
      lines.append(f"| `{temperature}` | {row['attempts']} | {row['accepted_attempts']} | {row['near_miss']} |")
  lines.append("")
  return "\n".join(lines)

def _configure_env(args:argparse.Namespace) -> None:
  os.environ["DEV"] = args.device
  os.environ["JIT"] = "1"
  os.environ["PYTHONPATH"] = "."
  os.environ["QK_PRIMITIVE_STORAGE"] = args.storage
  os.environ["Q4K_PRIMITIVE"] = "0"
  os.environ["Q6K_PRIMITIVE"] = "0"
  os.environ["QK_GENERATED_POLICY"] = str(args.policy)
  os.environ.pop("QK_GENERATED_POLICY_DEBUG", None)
  os.environ.pop("Q4K_PRIMITIVE_DEBUG", None)
  os.environ.pop("Q6K_PRIMITIVE_DEBUG", None)

def run_sampling(args:argparse.Namespace) -> dict[str, Any]:
  _configure_env(args)
  from tinygrad import Tensor
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer
  from extra.llm_adapter import load_adapter

  rows = load_sft_rows(args.input)
  train_rows, eval_rows = _split_source_rows(rows)
  if args.limit_train_rows > 0: train_rows = train_rows[:args.limit_train_rows]
  sample_plan = _sample_plan(train_rows, args.sample_categories)
  temperatures = args.temperatures
  if len(temperatures) != args.k:
    raise ValueError(f"--temperatures count ({len(temperatures)}) must match --k ({args.k})")
  args.out.mkdir(parents=True, exist_ok=True)
  samples_path = args.out / "samples.jsonl"
  samples: list[dict[str, Any]] = _read_jsonl(samples_path) if args.resume else []
  seen_sample_ids = {row["id"] for row in samples}
  model, kv = Transformer.from_gguf(pathlib.Path(args.model).expanduser(), args.max_context)
  load_adapter(model, pathlib.Path(args.adapter).expanduser())
  tok = SimpleTokenizer.from_gguf_kv(kv)
  st_all = time.perf_counter()
  with samples_path.open("a" if args.resume else "w") as f:
    for row_idx, row in sample_plan:
      prompt_row = _prompt_row(row, args.tokens)
      prompt_ids = build_prompt_ids(tok, row["prompt"], args.prompt_format)
      for sample_idx, temperature in enumerate(temperatures):
        sample_id = f"{row['id']}:sample{sample_idx:02d}"
        if sample_id in seen_sample_ids: continue
        sample_seed = args.seed + row_idx * 1009 + sample_idx
        Tensor.manual_seed(sample_seed)
        out: list[int] = []
        st = time.perf_counter()
        for tid in model.generate(prompt_ids, temperature=temperature):
          if tok.is_end(tid): break
          out.append(tid)
          if len(out) >= prompt_row["max_tokens"]: break
        elapsed = time.perf_counter() - st
        text = tok.decode(out)
        sample = {
          "id": sample_id,
          "source_id": row["id"],
          "split": "train",
          "category": row.get("category"),
          "prompt": row["prompt"],
          "prompt_len": len(prompt_ids),
          "tags": row.get("tags", []),
          "max_tokens": prompt_row["max_tokens"],
          "expected_json": prompt_row["expected_json"],
          "normalized_answer": row.get("normalized_answer"),
          "sample_idx": sample_idx,
          "temperature": temperature,
          "seed": sample_seed,
          "tokens": out,
          "text": text,
          "generated": len(out),
          "elapsed_s": round(elapsed, 6),
          "tok_s": 0.0 if elapsed == 0 else len(out) / elapsed,
        }
        sample["score"] = score_prompt(prompt_row, text)
        samples.append(sample)
        seen_sample_ids.add(sample_id)
        f.write(json.dumps(sample, sort_keys=True) + "\n")
        f.flush()
  # samples.jsonl is written incrementally above so a device timeout does not
  # discard completed attempts. Re-read it before deriving downstream artifacts.
  samples = _read_jsonl(samples_path)
  accepted, near_miss, sft_rows, summary = build_rejection_dataset(
    samples, train_rows, eval_rows, max_accepted_per_source=args.max_accepted_per_source)
  summary.update({
    "model": str(args.model),
    "policy": str(args.policy),
    "adapter": str(args.adapter),
    "input": str(args.input),
    "storage": args.storage,
    "prompt_format": args.prompt_format,
    "seed": args.seed,
    "k": args.k,
    "temperatures": temperatures,
    "sample_categories": args.sample_categories,
    "sample_train_rows": len(sample_plan),
    "skipped_train_rows": len(train_rows) - len(sample_plan),
    "elapsed_s": time.perf_counter() - st_all,
    "files": {"samples": "samples.jsonl", "accepted": "accepted.jsonl", "near_miss": "near-miss.jsonl", "sft": "sft.jsonl"},
  })
  _jsonl(args.out / "accepted.jsonl", accepted)
  _jsonl(args.out / "near-miss.jsonl", near_miss)
  _jsonl(args.out / "sft.jsonl", sft_rows)
  (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (args.out / "README.md").write_text(summary_markdown(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Build strict-JSON rejection-sampling SFT data")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--policy", type=pathlib.Path, required=True)
  parser.add_argument("--adapter", type=pathlib.Path, required=True)
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--storage", default="shared")
  parser.add_argument("--prompt-format", choices=prompt_format_choices(), default="chat")
  parser.add_argument("--tokens", type=int, default=24)
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=20260614)
  parser.add_argument("--k", type=int, default=4)
  parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.2, 0.5, 0.8])
  parser.add_argument("--max-accepted-per-source", type=int, default=1)
  parser.add_argument("--limit-train-rows", type=int, default=0)
  parser.add_argument("--sample-categories", nargs="*", default=[], help="only append samples for these train categories while summarizing all train rows")
  parser.add_argument("--resume", action="store_true", help="append missing samples to an existing samples.jsonl")
  args = parser.parse_args()
  summary = run_sampling(args)
  print(summary_markdown(summary))
  return 0 if summary["selected_train_rows"] > 0 else 1

if __name__ == "__main__":
  raise SystemExit(main())
