#!/usr/bin/env python3
from __future__ import annotations

import argparse, collections, json, pathlib, statistics
from typing import Any

from extra.llm_eval_common import load_json as _load_json, md_text, read_id_jsonl

def _display_path(value:Any) -> Any:
  if not isinstance(value, str): return value
  home = str(pathlib.Path.home())
  return value.replace(home + "/", "~/")

def _read_rollout_rows(path:pathlib.Path) -> list[dict[str, Any]]:
  rows_path = path / "rollouts.jsonl"
  if not rows_path.exists(): raise ValueError(f"{path}: missing rollouts.jsonl")
  rows = read_id_jsonl(rows_path)
  if not rows: raise ValueError(f"{rows_path}: no rollout rows")
  return rows

def _load_artifact(path:pathlib.Path) -> dict[str, Any]:
  summary_path = path / "summary.json"
  if not summary_path.exists(): raise ValueError(f"{path}: missing summary.json")
  summary = _load_json(summary_path)
  if summary.get("kind") != "llm_rollout_summary": raise ValueError(f"{summary_path}: expected kind=llm_rollout_summary")
  rows = _read_rollout_rows(path)
  if summary.get("prompts") != len(rows):
    raise ValueError(f"{path}: summary prompts={summary.get('prompts')} but rollouts has {len(rows)} rows")
  return {"path": path, "summary": summary, "rows": rows}

def _reason(row:dict[str, Any], *, min_completion_tokens:int, max_total_tokens:int | None, require_quality_pass:bool) -> str | None:
  if require_quality_pass and row.get("score", {}).get("status") != "pass": return "quality_not_pass"
  if not str(row.get("text", "")).strip(): return "empty_completion"
  if int(row.get("generated", 0)) < min_completion_tokens: return "completion_too_short"
  if max_total_tokens is not None and int(row.get("prompt_len", 0)) + int(row.get("generated", 0)) > max_total_tokens:
    return "too_long"
  return None

def build_training_data(artifacts:list[pathlib.Path], *, min_completion_tokens:int=1, max_total_tokens:int | None=4096,
                        require_quality_pass:bool=True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  loaded = [_load_artifact(path) for path in artifacts]
  out_rows: list[dict[str, Any]] = []
  filter_reasons: collections.Counter[str] = collections.Counter()
  tag_counts: collections.Counter[str] = collections.Counter()
  prompt_lens: list[int] = []
  completion_lens: list[int] = []
  input_rows = 0
  for artifact in loaded:
    summary = artifact["summary"]
    label = artifact["path"].name
    for row in artifact["rows"]:
      input_rows += 1
      if (reason := _reason(row, min_completion_tokens=min_completion_tokens, max_total_tokens=max_total_tokens,
                            require_quality_pass=require_quality_pass)) is not None:
        filter_reasons[reason] += 1
        continue
      prompt, completion = row["prompt"], row["text"]
      tags = row.get("tags") or []
      for tag in tags or ["untagged"]: tag_counts[tag] += 1
      prompt_lens.append(int(row.get("prompt_len", 0)))
      completion_lens.append(int(row.get("generated", 0)))
      out_rows.append({
        "id": f"{label}:{row['id']}",
        "source_artifact": str(artifact["path"]),
        "source_id": row["id"],
        "model": _display_path(summary["model"]),
        "policy": _display_path(summary.get("policy")),
        "mode": summary["mode"],
        "storage": summary["storage"],
        "prompt_format": row.get("prompt_format", summary.get("prompt_format")),
        "prompt": prompt,
        "completion": completion,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        "tags": tags,
        "prompt_len": int(row.get("prompt_len", 0)),
        "completion_tokens": int(row.get("generated", 0)),
        "score": row.get("score", {"status": "unscored", "passed": None, "checks": []}),
      })
  summary = {
    "kind": "llm_training_data_probe",
    "source_artifacts": [str(path) for path in artifacts],
    "input_rows": input_rows,
    "exported_rows": len(out_rows),
    "filtered_rows": input_rows - len(out_rows),
    "filter_reasons": dict(sorted(filter_reasons.items())),
    "min_completion_tokens": min_completion_tokens,
    "max_total_tokens": max_total_tokens,
    "require_quality_pass": require_quality_pass,
    "avg_prompt_len": None if not prompt_lens else statistics.fmean(prompt_lens),
    "avg_completion_tokens": None if not completion_lens else statistics.fmean(completion_lens),
    "max_prompt_len": None if not prompt_lens else max(prompt_lens),
    "max_completion_tokens": None if not completion_lens else max(completion_lens),
    "tag_distribution": dict(sorted(tag_counts.items())),
  }
  return out_rows, summary

def summary_markdown(summary:dict[str, Any], rows:list[dict[str, Any]]) -> str:
  lines = [
    "# LLM Training Data Probe",
    "",
    "This is a dry-run exporter from rollout artifacts to SFT-style rows. It",
    "validates dataset shape and filtering only; it is not a training loop.",
    "",
    "## Summary",
    "",
    f"- input rows: `{summary['input_rows']}`",
    f"- exported rows: `{summary['exported_rows']}`",
    f"- filtered rows: `{summary['filtered_rows']}`",
    f"- average prompt tokens: `{summary['avg_prompt_len']:.2f}`" if summary["avg_prompt_len"] is not None else "- average prompt tokens: `n/a`",
    f"- average completion tokens: `{summary['avg_completion_tokens']:.2f}`" if summary["avg_completion_tokens"] is not None else "- average completion tokens: `n/a`",
    "",
    "## Tag Distribution",
    "",
    "| tag | rows |",
    "|---|---:|",
  ]
  for tag, count in summary["tag_distribution"].items():
    lines.append(f"| `{tag}` | {count} |")
  lines += ["", "## Sample Rows", "", "| id | tags | prompt | completion |", "|---|---|---|---|"]
  for row in rows[:10]:
    lines.append(f"| `{row['id']}` | `{','.join(row.get('tags') or [])}` | {md_text(row['prompt'])} | {md_text(row['completion'])} |")
  lines.append("")
  return "\n".join(lines)

def write_probe(out:pathlib.Path, rows:list[dict[str, Any]], summary:dict[str, Any]) -> None:
  out.mkdir(parents=True, exist_ok=True)
  with (out / "sft.jsonl").open("w") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(summary_markdown(summary, rows))

def main() -> int:
  parser = argparse.ArgumentParser(description="Dry-run exporter from LLM rollout artifacts to SFT-style training rows")
  parser.add_argument("artifacts", nargs="+", type=pathlib.Path)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--min-completion-tokens", type=int, default=1)
  parser.add_argument("--max-total-tokens", type=int, default=4096)
  parser.add_argument("--allow-quality-fail", action="store_true")
  args = parser.parse_args()
  if args.min_completion_tokens < 1: raise ValueError("--min-completion-tokens must be >= 1")
  max_total = None if args.max_total_tokens <= 0 else args.max_total_tokens
  rows, summary = build_training_data(args.artifacts, min_completion_tokens=args.min_completion_tokens,
                                      max_total_tokens=max_total, require_quality_pass=not args.allow_quality_fail)
  write_probe(args.out, rows, summary)
  print(summary_markdown(summary, rows))
  return 0 if rows else 1

if __name__ == "__main__":
  raise SystemExit(main())
