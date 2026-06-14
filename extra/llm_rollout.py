#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys, time
from typing import Any

from extra.llm_eval_common import build_prompt_ids, md_text, quality_summary, read_prompt_jsonl, score_prompt

def _rate_ci(row:dict[str, Any]) -> str:
  if row.get("pass_rate") is None: return "n/a"
  ci = row.get("ci95") or {}
  if ci.get("low") is None or ci.get("high") is None: return f"{row['pass_rate']:.2f}"
  return f"{row['pass_rate']:.2f} [{ci['low']:.2f}, {ci['high']:.2f}]"

def configure_env(args:argparse.Namespace) -> None:
  os.environ["DEV"] = args.device
  os.environ["JIT"] = "1"
  os.environ["PYTHONPATH"] = "."
  os.environ["QK_PRIMITIVE_STORAGE"] = args.storage
  os.environ.pop("QK_GENERATED_POLICY", None)
  os.environ.pop("QK_GENERATED_POLICY_DEBUG", None)
  os.environ.pop("Q4K_PRIMITIVE_DEBUG", None)
  os.environ.pop("Q6K_PRIMITIVE_DEBUG", None)
  if args.mode == "generated":
    if args.policy is None: raise ValueError("--policy is required for mode=generated")
    os.environ["Q4K_PRIMITIVE"] = "0"
    os.environ["Q6K_PRIMITIVE"] = "0"
    os.environ["QK_GENERATED_POLICY"] = str(args.policy)
    if args.policy_debug: os.environ["QK_GENERATED_POLICY_DEBUG"] = "1"
  elif args.mode == "explicit":
    os.environ["Q4K_PRIMITIVE"] = "1"
    os.environ["Q6K_PRIMITIVE"] = "1"
  elif args.mode == "baseline":
    os.environ["Q4K_PRIMITIVE"] = "0"
    os.environ["Q6K_PRIMITIVE"] = "0"
  else:
    raise ValueError(f"unknown mode {args.mode!r}")

def summarize_rollouts(args:argparse.Namespace, rows:list[dict[str, Any]]) -> dict[str, Any]:
  elapsed = sum(row["elapsed_s"] for row in rows)
  generated = sum(row["generated"] for row in rows)
  summary = {
    "kind": "llm_rollout_summary",
    "mode": args.mode,
    "model": str(args.model),
    "policy": str(args.policy) if args.policy else None,
    "dataset": str(args.dataset),
    "storage": args.storage,
    "prompt_format": args.prompt_format,
    "temperature": args.temperature,
    "seed": args.seed,
    "prompts": len(rows),
    "generated": generated,
    "elapsed_s": elapsed,
    "tok_s": 0.0 if elapsed == 0 else generated / elapsed,
    "quality": quality_summary(rows),
  }
  if getattr(args, "adapter", None) is not None: summary["adapter"] = str(args.adapter)
  return summary

def summary_markdown(summary:dict[str, Any], rows:list[dict[str, Any]]) -> str:
  lines = [
    "# LLM Rollout Summary",
    "",
    "This is a dataset-style rollout artifact. It is useful for practical eval,",
    "future SFT/RLVR data generation, and compiler/search behavior gates. Timing",
    "is eval-loop sanity data, not the canonical decode benchmark.",
    "",
    "## Summary",
    "",
    f"- mode: `{summary['mode']}`",
    f"- model: `{summary['model']}`",
    f"- policy: `{summary['policy']}`",
    f"- dataset: `{summary['dataset']}`",
    f"- storage: `{summary['storage']}`",
    f"- prompt format: `{summary['prompt_format']}`",
    f"- prompts: `{summary['prompts']}`",
    f"- generated tokens: `{summary['generated']}`",
    f"- quality: `{summary['quality']['status']}` ({summary['quality']['passed']}/{summary['quality']['scored']})",
    "",
    "| id | tags | quality | generated | tok/s | text |",
    "|---|---|---:|---:|---:|---|",
  ]
  if summary.get("adapter") is not None: lines.insert(11, f"- adapter: `{summary['adapter']}`")
  for row in rows:
    lines.append(
      f"| `{row['id']}` | `{','.join(row.get('tags') or [])}` | `{row['score']['status']}` | "
      f"{row['generated']} | {row['tok_s']:.2f} | {md_text(row['text'])} |"
    )
  lines += ["", "## Quality By Tag", "", "| tag | passed | scored | pass rate |", "|---|---:|---:|---:|"]
  for tag, row in summary["quality"]["tags"].items():
    lines.append(f"| `{tag}` | {row['passed']} | {row['scored']} | {row['pass_rate']:.2f} |")
  if "json_axes" in summary["quality"]:
    lines += ["", "## JSON Quality Axes", "", "| axis | passed | scored | pass rate [95% CI] |", "|---|---:|---:|---:|"]
    for axis, row in summary["quality"]["json_axes"]["axes"].items():
      lines.append(f"| `{axis}` | {row['passed']} | {row['scored']} | {_rate_ci(row)} |")
  lines.append("")
  return "\n".join(lines)

def write_artifacts(out:pathlib.Path, rows:list[dict[str, Any]], summary:dict[str, Any]) -> None:
  out.mkdir(parents=True, exist_ok=True)
  with (out / "rollouts.jsonl").open("w") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "summary.md").write_text(summary_markdown(summary, rows))

def run_rollout(args:argparse.Namespace) -> int:
  configure_env(args)
  from tinygrad import Tensor
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

  Tensor.manual_seed(args.seed)
  dataset = read_prompt_jsonl(args.dataset)
  model, kv = Transformer.from_gguf(pathlib.Path(args.model).expanduser(), args.max_context)
  if args.adapter is not None:
    from extra.llm_adapter import load_adapter
    load_adapter(model, pathlib.Path(args.adapter).expanduser())
  tok = SimpleTokenizer.from_gguf_kv(kv)
  rows: list[dict[str, Any]] = []
  for item in dataset:
    ids = build_prompt_ids(tok, item["prompt"], args.prompt_format)
    out: list[int] = []
    max_tokens = item.get("max_tokens", args.tokens)
    st = time.perf_counter()
    for tid in model.generate(ids, temperature=args.temperature):
      if tok.is_end(tid): break
      out.append(tid)
      if len(out) >= max_tokens: break
    elapsed = time.perf_counter() - st
    text = tok.decode(out)
    result = {
      "id": item["id"], "mode": args.mode, "model": str(args.model), "policy": str(args.policy) if args.policy else None,
      "prompt": item["prompt"], "prompt_len": len(ids), "prompt_format": args.prompt_format,
      "tags": item.get("tags", []), "max_tokens": max_tokens, "tokens": out, "text": text,
      "generated": len(out), "elapsed_s": round(elapsed, 6), "tok_s": 0.0 if elapsed == 0 else len(out) / elapsed,
      "score": score_prompt(item, text),
    }
    if args.adapter is not None: result["adapter"] = str(args.adapter)
    rows.append(result)
  summary = summarize_rollouts(args, rows)
  write_artifacts(args.out, rows, summary)
  print(summary_markdown(summary, rows))
  return 1 if args.fail_on_quality and summary["quality"]["status"] == "fail" else 0

def _load_manifest(path:pathlib.Path) -> dict[str, Any]:
  data = json.loads(path.read_text())
  if data.get("kind") != "llm_rollout_manifest": raise ValueError(f"{path}: expected kind=llm_rollout_manifest")
  if not isinstance(data.get("rows"), list) or not data["rows"]: raise ValueError(f"{path}: expected non-empty rows")
  seen: set[str] = set()
  for idx, row in enumerate(data["rows"]):
    if not isinstance(row, dict): raise ValueError(f"{path}: row {idx} must be an object")
    for key in ("id", "model", "dataset", "out", "mode"):
      if not isinstance(row.get(key), str) or not row[key]: raise ValueError(f"{path}: row {idx} missing string {key}")
    if "adapter" in row and row["adapter"] is not None and not isinstance(row["adapter"], str):
      raise ValueError(f"{path}: row {idx} adapter must be a string")
    if row["id"] in seen: raise ValueError(f"{path}: duplicate row id {row['id']!r}")
    seen.add(row["id"])
  return data

def _cmd_path(value:str) -> str:
  path = pathlib.Path(value)
  if value.startswith("~") or path.is_absolute(): return str(path.expanduser())
  return value

def run_manifest(args:argparse.Namespace) -> int:
  manifest = _load_manifest(args.manifest)
  selected = set(args.only)
  matched = False
  rc = 0
  for row in manifest["rows"]:
    if selected and row["id"] not in selected: continue
    if row.get("enabled", True) is False and not args.include_disabled: continue
    matched = True
    out = pathlib.Path(row["out"])
    if args.reuse and (out / "summary.json").exists():
      print(f"reuse {row['id']}: {out / 'summary.json'}")
      continue
    cmd = [
      sys.executable, __file__,
      "--model", _cmd_path(row["model"]),
      "--dataset", _cmd_path(row["dataset"]),
      "--out", _cmd_path(row["out"]),
      "--mode", row["mode"],
      "--tokens", str(row.get("tokens", manifest.get("tokens", 64))),
      "--max-context", str(row.get("max_context", manifest.get("max_context", 4096))),
      "--temperature", str(row.get("temperature", manifest.get("temperature", 0.0))),
      "--seed", str(row.get("seed", manifest.get("seed", 20260612))),
      "--device", str(row.get("device", manifest.get("device", "AMD"))),
      "--storage", str(row.get("storage", manifest.get("storage", "shared"))),
      "--prompt-format", str(row.get("prompt_format", manifest.get("prompt_format", "chat"))),
    ]
    if row.get("policy"): cmd += ["--policy", _cmd_path(row["policy"])]
    if row.get("adapter"): cmd += ["--adapter", _cmd_path(row["adapter"])]
    if args.fail_on_quality or row.get("fail_on_quality", manifest.get("fail_on_quality", False)): cmd.append("--fail-on-quality")
    if args.policy_debug or row.get("policy_debug", manifest.get("policy_debug", False)): cmd.append("--policy-debug")
    print(f"run {row['id']}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
      rc = proc.returncode
      if not args.keep_going: return rc
  if selected and not matched:
    raise ValueError(f"no manifest rows matched {sorted(selected)}")
  return rc

def main() -> int:
  parser = argparse.ArgumentParser(description="Dataset-style LLM rollout runner")
  parser.add_argument("--manifest", type=pathlib.Path, help="run rows from an llm_rollout_manifest")
  parser.add_argument("--only", nargs="*", default=[], help="manifest row ids to run")
  parser.add_argument("--include-disabled", action="store_true")
  parser.add_argument("--keep-going", action="store_true")
  parser.add_argument("--reuse", action="store_true")
  parser.add_argument("--model", type=pathlib.Path)
  parser.add_argument("--policy", type=pathlib.Path)
  parser.add_argument("--adapter", type=pathlib.Path)
  parser.add_argument("--dataset", type=pathlib.Path)
  parser.add_argument("--out", type=pathlib.Path)
  parser.add_argument("--mode", choices=("generated", "explicit", "baseline"), default="generated")
  parser.add_argument("--tokens", type=int, default=64)
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=20260612)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--storage", default="shared")
  parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
  parser.add_argument("--policy-debug", action="store_true")
  parser.add_argument("--fail-on-quality", action="store_true")
  args = parser.parse_args()

  if args.manifest is not None: return run_manifest(args)
  for name in ("model", "dataset", "out"):
    if getattr(args, name) is None: parser.error(f"--{name.replace('_', '-')} is required without --manifest")
  return run_rollout(args)

if __name__ == "__main__":
  raise SystemExit(main())
