#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, subprocess, sys, time
from typing import Any

from extra.llm_eval_common import md_text, quality_summary, read_prompt_jsonl as _read_jsonl, score_prompt
from extra.llm_generate import child_env, generate_one, load_model_and_tokenizer
from extra.qk_modes import eval_run_mode_choices, prompt_format_choices

DEFAULT_TIMEOUT = 1800.0

def _json_from_output(out:str) -> dict[str, Any]:
  dict_lines:list[tuple[int, dict[str, Any]]] = []
  for lineno, line in enumerate(out.strip().splitlines(), start=1):
    try:
      data = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(data, dict): dict_lines.append((lineno, data))
  summaries = [(lineno, data) for lineno, data in dict_lines if {"mode", "results", "tok_s"}.issubset(data)]
  if not summaries:
    if dict_lines:
      raise RuntimeError("child produced JSON dict lines, but none matched the eval summary schema\n" + out[-4000:])
    raise RuntimeError("child produced no JSON line\n" + out[-4000:])
  summary_lineno, summary = summaries[-1]
  trailing = [lineno for lineno, _ in dict_lines if lineno > summary_lineno]
  if trailing:
    raise RuntimeError(f"child produced JSON dict line(s) after eval summary at lines {trailing}\n" + out[-4000:])
  return summary

def _env_for_mode(args:argparse.Namespace, mode:str) -> dict[str, str]:
  return child_env(mode, device=args.device, storage=args.storage, policy=args.policy, policy_debug=args.policy_debug)

def _child_cmd(args:argparse.Namespace, mode:str) -> list[str]:
  cmd = [
    sys.executable, __file__, "--child", "--run-mode", mode,
    "--model", str(args.model), "--prompts", str(args.prompts),
    "--tokens", str(args.tokens), "--max-context", str(args.max_context),
    "--seed", str(args.seed), "--temperature", str(args.temperature),
    "--storage", args.storage, "--prompt-format", args.prompt_format,
  ]
  if args.policy is not None: cmd += ["--policy", str(args.policy)]
  return cmd

def _run_mode(args:argparse.Namespace, mode:str) -> dict[str, Any]:
  args.out.mkdir(parents=True, exist_ok=True)
  st = time.perf_counter()
  proc = subprocess.run(_child_cmd(args, mode), cwd=args.repo, env=_env_for_mode(args, mode), text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
  elapsed = time.perf_counter() - st
  (args.out / f"{mode}.log").write_text(proc.stdout)
  data = _json_from_output(proc.stdout)
  data["returncode"] = proc.returncode
  data["parent_elapsed_s"] = round(elapsed, 3)
  data["tail"] = "\n".join(proc.stdout.strip().splitlines()[-args.tail_lines:])
  if proc.returncode != 0:
    raise RuntimeError(f"{mode} child failed rc={proc.returncode}\n{proc.stdout[-4000:]}")
  (args.out / f"{mode}.json").write_text(json.dumps(data, indent=2, sort_keys=True))
  return data

def _as_prompt_map(data:dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {row["id"]: row for row in data["results"]}

def summarize_results(explicit:dict[str, Any], generated:dict[str, Any]) -> dict[str, Any]:
  explicit_rows, generated_rows = _as_prompt_map(explicit), _as_prompt_map(generated)
  if set(explicit_rows) != set(generated_rows):
    raise ValueError(f"prompt id mismatch explicit={sorted(explicit_rows)} generated={sorted(generated_rows)}")
  prompt_rows = []
  for prompt_id in sorted(explicit_rows):
    lhs, rhs = explicit_rows[prompt_id], generated_rows[prompt_id]
    prompt_rows.append({
      "id": prompt_id,
      "tokens_match": lhs["tokens"] == rhs["tokens"],
      "explicit_generated": lhs["generated"],
      "generated_generated": rhs["generated"],
      "explicit_tok_s": lhs["tok_s"],
      "generated_tok_s": rhs["tok_s"],
      "tags": rhs.get("tags", []),
      "explicit_score": lhs.get("score", {"status": "unscored", "passed": None, "checks": []}),
      "generated_score": rhs.get("score", {"status": "unscored", "passed": None, "checks": []}),
    })
  modes = {
    "explicit": {"generated": explicit["generated"], "elapsed_s": explicit["elapsed_s"], "tok_s": explicit["tok_s"]},
    "generated": {"generated": generated["generated"], "elapsed_s": generated["elapsed_s"], "tok_s": generated["tok_s"]},
  }
  all_match = all(row["tokens_match"] for row in prompt_rows)
  return {
    "kind": "llm_eval_summary",
    "status": "pass" if all_match else "fail",
    "tokens_match": all_match,
    "model": generated.get("model") or explicit.get("model"),
    "policy": generated.get("policy"),
    "storage": generated.get("storage") or explicit.get("storage"),
    "prompt_format": generated.get("prompt_format") or explicit.get("prompt_format"),
    "prompts": len(prompt_rows),
    "quality": quality_summary(generated["results"]),
    "modes": modes,
    "prompt_rows": prompt_rows,
  }

def summary_markdown(summary:dict[str, Any], explicit:dict[str, Any], generated:dict[str, Any]) -> str:
  generated_rows = _as_prompt_map(generated)
  lines = [
    "# LLM Eval Harness",
    "",
    "This is a rollout/eval parity check for generated-policy inference paths.",
    "It compares explicit Q4/Q6 primitive flags against a generated policy on",
    "fixed prompts with greedy decoding. The correctness gate is exact generated",
    "token parity between the two modes.",
    "",
    "## Summary",
    "",
    f"- status: `{summary['status']}`",
    f"- model: `{summary.get('model')}`",
    f"- policy: `{summary.get('policy')}`",
    f"- storage: `{summary.get('storage')}`",
    f"- prompt format: `{summary.get('prompt_format')}`",
    f"- prompts: `{summary['prompts']}`",
    f"- token parity: `{summary['tokens_match']}`",
    f"- quality status: `{summary['quality']['status']}`",
    f"- quality score: `{summary['quality']['passed']}/{summary['quality']['scored']}`",
    "- timing note: this harness is a rollout correctness gate; first-prompt",
    "  timings include session/JIT effects and are not the canonical decode benchmark.",
    "",
    "| mode | generated tokens | elapsed s | tok/s |",
    "|---|---:|---:|---:|",
  ]
  for mode, row in summary["modes"].items():
    lines.append(f"| `{mode}` | {row['generated']} | {row['elapsed_s']:.3f} | {row['tok_s']:.2f} |")
  lines += [
    "",
    "## Prompt Outputs",
    "",
    "| id | tags | token match | quality | explicit tok/s | generated tok/s | generated text |",
    "|---|---|---:|---:|---:|---:|---|",
  ]
  for row in summary["prompt_rows"]:
    prompt_id = row["id"]
    quality = row["generated_score"]["status"]
    tags = ",".join(row.get("tags") or [])
    lines.append(
      f"| `{prompt_id}` | `{tags}` | `{row['tokens_match']}` | `{quality}` | "
      f"{row['explicit_tok_s']:.2f} | {row['generated_tok_s']:.2f} | {md_text(generated_rows[prompt_id]['text'])} |"
    )
  lines += ["", "## Quality By Tag", "", "| tag | passed | scored | pass rate |", "|---|---:|---:|---:|"]
  for tag, row in summary["quality"]["tags"].items():
    lines.append(f"| `{tag}` | {row['passed']} | {row['scored']} | {row['pass_rate']:.2f} |")
  lines += [
    "",
    "## Training Readiness Verdict",
    "",
    "The faster inference path is ready to be used as a deterministic rollout/eval backend",
    "when the model and policy artifact are pinned. This does not establish a tinygrad",
    "LLM training path; it only validates the decode side needed by future SFT/RLVR",
    "experiments.",
    "",
  ]
  return "\n".join(lines)

def write_artifacts(args:argparse.Namespace, explicit:dict[str, Any], generated:dict[str, Any], summary:dict[str, Any]) -> None:
  with (args.out / "generations.jsonl").open("w") as f:
    for mode, data in (("explicit", explicit), ("generated", generated)):
      for row in data["results"]:
        f.write(json.dumps({"mode": mode, **row}, sort_keys=True) + "\n")
  (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (args.out / "README.md").write_text(summary_markdown(summary, explicit, generated))

def run_child(args:argparse.Namespace) -> None:
  prompts = _read_jsonl(args.prompts)
  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  results = []
  total_tokens = 0
  st_total = time.perf_counter()
  for prompt in prompts:
    gen = generate_one(model, tok, prompt["prompt"], args.prompt_format, temperature=args.temperature, max_tokens=args.tokens)
    total_tokens += gen["generated"]
    results.append({
      "id": prompt["id"], "prompt": prompt["prompt"], "tokens": gen["tokens"], "text": gen["text"],
      "tags": prompt.get("tags", []), "score": score_prompt(prompt, gen["text"]),
      "prompt_len": gen["prompt_len"], "generated": gen["generated"], "elapsed_s": gen["elapsed_s"],
      "tok_s": gen["tok_s"],
    })
  elapsed_total = time.perf_counter() - st_total
  print(json.dumps({
    "mode": args.run_mode, "model": str(args.model), "policy": str(args.policy) if args.policy else None,
    "storage": args.storage, "prompt_format": args.prompt_format, "seed": args.seed,
    "temperature": args.temperature, "max_context": args.max_context,
    "tokens_per_prompt": args.tokens, "prompts": len(prompts), "generated": total_tokens,
    "elapsed_s": round(elapsed_total, 6), "tok_s": 0.0 if elapsed_total == 0 else total_tokens / elapsed_total,
    "results": results,
  }, sort_keys=True))

def run_parent(args:argparse.Namespace) -> int:
  explicit = _run_mode(args, "explicit")
  generated = _run_mode(args, "generated")
  summary = summarize_results(explicit, generated)
  write_artifacts(args, explicit, generated, summary)
  print(summary_markdown(summary, explicit, generated))
  return 0 if summary["status"] == "pass" else 1

def main(argv:list[str] | None=None) -> int:
  parser = argparse.ArgumentParser(description="Small LLM rollout/eval harness for explicit vs generated QK primitive policies")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--policy", type=pathlib.Path, help="QK_GENERATED_POLICY artifact for generated mode")
  parser.add_argument("--prompts", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/llm-eval/run"))
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--tokens", type=int, default=24)
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=20260612)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--storage", default="shared")
  parser.add_argument("--prompt-format", choices=prompt_format_choices(), default="chat")
  parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
  parser.add_argument("--tail-lines", type=int, default=12)
  parser.add_argument("--policy-debug", action="store_true")
  parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--run-mode", choices=eval_run_mode_choices(), default="explicit", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)

  if args.child:
    run_child(args)
    return 0
  if args.policy is None: parser.error("--policy is required")
  return run_parent(args)

if __name__ == "__main__":
  raise SystemExit(main())
