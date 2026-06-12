#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys, time
from typing import Any

DEFAULT_TIMEOUT = 1800.0

def _read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  seen: set[str] = set()
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    line = raw.strip()
    if not line or line.startswith("#"): continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    prompt_id, prompt = row.get("id"), row.get("prompt")
    if not isinstance(prompt_id, str) or not prompt_id: raise ValueError(f"{path}:{lineno}: missing string id")
    if not isinstance(prompt, str) or not prompt: raise ValueError(f"{path}:{lineno}: missing string prompt")
    if prompt_id in seen: raise ValueError(f"{path}:{lineno}: duplicate id {prompt_id!r}")
    seen.add(prompt_id)
    rows.append(row)
  if not rows: raise ValueError(f"{path}: no prompts")
  return rows

def _json_from_output(out:str) -> dict[str, Any]:
  for line in reversed(out.strip().splitlines()):
    try:
      data = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(data, dict): return data
  raise RuntimeError("child produced no JSON line\n" + out[-4000:])

def _env_for_mode(args:argparse.Namespace, mode:str) -> dict[str, str]:
  env = {**os.environ, "DEV": args.device, "JIT": "1", "PYTHONPATH": "."}
  env["QK_PRIMITIVE_STORAGE"] = args.storage
  env.pop("QK_GENERATED_POLICY", None)
  env.pop("QK_GENERATED_POLICY_DEBUG", None)
  env.pop("Q4K_PRIMITIVE_DEBUG", None)
  env.pop("Q6K_PRIMITIVE_DEBUG", None)
  if mode == "explicit":
    env["Q4K_PRIMITIVE"] = "1"
    env["Q6K_PRIMITIVE"] = "1"
  elif mode == "generated":
    if args.policy is None: raise ValueError("--policy is required for generated mode")
    env["Q4K_PRIMITIVE"] = "0"
    env["Q6K_PRIMITIVE"] = "0"
    env["QK_GENERATED_POLICY"] = str(args.policy)
    if args.policy_debug: env["QK_GENERATED_POLICY_DEBUG"] = "1"
  else:
    raise ValueError(f"unknown mode {mode!r}")
  return env

def _child_cmd(args:argparse.Namespace, mode:str) -> list[str]:
  cmd = [
    sys.executable, __file__, "--child", "--run-mode", mode,
    "--model", str(args.model), "--prompts", str(args.prompts),
    "--tokens", str(args.tokens), "--max-context", str(args.max_context),
    "--seed", str(args.seed), "--temperature", str(args.temperature),
    "--storage", args.storage,
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
    })
  modes = {
    "explicit": {"generated": explicit["generated"], "elapsed_s": explicit["elapsed_s"], "tok_s": explicit["tok_s"]},
    "generated": {"generated": generated["generated"], "elapsed_s": generated["elapsed_s"], "tok_s": generated["tok_s"]},
  }
  all_match = all(row["tokens_match"] for row in prompt_rows)
  return {
    "kind": "qwen_eval_summary",
    "status": "pass" if all_match else "fail",
    "tokens_match": all_match,
    "model": generated.get("model") or explicit.get("model"),
    "policy": generated.get("policy"),
    "storage": generated.get("storage") or explicit.get("storage"),
    "prompts": len(prompt_rows),
    "modes": modes,
    "prompt_rows": prompt_rows,
  }

def _md_text(text:str) -> str:
  return text.replace("\n", "\\n").replace("|", "\\|")

def summary_markdown(summary:dict[str, Any], explicit:dict[str, Any], generated:dict[str, Any]) -> str:
  explicit_rows, generated_rows = _as_prompt_map(explicit), _as_prompt_map(generated)
  lines = [
    "# Qwen Eval Harness",
    "",
    "This is a smallest-real rollout/eval check for the accepted QK generated-policy path.",
    "It compares explicit Q4/Q6 primitive flags against the generated shared-storage policy",
    "on fixed prompts with greedy decoding. The correctness gate is exact generated-token",
    "parity between the two modes.",
    "",
    "## Summary",
    "",
    f"- status: `{summary['status']}`",
    f"- model: `{summary.get('model')}`",
    f"- policy: `{summary.get('policy')}`",
    f"- storage: `{summary.get('storage')}`",
    f"- prompts: `{summary['prompts']}`",
    f"- token parity: `{summary['tokens_match']}`",
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
    "| id | token match | explicit tok/s | generated tok/s | explicit text | generated text |",
    "|---|---:|---:|---:|---|---|",
  ]
  for row in summary["prompt_rows"]:
    prompt_id = row["id"]
    lines.append(
      f"| `{prompt_id}` | `{row['tokens_match']}` | {row['explicit_tok_s']:.2f} | {row['generated_tok_s']:.2f} | "
      f"{_md_text(explicit_rows[prompt_id]['text'])} | {_md_text(generated_rows[prompt_id]['text'])} |"
    )
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
  from tinygrad import Tensor
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

  Tensor.manual_seed(args.seed)
  prompts = _read_jsonl(args.prompts)
  model, kv = Transformer.from_gguf(pathlib.Path(args.model).expanduser(), args.max_context)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  results = []
  total_tokens = 0
  st_total = time.perf_counter()
  for prompt in prompts:
    prompt_ids = tok.prefix() + tok.role("user") + tok.encode(prompt["prompt"]) + tok.end_turn() + tok.role("assistant")
    out: list[int] = []
    st = time.perf_counter()
    for tid in model.generate(prompt_ids, temperature=args.temperature):
      if tok.is_end(tid): break
      out.append(tid)
      if len(out) >= args.tokens: break
    elapsed = time.perf_counter() - st
    total_tokens += len(out)
    results.append({
      "id": prompt["id"], "prompt": prompt["prompt"], "tokens": out, "text": tok.decode(out),
      "prompt_len": len(prompt_ids), "generated": len(out), "elapsed_s": round(elapsed, 6),
      "tok_s": 0.0 if elapsed == 0 else len(out) / elapsed,
    })
  elapsed_total = time.perf_counter() - st_total
  print(json.dumps({
    "mode": args.run_mode, "model": str(args.model), "policy": str(args.policy) if args.policy else None,
    "storage": args.storage, "seed": args.seed, "temperature": args.temperature, "max_context": args.max_context,
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

def main() -> int:
  parser = argparse.ArgumentParser(description="Small Qwen rollout/eval harness for explicit vs generated QK primitive policies")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--policy", type=pathlib.Path, help="QK_GENERATED_POLICY artifact for generated mode")
  parser.add_argument("--prompts", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/qwen-eval-20260612/8b-shared"))
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--tokens", type=int, default=24)
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=20260612)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--storage", default="shared")
  parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
  parser.add_argument("--tail-lines", type=int, default=12)
  parser.add_argument("--policy-debug", action="store_true")
  parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--run-mode", choices=("explicit", "generated"), default="explicit", help=argparse.SUPPRESS)
  args = parser.parse_args()

  if args.child:
    run_child(args)
    return 0
  if args.policy is None: parser.error("--policy is required")
  return run_parent(args)

if __name__ == "__main__":
  raise SystemExit(main())
