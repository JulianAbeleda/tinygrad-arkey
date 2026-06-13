#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys, time

DEFAULT_PROMPT = "Give a concise answer: what is the purpose of a compiler optimization pass?"

def _json_from_output(out:str) -> dict:
  for line in reversed(out.strip().splitlines()):
    try: return json.loads(line)
    except json.JSONDecodeError: pass
  raise RuntimeError("child produced no JSON line\n" + out[-2000:])

def run_child(args, primitive:bool) -> dict:
  env = {**os.environ, "PYTHONPATH": "."}
  env.pop("QK_GENERATED_POLICY", None)
  env.pop("QK_GENERATED_POLICY_DEBUG", None)
  if primitive and args.candidate_policy is not None:
    env["QK_GENERATED_POLICY"] = str(args.candidate_policy)
    if args.policy_debug: env["QK_GENERATED_POLICY_DEBUG"] = "1"
    env["Q4K_PRIMITIVE"] = "0"
    env["Q6K_PRIMITIVE"] = "0"
  elif not primitive and args.baseline_policy is not None:
    env["QK_GENERATED_POLICY"] = str(args.baseline_policy)
    if args.policy_debug: env["QK_GENERATED_POLICY_DEBUG"] = "1"
    env["Q4K_PRIMITIVE"] = "0"
    env["Q6K_PRIMITIVE"] = "0"
  else:
    env["Q4K_PRIMITIVE"] = "1" if primitive else "0"
    env["Q6K_PRIMITIVE"] = "1" if primitive and args.q6_primitive else "0"
  cmd = [sys.executable, __file__, "--child", "--model", str(args.model), "--tokens", str(args.tokens),
         "--max-context", str(args.max_context), "--seed", str(args.seed), "--temperature", str(args.temperature),
         "--prompt", args.prompt, "--primitive", "1" if primitive else "0"]
  if primitive and args.candidate_policy is not None: cmd += ["--candidate-policy", str(args.candidate_policy)]
  if not primitive and args.baseline_policy is not None: cmd += ["--baseline-policy", str(args.baseline_policy)]
  if primitive and args.policy_debug: cmd += ["--policy-debug"]
  if not primitive and args.baseline_policy is not None and args.policy_debug: cmd += ["--policy-debug"]
  st = time.perf_counter()
  proc = subprocess.run(cmd, cwd=args.repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        timeout=args.timeout)
  elapsed = time.perf_counter() - st
  data = _json_from_output(proc.stdout)
  data["returncode"], data["elapsed_s"] = proc.returncode, round(elapsed, 3)
  data["tail"] = "\n".join(proc.stdout.strip().splitlines()[-args.tail_lines:])
  if proc.returncode != 0:
    raise RuntimeError(f"child primitive={primitive} failed rc={proc.returncode}\n{proc.stdout[-4000:]}")
  return data

def run_generation(args) -> None:
  if args.primitive and args.candidate_policy is not None:
    os.environ["QK_GENERATED_POLICY"] = str(args.candidate_policy)
    if args.policy_debug: os.environ["QK_GENERATED_POLICY_DEBUG"] = "1"
    os.environ["Q4K_PRIMITIVE"] = "0"
    os.environ["Q6K_PRIMITIVE"] = "0"
  elif not args.primitive and args.baseline_policy is not None:
    os.environ["QK_GENERATED_POLICY"] = str(args.baseline_policy)
    if args.policy_debug: os.environ["QK_GENERATED_POLICY_DEBUG"] = "1"
    os.environ["Q4K_PRIMITIVE"] = "0"
    os.environ["Q6K_PRIMITIVE"] = "0"
  else:
    os.environ.pop("QK_GENERATED_POLICY", None)
    os.environ.pop("QK_GENERATED_POLICY_DEBUG", None)
    os.environ["Q4K_PRIMITIVE"] = "1" if args.primitive else "0"
  from tinygrad import Tensor
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

  Tensor.manual_seed(args.seed)
  model, kv = Transformer.from_gguf(pathlib.Path(args.model).expanduser(), args.max_context)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  prompt_ids = tok.prefix() + tok.role("user") + tok.encode(args.prompt) + tok.end_turn() + tok.role("assistant")
  out: list[int] = []
  for tid in model.generate(prompt_ids, temperature=args.temperature):
    if tok.is_end(tid): break
    out.append(tid)
    if len(out) >= args.tokens: break
  print(json.dumps({
    "primitive": bool(args.primitive), "candidate_policy": str(args.candidate_policy) if args.candidate_policy else None,
    "baseline_policy": str(args.baseline_policy) if args.baseline_policy else None,
    "tokens": out, "text": tok.decode(out),
    "prompt_len": len(prompt_ids), "generated": len(out), "model": str(args.model),
  }, sort_keys=True))

def print_summary(baseline:dict, primitive:dict) -> None:
  print("| path | generated | elapsed_s | first tokens |")
  print("|---|---:|---:|---|")
  for name, row in (("baseline", baseline), ("primitive", primitive)):
    if name == "primitive" and row.get("candidate_policy"): name = "generated_policy"
    if name == "baseline" and row.get("baseline_policy"): name = "baseline_policy"
    print(f"| {name} | {row['generated']} | {row['elapsed_s']} | {row['tokens'][:8]} |")
  print(f"match={baseline['tokens'] == primitive['tokens']}")
  if baseline["tokens"] != primitive["tokens"]:
    for idx, (a, b) in enumerate(zip(baseline["tokens"], primitive["tokens"])):
      if a != b:
        print(f"first_mismatch_index={idx} baseline={a} primitive={b}")
        break
    if len(baseline["tokens"]) != len(primitive["tokens"]):
      print(f"length_mismatch baseline={len(baseline['tokens'])} primitive={len(primitive['tokens'])}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Greedy output A/B for baseline vs Q4_K primitive decode")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--tokens", type=int, default=32)
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=20260611)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--prompt", default=DEFAULT_PROMPT)
  parser.add_argument("--timeout", type=float, default=900)
  parser.add_argument("--tail-lines", type=int, default=8)
  parser.add_argument("--json", type=pathlib.Path, help="write comparison JSON")
  parser.add_argument("--q6-primitive", action="store_true", help="enable Q6K_PRIMITIVE=1 only for the primitive child")
  parser.add_argument("--baseline-policy", type=pathlib.Path, help="compare this QK_GENERATED_POLICY baseline against the candidate")
  parser.add_argument("--candidate-policy", type=pathlib.Path, help="compare baseline against this QK_GENERATED_POLICY instead of explicit primitive flags")
  parser.add_argument("--policy-debug", action="store_true", help="enable QK_GENERATED_POLICY_DEBUG in the candidate child")
  parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--primitive", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
  args = parser.parse_args()

  if args.child:
    run_generation(args)
  else:
    baseline = run_child(args, False)
    primitive = run_child(args, True)
    result = {"match": baseline["tokens"] == primitive["tokens"], "baseline": baseline, "primitive": primitive,
              "prompt": args.prompt, "tokens": args.tokens, "seed": args.seed, "temperature": args.temperature}
    print_summary(baseline, primitive)
    if args.json:
      args.json.parent.mkdir(parents=True, exist_ok=True)
      args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["match"] else 1)
