#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, platform, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_gemm_generation_coverage_result.json"


def _git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def _prompt_windows(tok: Any, prompts: int, prompt_tokens: int, stride: int) -> list[list[int]]:
  from extra.qk_prefill_v2_nll_eval import CALIB_TEXT
  prefix = tok.prefix() if hasattr(tok, "prefix") else []
  body = tok.encode(CALIB_TEXT)
  need = stride * (prompts - 1) + prompt_tokens
  ids = prefix + body
  while len(ids) < need:
    ids += body
  return [ids[i * stride:i * stride + prompt_tokens] for i in range(prompts)]


def _child_eval(args: argparse.Namespace) -> dict[str, Any]:
  from extra.llm_generate import load_model_and_tokenizer
  from tinygrad import Tensor

  Tensor.manual_seed(args.seed)
  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  if not getattr(model, "_pf16_warmstart", None):
    raise RuntimeError("PREFILL_V2 warmstart missing; run with PREFILL_V2=1")

  rows = []
  for idx, ids in enumerate(_prompt_windows(tok, args.prompts, args.prompt_tokens, args.stride)):
    out: list[int] = []
    for tid in model.generate(list(ids), chunk_size=args.chunk_size, temperature=0.0):
      out.append(int(tid))
      if len(out) >= args.max_new_tokens: break
    text = tok.decode(out)
    rows.append({"prompt": idx, "prompt_tokens": len(ids), "tokens": out, "text": text})
    print(f"{args.child_label} prompt {idx}: tokens={out}", file=sys.__stdout__)
  return {
    "label": args.child_label,
    "prefill_graph_gemm": bool(int(os.environ.get("PREFILL_GRAPH_GEMM", "0"))),
    "rows": rows,
  }


def _run_child(model: str, label: str, graph: bool, args: argparse.Namespace) -> dict[str, Any]:
  env = {**os.environ, "DEV": os.environ.get("DEV", "AMD"), "PREFILL_V2": "1", "PYTHONPATH": "."}
  if graph: env["PREFILL_GRAPH_GEMM"] = "1"
  else: env.pop("PREFILL_GRAPH_GEMM", None)
  cmd = [
    sys.executable, str(pathlib.Path(__file__).resolve()), "--child", "--child-label", label,
    "--model", model, "--max-context", str(args.max_context), "--prompts", str(args.prompts),
    "--prompt-tokens", str(args.prompt_tokens), "--stride", str(args.stride),
    "--max-new-tokens", str(args.max_new_tokens), "--chunk-size", str(args.chunk_size),
    "--seed", str(args.seed),
  ]
  last: subprocess.CompletedProcess[str] | None = None
  failures = 0
  for attempt in range(args.retries + 1):
    last = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if last.returncode == 0: break
    failures += 1
    print(last.stdout, file=sys.__stdout__)
    if attempt < args.retries:
      time.sleep(2.0)
      print(f"retrying {label} child after failure ({attempt + 1}/{args.retries})", file=sys.__stdout__)
  assert last is not None
  if last.returncode != 0:
    raise RuntimeError(f"{label} child failed with code {last.returncode}")
  print(last.stdout, file=sys.__stdout__)
  for line in reversed(last.stdout.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      ret = json.loads(line)
      ret["retry_failures"] = failures
      return ret
  raise RuntimeError(f"{label} child produced no JSON result")


def main() -> int:
  ap = argparse.ArgumentParser(description="Greedy generation coverage gate for PREFILL_GRAPH_GEMM")
  ap.add_argument("pos_model", nargs="?")
  ap.add_argument("--model", default=None)
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--prompts", type=int, default=4)
  ap.add_argument("--prompt-tokens", type=int, default=512)
  ap.add_argument("--stride", type=int, default=128)
  ap.add_argument("--max-new-tokens", type=int, default=8)
  ap.add_argument("--chunk-size", type=int, default=32)
  ap.add_argument("--seed", type=int, default=20260620)
  ap.add_argument("--retries", type=int, default=1)
  ap.add_argument("--artifact", default=str(OUT))
  ap.add_argument("--child", action="store_true")
  ap.add_argument("--child-label", default="child")
  args = ap.parse_args()
  args.model = args.model or args.pos_model or os.environ.get("QK_MODEL") or os.environ.get("MODEL")
  if not args.model:
    print("ERROR: pass a model gguf path or set QK_MODEL / MODEL", file=sys.__stdout__)
    return 2
  if args.prompt_tokens < 512:
    print("ERROR: prompt_tokens must be >=512 so PREFILL_V2/PREFILL_GRAPH_GEMM are exercised", file=sys.__stdout__)
    return 2
  if args.child:
    res = _child_eval(args)
    print(json.dumps(res, sort_keys=True), file=sys.__stdout__)
    return 0

  baseline = _run_child(args.model, "baseline_prefill_v2", False, args)
  graph = _run_child(args.model, "graph_gemm", True, args)
  rows = []
  token_mismatches = 0
  prompt_mismatches = 0
  for brow, grow in zip(baseline["rows"], graph["rows"]):
    pairs = list(zip(brow["tokens"], grow["tokens"]))
    mismatches = [{"position": i, "baseline": b, "graph": g} for i, (b, g) in enumerate(pairs) if b != g]
    token_mismatches += len(mismatches) + abs(len(brow["tokens"]) - len(grow["tokens"]))
    prompt_mismatches += 1 if mismatches or len(brow["tokens"]) != len(grow["tokens"]) else 0
    rows.append({
      "prompt": brow["prompt"], "prompt_tokens": brow["prompt_tokens"],
      "baseline_tokens": brow["tokens"], "graph_tokens": grow["tokens"],
      "baseline_text": brow["text"], "graph_text": grow["text"],
      "exact_match": not mismatches and len(brow["tokens"]) == len(grow["tokens"]),
      "mismatches": mismatches,
    })

  retry_failures = int(baseline.get("retry_failures", 0)) + int(graph.get("retry_failures", 0))
  gates = {
    "prompt_count_expected": len(rows) == args.prompts,
    "token_mismatches_zero": token_mismatches == 0,
    "prompt_mismatches_zero": prompt_mismatches == 0,
    "child_retry_failures_zero": retry_failures == 0,
  }
  verdict = "PASS_PREFILL_GRAPH_GEMM_GENERATION_COVERAGE" if all(gates.values()) else "BLOCKED_PREFILL_GRAPH_GEMM_GENERATION_COVERAGE"
  result = {
    "date": "2026-06-20", "phase": "PREFILL_GRAPH_GEMM_GENERATION_COVERAGE",
    "schema": "prefill_graph_gemm_generation_coverage_v1", "verdict": verdict,
    "gate_pass": all(gates.values()), "model_id": pathlib.Path(args.model).name,
    "hardware": platform.node(), "commit": _git_sha(), "prompts": args.prompts,
    "prompt_tokens": args.prompt_tokens, "stride": args.stride, "max_new_tokens": args.max_new_tokens,
    "token_mismatches": token_mismatches, "prompt_mismatches": prompt_mismatches,
    "retry_failures": retry_failures, "rows": rows, "gates": gates,
  }
  out = pathlib.Path(args.artifact)
  if not out.is_absolute(): out = ROOT / out
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  try: out_name = str(out.relative_to(ROOT))
  except ValueError: out_name = str(out)
  print(json.dumps({"verdict": verdict, "gate_pass": result["gate_pass"],
                    "token_mismatches": token_mismatches, "prompt_mismatches": prompt_mismatches,
                    "gates": gates, "out": out_name}, indent=2), file=sys.__stdout__)
  return 0 if all(gates.values()) else 1


if __name__ == "__main__":
  raise SystemExit(main())
