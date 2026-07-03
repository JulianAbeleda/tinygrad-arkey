#!/usr/bin/env python3
"""Prefix/cache reuse benchmark.

This is intentionally separate from raw prefill and decode benchmarks. It measures
runtime/session behavior: how much first-token latency is avoided when the next
request shares a stable prompt prefix with the previous request.

Outputs:
  bench/qk-prefix-cache-reuse/{latest.json,summary.md}

Example:
  DEV=AMD JIT=1 PYTHONPATH=. python extra/qk/prefix_cache_bench.py \
    --model /home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf --max-context 2048 \
    --stable-prefix-tokens 1024 --suffix-tokens 128 --decode-tokens 16
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, time
from typing import Any

from tinygrad.helpers import fetch
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.model import Transformer

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "qk-prefix-cache-reuse"


def _repeat_tokens(tok:SimpleTokenizer, text:str, n:int) -> list[int]:
  if n <= 0: return []
  seed = tok.encode(text)
  if not seed: raise RuntimeError("tokenizer produced no tokens for benchmark seed text")
  reps = (n + len(seed) - 1) // len(seed)
  return (seed * reps)[:n]


def _median(xs:list[float]) -> float:
  return statistics.median(xs) if xs else 0.0


def _run_once(model:Transformer, prompt:list[int], decode_tokens:int, *, clear_cache:bool) -> dict[str, Any]:
  if clear_cache: model._cached_tokens = []
  cached_prefix = model.get_start_pos(prompt)
  uncached = len(prompt) - cached_prefix
  start = time.perf_counter()
  first = None
  out: list[int] = []
  for tid in model.generate(list(prompt), temperature=0.0):
    if first is None: first = time.perf_counter()
    out.append(tid)
    if len(out) >= decode_tokens: break
  end = time.perf_counter()
  ttft_s = (first or end) - start
  total_s = end - start
  decode_tail_s = max(end - (first or end), 0.0)
  return {
    "prompt_tokens": len(prompt),
    "cached_prefix_tokens": cached_prefix,
    "uncached_prefill_tokens": uncached,
    "cache_hit_ratio": round(cached_prefix / len(prompt), 6) if prompt else 0.0,
    "completion_tokens": len(out),
    "ttft_ms": round(ttft_s * 1000, 3),
    "total_ms": round(total_s * 1000, 3),
    "decode_tok_s_after_first": round(max(len(out) - 1, 0) / decode_tail_s, 3) if decode_tail_s > 0 else 0.0,
    "effective_input_tok_s_all_tokens": round(len(prompt) / ttft_s, 3) if ttft_s > 0 else 0.0,
    "effective_input_tok_s_cache_miss_only": round(uncached / ttft_s, 3) if ttft_s > 0 else 0.0,
  }


def _seed_cache(model:Transformer, prompt:list[int]) -> None:
  list(zip(range(1), model.generate(list(prompt), temperature=0.0)))


def _case_summary(name:str, rows:list[dict[str, Any]], expected:str) -> dict[str, Any]:
  return {
    "case": name,
    "expected": expected,
    "reps": len(rows),
    "prompt_tokens": rows[-1]["prompt_tokens"] if rows else None,
    "cached_prefix_tokens_median": _median([r["cached_prefix_tokens"] for r in rows]),
    "uncached_prefill_tokens_median": _median([r["uncached_prefill_tokens"] for r in rows]),
    "cache_hit_ratio_median": round(_median([r["cache_hit_ratio"] for r in rows]), 6),
    "ttft_ms_median": round(_median([r["ttft_ms"] for r in rows]), 3),
    "total_ms_median": round(_median([r["total_ms"] for r in rows]), 3),
    "effective_input_tok_s_all_tokens_median": round(_median([r["effective_input_tok_s_all_tokens"] for r in rows]), 3),
    "effective_input_tok_s_cache_miss_only_median": round(_median([r["effective_input_tok_s_cache_miss_only"] for r in rows]), 3),
    "rows": rows,
  }


def main() -> None:
  ap = argparse.ArgumentParser(description="Benchmark local prefix/KV cache reuse as a separate session-runtime metric.")
  ap.add_argument("--model", required=True, help="GGUF model path or fetchable URL")
  ap.add_argument("--id", default=None, help="model id for the artifact")
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--stable-prefix-tokens", type=int, default=1024)
  ap.add_argument("--suffix-tokens", type=int, default=128)
  ap.add_argument("--decode-tokens", type=int, default=16)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--warmup-reps", type=int, default=1, help="unmeasured runs to compile benchmark prompt shapes")
  ap.add_argument("--out", default=str(OUT))
  args = ap.parse_args()

  if args.stable_prefix_tokens + args.suffix_tokens + args.decode_tokens >= args.max_context:
    raise SystemExit("stable-prefix-tokens + suffix-tokens + decode-tokens must be less than max-context")

  src = fetch(args.model)
  model, kv = Transformer.from_gguf(src, args.max_context)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  model_id = args.id or pathlib.Path(str(args.model)).stem

  stable = tok.prefix() + _repeat_tokens(tok, " stable repository context. invariant route ledger. ", args.stable_prefix_tokens)
  suffix_a = _repeat_tokens(tok, " user asks for a cache-friendly answer. ", args.suffix_tokens)
  suffix_b = _repeat_tokens(tok, " user asks for a slightly different tail request. ", args.suffix_tokens)
  changed_front = _repeat_tokens(tok, " volatile request metadata changes before stable context. ", args.suffix_tokens)

  prompts = {
    "cold_full": stable + suffix_a,
    "warm_same_prefix_changed_suffix": stable + suffix_b,
    "prefix_broken_changed_front": changed_front + stable,
  }

  # Warm the decode graph and benchmark prompt shapes without leaving a useful
  # benchmark prefix in the cache. Prefix/cache reuse should measure avoided
  # prefill, not first-run JIT compile tax.
  list(zip(range(2), model.generate(tok.prefix() + _repeat_tokens(tok, " warmup ", 16))))
  for _ in range(args.warmup_reps):
    for prompt in prompts.values():
      model._cached_tokens = []
      _seed_cache(model, prompt)
  model._cached_tokens = []

  cases: list[dict[str, Any]] = []
  for name, seed, expected in [
    ("cold_full", None, "no reusable prefix; full prompt prefill"),
    ("warm_same_prefix_changed_suffix", "cold_full", "large stable prefix reused; only changed suffix prefills"),
    ("prefix_broken_changed_front", "cold_full", "early-token change breaks prefix reuse; mostly cache miss"),
  ]:
    rows = []
    for rep in range(args.reps):
      # Every repetition starts from the same cache state. Warm cases seed the
      # cache with prompt A, then measure their target prompt. This prevents a
      # warm case from caching itself on rep 2 and overstating prefix reuse.
      model._cached_tokens = []
      if seed is not None: _seed_cache(model, prompts[seed])
      rows.append(_run_once(model, prompts[name], args.decode_tokens, clear_cache=False))
    cases.append(_case_summary(name, rows, expected))

  cold = next(c for c in cases if c["case"] == "cold_full")
  for c in cases:
    c["ttft_speedup_vs_cold"] = round(cold["ttft_ms_median"] / c["ttft_ms_median"], 3) if c["ttft_ms_median"] else None
    c["ttft_saved_ms_vs_cold"] = round(cold["ttft_ms_median"] - c["ttft_ms_median"], 3)

  artifact = {
    "schema": "tinygrad.prefix_cache_reuse_bench.v1",
    "benchmark_type": "prefix_cache_reuse",
    "separate_from": ["prefill", "decode"],
    "model_id": model_id,
    "model_path": str(args.model),
    "architecture": kv.get("general.architecture"),
    "max_context": model.max_context,
    "config": {
      "stable_prefix_tokens_requested": args.stable_prefix_tokens,
      "suffix_tokens": args.suffix_tokens,
      "decode_tokens": args.decode_tokens,
      "reps": args.reps,
      "warmup_reps": args.warmup_reps,
    },
    "metric_definitions": {
      "ttft_ms": "time to first generated token; includes uncached prefill plus first decode token",
      "cached_prefix_tokens": "tokens skipped by Transformer.get_start_pos prefix reuse",
      "uncached_prefill_tokens": "prompt_tokens - cached_prefix_tokens",
      "effective_input_tok_s_all_tokens": "prompt_tokens / ttft; useful for user-visible latency",
      "effective_input_tok_s_cache_miss_only": "uncached_prefill_tokens / ttft; useful for compute accounting",
    },
    "cases": cases,
    "verdict": "PREFIX_CACHE_BENCH_COMPLETE",
  }

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / "latest.json").write_text(json.dumps(artifact, indent=2))

  lines = [
    "# Prefix/cache reuse benchmark",
    "",
    f"**Verdict:** {artifact['verdict']}",
    "",
    "| case | cached prefix | uncached prefill | hit ratio | TTFT ms | saved vs cold | speedup |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for c in cases:
    lines.append("| {case} | {cached_prefix_tokens_median:.0f} | {uncached_prefill_tokens_median:.0f} | "
                 "{cache_hit_ratio_median:.3f} | {ttft_ms_median:.1f} | {ttft_saved_ms_vs_cold:.1f} | "
                 "{ttft_speedup_vs_cold:.2f}x |".format(**c))
  lines += [
    "",
    "This benchmark is a session/runtime cache benchmark, not a raw prefill or decode benchmark.",
    "Benchmark prompt shapes are pre-warmed before measurement so cold rows do not include first-run JIT compile tax.",
    "TTFT includes the uncached prefill suffix and the first decoded token; decode throughput remains measured by the decode authority.",
  ]
  (out / "summary.md").write_text("\n".join(lines) + "\n")
  print(f"wrote {out/'latest.json'} and {out/'summary.md'}")


if __name__ == "__main__":
  main()
