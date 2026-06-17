#!/usr/bin/env python3
"""Stage 1 of the flash-threshold search: sweep decode tok/s vs context, SDPA vs flash.

Flash decode (`FLASH_DECODE=1`) wins at long context but regresses short context (5 extra
kernels/layer); the crossover is ~ctx 400. This runner measures the crossover precisely by generating
one continuous decode run (ctx 0 -> N_max) and sampling steady tok/s in a window around each context
bucket. FLASH_DECODE is a graph-CAPTURE-time flag, so each invocation measures ONE mode (the caller
runs two processes: FLASH_DECODE=0 for SDPA, =1 for flash) -- `extra/qk_flash_search.py` orchestrates.

Confound-controlled: warm (a few discarded tokens first), median over a per-bucket window.
Output: one JSON line `{"mode": "sdpa"|"flash", "by_ctx": {ctx: tok_s, ...}}`.
"""
from __future__ import annotations

import argparse, json, time

DEFAULT_BUCKETS = (8, 256, 384, 512, 768, 1024, 1536, 2048, 3072)

def sweep(model_path:str, max_context:int, buckets:tuple[int, ...], window:int, seed:int) -> dict:
  import statistics

  from extra.llm_generate import load_model_and_tokenizer
  model, tok = load_model_and_tokenizer(model_path, max_context, seed=seed)

  n_max = max(buckets) + window + 4
  per_ctx: dict[int, float] = {}          # ctx -> tok/s for that single decode step
  gen = model.generate([tok.bos_id or 0])
  next(gen)                                # absorb prefill + first (cold) token
  for _ in range(3): next(gen)            # warm the clock/JIT
  ctx = 4
  while ctx < n_max:
    t0 = time.perf_counter()
    next(gen)
    dt = time.perf_counter() - t0
    per_ctx[ctx] = 1.0 / dt if dt > 0 else 0.0
    ctx += 1

  by_ctx = {}
  for b in buckets:
    vals = [v for c, v in per_ctx.items() if abs(c - b) <= window]
    if vals: by_ctx[b] = round(statistics.median(vals), 2)
  return {"by_ctx": by_ctx, "n_sampled": len(per_ctx)}

def main():
  ap = argparse.ArgumentParser(description="flash-vs-SDPA decode tok/s context sweep (one mode per run)")
  ap.add_argument("--model", required=True)
  ap.add_argument("--max-context", type=int, default=4096)
  ap.add_argument("--buckets", type=int, nargs="*", default=list(DEFAULT_BUCKETS))
  ap.add_argument("--window", type=int, default=24, help="+/- token window around each bucket for the median")
  ap.add_argument("--seed", type=int, default=20260616)
  ap.add_argument("--mode", default="sdpa", help="label only (sdpa|flash); FLASH_DECODE env sets the actual path")
  args = ap.parse_args()
  res = sweep(args.model, args.max_context, tuple(args.buckets), args.window, args.seed)
  print(json.dumps({"mode": args.mode, **res}))

if __name__ == "__main__":
  main()
