#!/usr/bin/env python3
"""Arc 1 Phase 4: in-model validation of lowering FLASH_DECODE_THRESHOLD. Real model.generate at ctx>=512 (a
~512-token prompt + 48 generated), greedy. Measures decode-only tok/s (clock starts after the first token, so
prefill is excluded) and records the exact generated token ids. Read FLASH_DECODE_THRESHOLD from env; run once
per threshold in separate processes, then diff: tokens MUST be identical (flash-decode is exact) and ctx>=512
tok/s should improve. No default changed by this harness (env-driven).

Run: DEV=AMD JIT=1 FLASH_DECODE_THRESHOLD=512  PYTHONPATH=. .venv/bin/python extra/qk_flash_threshold_validate.py
     DEV=AMD JIT=1 FLASH_DECODE_THRESHOLD=1024 PYTHONPATH=. .venv/bin/python extra/qk_flash_threshold_validate.py
"""
from __future__ import annotations
import json, os, pathlib, statistics, sys, time

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  thr = int(os.environ.get("FLASH_DECODE_THRESHOLD", "1024"))
  n_new = int(os.environ.get("QK_NNEW", "48"))
  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, 1024, seed=20260617)
  # ~520-token prompt so the decode loop runs entirely at ctx >= 512 (where threshold 512 picks flash, 1024 SDPA)
  base = tok.encode("The history of computing is a long and winding road. ")
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + (base * (520 // max(1, len(base)) + 1))[:520]
  per_tok, gen = [], []
  prev = None
  for i, t in enumerate(m.generate(list(ids), temperature=0.0)):
    now = time.perf_counter()
    if prev is not None: per_tok.append(now - prev)
    prev = now; gen.append(int(t))
    if len(gen) >= n_new or t == getattr(tok, "eos_id", -1): break
  # drop the first interval (includes prefill tail / warm); use steady-state median
  steady = per_tok[2:] if len(per_tok) > 3 else per_tok
  tok_s = round(1 / statistics.median(steady), 2) if steady else 0
  out = {"threshold": thr, "ctx_start": len(ids), "n_new": len(gen), "decode_tok_s": tok_s,
         "tokens": gen, "tokens_head": gen[:12]}
  art = pathlib.Path(f"bench/qk-8b-attention-threshold/thr_{thr}.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"@@ thr={thr} ctx_start={len(ids)} decode_tok_s={tok_s} n_new={len(gen)} head={gen[:8]}", file=sys.__stderr__)
  print("@@DONE@@")

if __name__ == "__main__":
  main()
