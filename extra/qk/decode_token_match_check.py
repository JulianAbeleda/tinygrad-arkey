#!/usr/bin/env python3
"""Greedy in-model token-correctness check: dump the first N greedy decode token ids at a fixed ctx.
Run twice (baseline vs a flag stack) and diff the printed lists -- IDENTICAL => the flags are token-correct
in-model (the real authority the W==D tok/s harness does NOT check). Self-review of the model-wide firing of
SCHED_UNROLL / COALESCED_LOAD_LOWERING / DECODE_STAGE_COALESCE / DECODE_FAST_EXP2.

Run: DEV=AMD JIT=1 <flags> QK_CKPT=4096 PYTHONPATH=. .venv/bin/python extra/qk/decode_token_match_check.py
"""
from __future__ import annotations
import os, json
N = int(os.environ.get("QK_NTOK", "24")); CK = int(os.environ.get("QK_CKPT", "4096")); MAXC = 4608


def main() -> int:
  from extra.qk.harness_contract import DEFAULT_MODEL
  model = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  from tinygrad import Tensor, UOp, TinyJit
  from extra.llm.generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
  for b in m.blk: b._use_flash, b._prefill_v2 = CK >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512")), False
  step = TinyJit(m.forward)
  out = Tensor([[int(ids[CK])]], dtype="int32").contiguous()
  toks = []
  for i in range(N):
    out = step(out, v_sp.bind(CK + i), temp)
    toks.append(int(out.item()))
  print("TOKENS", json.dumps(toks))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
