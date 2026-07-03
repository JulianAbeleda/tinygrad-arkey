#!/usr/bin/env python3
"""Prefilled in-model correctness authority for the generated decode-attention route.

Unlike qk_decode_token_match_check.py (which never prefills the KV cache and is therefore non-deterministic), this
does a REAL prefill of a long deterministic prompt (>=512 tokens) via model.generate, then greedy-decodes (temp=0).
Because the prompt is prefilled with real content, the KV cache is populated deterministically, so two runs with the
same flags are bit-reproducible and two runs with different attention routes are directly comparable. The decode
steps run at ctx>=512, so the g5 live-split route actually fires (route-binding is separately confirmed by the W==D
harness). Set the route via env flags; this script just prints the greedy token ids.

Run: DEV=AMD JIT=1 <route flags> PYTHONPATH=. python3 extra/qk/prefilled_route_parity.py
"""
from __future__ import annotations
import json, os

N = int(os.environ.get("QK_NTOK", "48"))
# ~600 tokens of deterministic text so the decode context is >= 512 (g5 route eligible).
PROMPT = "The history of computation is a history of moving work closer to the data it acts on. " * 40


def main() -> int:
  from extra.qk.harness_contract import DEFAULT_MODEL
  from extra.llm.generate import load_model_and_tokenizer
  from extra.llm.eval_common import build_prompt_ids
  model = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  m, tok = load_model_and_tokenizer(model, 4608, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  # collect token IDs directly (avoid tok.decode, which KeyErrors on out-of-dict special tokens) -- IDs are the
  # correctness signal for route parity.
  ids = build_prompt_ids(tok, PROMPT, os.environ.get("QK_PROMPT_FORMAT", "raw"))
  out = []
  for tid in m.generate(ids, temperature=0.0):
    if tok.is_end(tid): break
    out.append(int(tid))
    if len(out) >= N: break
  print("@@PARITY@@" + json.dumps({"prompt_len": len(ids), "tokens": out}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
