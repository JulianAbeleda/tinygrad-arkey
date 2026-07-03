#!/usr/bin/env python3
"""Teacher-forced decode-path NLL evaluator — the quality gate for the B3 demotion search.

Why decode-path (not prefill): a demoted Q6->Q4 tensor only swaps the bytes the DECODE primitive
reads (the requantized `words`); the prefill/fallback path still uses the original fp `weight`
(model.py Q4KPrimitiveLinear._fallback). So a prefill NLL would be blind to the demotion. We therefore run the
real decode path (T=1, decode_enabled=True, symbolic start_pos so the graph compiles ONCE) over a
fixed calibration sequence, teacher-forced, and accumulate -log p(true_next | logits_at_pos).

dNLL(config) = nll(config) - nll(baseline). Lower is better; the ffn_down demotion was accepted at
dNLL ~ -0.0028 (free) -- this evaluator must reproduce that (the Stage-A anchor).

Env flags (Q6K_DEMOTE_FFNDOWN / QK_DEMOTE_TENSORS / Q4K_PRIMITIVE ...) must be set BEFORE this runs
(load_model_and_tokenizer imports tinygrad lazily, preserving the env-ordering invariant).
"""
from __future__ import annotations

import argparse, json

# A fixed, deterministic calibration passage (public-domain-style prose; content is irrelevant,
# only that it is constant across runs so dNLL is comparable).
CALIB_TEXT = (
  "The history of computation is a history of moving work closer to the data it acts on. "
  "Early machines shuttled numbers between memory and a single arithmetic unit, and the cost of "
  "that shuttling came to dominate everything else. Caches, pipelines, and vector units were all "
  "attempts to amortize the same stubborn expense: reading a value from far away is slow, and a "
  "processor that cannot keep its arithmetic units fed will idle no matter how fast they are. "
  "Modern accelerators push this idea to an extreme, with thousands of lanes that must all be "
  "supplied from a shared pool of bandwidth. When the supply runs short, the lanes wait, and the "
  "advertised peak becomes a number that no real program ever reaches. Understanding a workload "
  "therefore means understanding where its bytes come from and how often the same bytes are read."
)

def eval_nll(model_path:str, max_context:int, n_tokens:int, seed:int) -> dict:
  from tinygrad import Tensor, UOp, TinyJit

  from extra.llm.generate import load_model_and_tokenizer

  model, tok = load_model_and_tokenizer(model_path, max_context, seed=seed)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  ids = ids[: n_tokens + 1]
  if len(ids) < 8: raise ValueError(f"calibration too short ({len(ids)} tokens)")

  # JIT the decode-logits step with a symbolic start_pos (compile once, replay) -- same trick the
  # real decode uses (model.generate), so per-step Python re-tracing doesn't dominate.
  v_sp = UOp.variable("start_pos", 0, max_context - 1)
  step = TinyJit(lambda t, sp: model.logits(t, sp).realize())

  total_nll, counted = 0.0, 0
  for i in range(len(ids) - 1):
    lg = step(Tensor([[ids[i]]], dtype="int32").contiguous(), v_sp.bind(i))   # (1,1,vocab), decode path
    total_nll += -float(lg[0, 0].log_softmax()[ids[i + 1]].item())
    counted += 1
  return {"nll": total_nll / counted, "tokens": counted, "model": model_path}

def main():
  ap = argparse.ArgumentParser(description="teacher-forced decode-path NLL (B3 quality gate)")
  ap.add_argument("--model", required=True)
  ap.add_argument("--max-context", type=int, default=4096)
  ap.add_argument("--tokens", type=int, default=160)
  ap.add_argument("--seed", type=int, default=20260616)
  args = ap.parse_args()
  print(json.dumps(eval_nll(args.model, args.max_context, args.tokens, args.seed)))

if __name__ == "__main__":
  main()
