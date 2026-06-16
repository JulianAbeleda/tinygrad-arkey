#!/usr/bin/env python3
"""S1 — greedy speculative decoding (exact): 1.7B drafts K tokens, 8B verifies them in one batched
forward (the S3 batched GEMM primitive), accept the longest greedy-matching prefix + one correction.
Greedy accept => output is IDENTICAL to pure greedy 8B. Measures effective tok/s and accept length.

Run: DEV=AMD Q4K_PRIMITIVE=1 Q4K_BATCHED=1 PYTHONPATH=. .venv/bin/python extra/qk_speculative.py
"""
import time, sys, argparse
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp
from tinygrad.llm.model import Transformer

def argmax_last(t: Tensor) -> int: return int(t[0, -1].argmax().item())

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--target", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--draft", default="/home/ubuntu/models/Qwen3-1.7B-Q8_0.gguf")
  ap.add_argument("--k", type=int, default=6)
  ap.add_argument("--n", type=int, default=64)
  ap.add_argument("--prompt", type=int, nargs="*", default=[9707, 11, 358, 1079, 264, 4128, 1614, 11])
  args = ap.parse_args()
  K = args.k
  tgt, _ = Transformer.from_gguf(args.target, 4096)
  drf, _ = Transformer.from_gguf(args.draft, 4096)
  temp = Tensor([0.0]); mc = tgt.max_context
  out = sys.__stdout__

  # ---- reference: pure greedy 8B ----
  ref = []
  gr = tgt.generate(list(args.prompt), temperature=0.0)
  for _ in range(args.n): ref.append(next(gr))
  print(f"reference (greedy 8B): {ref[:12]}...", file=out)

  # fresh target instance so the speculative run doesn't reuse the reference's cache state
  tgt, _ = Transformer.from_gguf(args.target, 4096)

  # ---- verify forward: concrete K+1 tokens (for the batched primitive), symbolic start_pos (JIT reuse) ----
  for lin in tgt._q4k_linears.linears: lin.decode_enabled = True  # enable the batched GEMM primitive (we call logits directly, bypassing __call__)
  v_sp = UOp.variable("start_pos", 0, mc - 1)
  @TinyJit
  def verify(toks: Tensor, sp) -> Tensor:
    return tgt.logits(toks, sp).argmax(-1).realize()  # [1, K+1] predicted tokens

  # draft: manual 1-token autoregressive loop with its own KV cache (symbolic start_pos -> one JIT)
  d_sp = UOp.variable("start_pos", 0, mc - 1)
  @TinyJit
  def draft_step(tok1: Tensor, sp) -> Tensor:
    return drf.logits(tok1, sp).argmax(-1).realize()  # [1,1] next token
  for _ in range(3): draft_step(Tensor([[1]]), d_sp.bind(8))

  def draft_k(last_tok, pos, k):  # re-process last_tok at `pos`, then draft k tokens
    drafts = []; cur = last_tok; p = pos
    for _ in range(k):
      cur = int(draft_step(Tensor([[cur]]), d_sp.bind(p))[0, 0].item()); p += 1; drafts.append(cur)
    return drafts

  # ---- bootstrap: prefill target on the prompt, get the first real token ----
  toks = list(args.prompt)
  first = argmax_last(tgt.logits(Tensor([toks]), 0))   # target cache[0:len(prompt)] written
  drf.logits(Tensor([toks]), 0).realize()              # draft cache[0:len(prompt)] written
  toks.append(first); n = len(toks)                    # `first` at pos len(prompt), not yet cached

  # warm the verify JIT (needs 3 traces) at a fixed K+1 shape
  for _ in range(3): verify(Tensor([[1] * (K + 1)]), v_sp.bind(8))

  accept_lengths = []; t_draft = t_verify = 0.0
  st = time.perf_counter()
  while len(toks) < len(args.prompt) + args.n:
    _t = time.perf_counter()
    drafts = draft_k(toks[n - 1], n - 1, K)                     # re-process last accepted, draft d_1..d_K
    t_draft += time.perf_counter() - _t; _t = time.perf_counter()
    V = Tensor([[toks[n - 1]] + drafts])                        # [1, K+1], re-processes last accepted
    P = verify(V, v_sp.bind(n - 1))                             # preds for positions n..n+K
    t_verify += time.perf_counter() - _t
    preds = [int(x) for x in P[0].tolist()]
    m = 0
    while m < K and drafts[m] == preds[m]: m += 1              # greedy accept longest prefix
    new = drafts[:m] + [preds[m]]                               # m accepted + 1 correction/bonus
    toks.extend(new); n += len(new); accept_lengths.append(len(new))
  dt = time.perf_counter() - st
  gen = toks[len(args.prompt):]

  # ---- verdict ----
  match = gen[:args.n] == ref[:len(gen[:args.n])]
  print(f"speculative (K={K}): {gen[:12]}...", file=out)
  print(f"EXACT vs greedy 8B: {match}", file=out)
  print(f"tokens={len(gen)} wall={dt*1000:.0f}ms -> {len(gen)/dt:.1f} tok/s | "
        f"mean accept={sum(accept_lengths)/len(accept_lengths):.2f}/{K+1} | rounds={len(accept_lengths)}", file=out)
  print(f"  draft={t_draft*1000:.0f}ms ({t_draft/dt*100:.0f}%)  verify={t_verify*1000:.0f}ms ({t_verify/dt*100:.0f}%)  "
        f"per-round: draft={t_draft/len(accept_lengths)*1000:.1f}ms verify={t_verify/len(accept_lengths)*1000:.1f}ms", file=out)
  return 0 if match else 1

if __name__ == "__main__":
  raise SystemExit(main())
