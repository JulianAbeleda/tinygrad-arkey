#!/usr/bin/env python3
"""B3 phase-0: is there headroom for per-tensor adaptive bit-width? Measure each tensor's perplexity
sensitivity to FEWER bits (simulated by per-block min/max quantization of its fp16 weight). Tensors with
near-zero delta at lower bits are over-provisioned by llama.cpp's fixed Q4_K_M -> bytes a search could cut.

Simulated (no real Q4_K/Q6_K quantizer exists); informative for the go/no-go, not a shippable win.
Run: DEV=AMD Q4K_PRIMITIVE=1 PYTHONPATH=. .venv/bin/python extra/qk_quant_sensitivity.py
"""
import sys, numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import Transformer

def block_quant(w: np.ndarray, bits: int, block: int = 32) -> np.ndarray:
  # per-block (last-axis) symmetric min/max quantization to 2^bits levels -- mimics K-quant granularity
  oshape = w.shape; wf = w.reshape(-1, block).astype(np.float32)
  lo = wf.min(1, keepdims=True); hi = wf.max(1, keepdims=True)
  scale = (hi - lo) / (2**bits - 1); scale[scale == 0] = 1
  q = np.round((wf - lo) / scale); deq = q * scale + lo
  return deq.reshape(oshape)

def perplexity(model, seq) -> float:
  lg = model.logits(Tensor([seq]), 0)[0].cast(dtypes.float32)   # [L, vocab], _fallback (fp weights)
  lg = lg - lg.max(axis=-1, keepdim=True)
  logp = lg - lg.exp().sum(axis=-1, keepdim=True).log()         # log-softmax
  tgt = Tensor(seq[1:])                                          # next-token targets
  nll = -logp[:-1].gather(1, tgt.reshape(-1, 1)).mean().item()
  return float(nll)

def main():
  m, _ = Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 4096)
  out = sys.__stdout__
  # eval sequence: greedy-generate 48 tokens from a seed (deltas under perturbation are what matter)
  seq = [9707, 11, 358, 1079, 264, 4128, 1614, 11]
  g = m.generate(list(seq), temperature=0.0)
  for _ in range(48): seq.append(next(g))
  m2, _ = Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 4096)  # fresh (no cache state)
  base = perplexity(m2, seq)
  print(f"baseline NLL (Q4_K_M as loaded, fp fallback) = {base:.4f}", file=out)

  lins = m2._q4k_linears.linears
  # CUMULATIVE: perturb ALL layers of a role at once (the real headroom test, not 1-of-36)
  roles = ["ffn_down", "ffn_gate", "ffn_up", "attn_q", "attn_output", "output.weight"]
  q6_roles = {"ffn_down", "output.weight", "attn_v"}
  print(f"{'role (all layers)':<20} {'q':>3} {'n':>3}  dNLL@4bit  dNLL@3bit  dNLL@2bit", file=out)
  for role in roles:
    group = [l for l in lins if (l.name == role) or (f".{role}.weight" in l.name)]
    if not group: continue
    saved = [(l, l.weight.numpy().copy()) for l in group]
    deltas = []
    for bits in (4, 3, 2):
      for l, w0 in saved: l.weight = Tensor(block_quant(w0, bits).astype(np.float16)).to(l.weight.device).realize()
      deltas.append(perplexity(m2, seq) - base)
    for l, w0 in saved: l.weight = Tensor(w0).to(l.weight.device).realize()  # restore
    q = "Q6" if any(r in role for r in q6_roles) else "Q4"
    print(f"{role:<20} {q:>3} {len(group):>3}  {deltas[0]:+8.4f}  {deltas[1]:+8.4f}  {deltas[2]:+8.4f}", file=out)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
