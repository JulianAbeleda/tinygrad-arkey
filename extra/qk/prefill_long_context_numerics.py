#!/usr/bin/env python3
"""Long-context prefill numerics gate.

THEORY 6 long-context numerics: same real path as extra/qk/prefill_hd_sweep_numerics.py, but with
kv/q sweepable up to 4096 and the reference computed per-head in numpy (fp32) so VRAM does not cap it.

A softmax-reduction change that breaks numerical stability shows up at long kv before short kv, which the
shipped sweep (kv=512 only) cannot see.
"""
from __future__ import annotations
import os
os.environ.setdefault("DEV", "AMD")
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.fused_attention import custom_kernel_attention
from tinygrad.uop.ops import SharedAttentionCandidateContext

HQ, HKV = 32, 8


def ref_np(qn, kn, vn, hq, hkv, scale):
  """Per-head fp32 causal softmax attention in numpy (chunk-free on heads, O(T*KV) per head)."""
  g = hq // hkv
  out = np.empty_like(qn, dtype=np.float32)
  T, KV = qn.shape[2], kn.shape[2]
  causal = np.tril(np.ones((T, KV), dtype=bool), k=KV - T)
  for h in range(hq):
    s = qn[0, h].astype(np.float32) @ kn[0, h // g].astype(np.float32).T * scale
    s = np.where(causal, s, -np.inf)
    s -= s.max(axis=-1, keepdims=True)
    p = np.exp(s); p /= p.sum(axis=-1, keepdims=True)
    out[0, h] = p @ vn[0, h // g].astype(np.float32)
  return out


def run(hd: int, kv: int) -> str:
  # q_tokens is pinned to 512 by ADMITTED_GRIDS; long context is the real CHUNKED prefill shape
  # (q=512 chunk, kv=kv, start_pos=kv-512), which is exactly what prefill_whole_synced drives.
  tokens = 512
  ctx = SharedAttentionCandidateContext(
    "qwen3_8b_q4k_m_gfx1100", "FULL_RESIDENT_OVERLAY", tokens, kv, kv - tokens, HQ, HKV, hd,
    True, acc_blocks=hd // 16, output_block_base=0)
  rng = np.random.default_rng(20260724 + hd + kv)
  qn = rng.normal(0, .04, (1, HQ, tokens, hd)).astype(np.float16)
  kn = rng.normal(0, .04, (1, HKV, kv, hd)).astype(np.float16)
  vn = rng.normal(0, .04, (1, HKV, kv, hd)).astype(np.float16)
  q, k, v = (Tensor(x, device="AMD") for x in (qn, kn, vn))
  got = custom_kernel_attention(q, k, v, scale=None, causal=True, ctx=ctx).numpy().astype(np.float32)
  ref = ref_np(qn, kn, vn, HQ, HKV, hd ** -0.5)
  max_abs = float(np.max(np.abs(got - ref)))
  finite = bool(np.all(np.isfinite(got)))
  ok = bool(np.allclose(got, ref, rtol=.03, atol=.006)) and finite
  return f"Hd={hd} kv={kv}: max_abs_err={max_abs:.4g} finite={finite} {'PASS' if ok else 'FAIL'}"


if __name__ == "__main__":
  print(f'--- PREFILL_SOFTMAX_REDUCE_FUSE={os.environ.get("PREFILL_SOFTMAX_REDUCE_FUSE", "0")}')
  for hd, tok in ((64, 512), (128, 512), (128, 1024), (128, 2048), (128, 4096)):
    try:
      print(run(hd, tok), flush=True)
    except Exception as e:
      print(f"Hd={hd} kv={tok}: FAIL {type(e).__name__}: {str(e)[:140]}", flush=True)
