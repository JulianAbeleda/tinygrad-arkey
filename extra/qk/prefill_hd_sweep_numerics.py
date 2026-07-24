#!/usr/bin/env python3
"""Prefill-route Hd-generic numerics sweep.

Mirrors extra/qk/decode_hd_sweep_numerics.py for the PREFILL fused-attention route: proves whether
tinygrad/llm/fused_attention.py:custom_kernel_attention -> FlashPrefillAttentionSpec (P4a, just made
Hd-generic: Hd threaded into the emitter; spec.validate() now allows any positive 16-multiple Hd<=128,
not just 128) actually LOWERS and is NUMERICALLY CORRECT at Hd=64, not merely that it de-literalizes/
builds. Hd=128 is the anchor (the shipped, previously-proven shape; must PASS). Hd=64 is the real test,
driven with SYNTHETIC Q/K/V (no Hd=64 model exists) through the REAL prefill path -- custom_kernel_attention
is called directly, not reimplemented.

REAL entry point driven: tinygrad/llm/fused_attention.py:93 custom_kernel_attention(q,k,v,scale,causal,ctx).
This is the exact function extra/qk/../a4_numerics.py-style harnesses drive at Hd=128 for the 8B/14B
production grids (ROUTES in attention_harness_common.py).

WHERE Hd ACTUALLY COMES FROM in custom_kernel_attention (fused_attention.py:113-115):
  grid = prefill_grid_spec(q, k)  # AMDAttentionGridSpec built from q.shape[-1] etc (line 84-85)
  Hq, Hkv, T, KV, Hd = grid.q_heads, grid.kv_heads, grid.q_tokens, grid.kv_tokens, grid.head_dim
So Hd is read from the ACTUAL Q/K TENSOR SHAPE (q.shape[-1]), not hardcoded, and
AMDAttentionGridSpec.validate() (tinygrad/uop/ops.py:1722-1731) was already de-literalized to accept any
positive 16-wide head_dim (comment there: "head_dim was pinned '!=128' here; de-literalized"). Good --
that part is genuinely Hd-generic.

TWO REAL Hd=128 PINS FOUND (this is the KEY finding -- Hd=64 CANNOT be driven through
custom_kernel_attention(..., ctx=candidate_context(...)) as a4_numerics.py builds ctx; a workaround is
required and is documented+applied below):

  PIN 1 -- tinygrad/uop/ops.py:1419, SharedAttentionCandidateContext.validate():
    `if self.hd != 128 or self.hq % self.hkv: raise ValueError(...)`
    attention_harness_common.candidate_context(...) ALWAYS calls .validate() before returning the ctx,
    so building a ctx with hd=64 via that helper raises immediately -- before custom_kernel_attention
    is even entered. This is a ctx-construction-time guard, unrelated to the emitter/spec's own Hd
    generality.

  PIN 2 -- tinygrad/llm/fused_attention.py:131-133, custom_kernel_attention forwards
    `acc_blocks=ctx.acc_blocks` UNCONDITIONALLY (ctx.acc_blocks defaults to 8, the Hd=128 full-accumulator
    value) into FlashPrefillAttentionSpec. Because ctx.acc_blocks is always a concrete int (never None),
    FlashPrefillAttentionSpec.__post_init__'s Hd-generic default (`acc_blocks = Hd // 16` when None,
    flash_prefill_attention_spec.py:54-56) never fires -- ctx's int always wins. At Hd=64, hd_blocks=4,
    so the emitter itself (tinygrad/schedule/wmma/kernels.py:261) rejects (output_block_base,acc_blocks)
    =(0,8) since (0,8) != (0,hd_blocks=4) and 8 not in {1,2,4}: "grid loop requires a full or aligned
    accumulator slice" ValueError. So even if PIN 1 were bypassed by hand-constructing a ctx object with
    hd=64 but leaving acc_blocks at its default 8, the kernel BUILDER (not just the spec) would still
    reject it on the acc_blocks/hd_blocks mismatch.

  Neither pin lives in the emitter/spec that P4a made Hd-generic -- both live in the ctx layer
  (SharedAttentionCandidateContext + custom_kernel_attention's ctx->spec forwarding), which was written
  when Hd=128 was the only shape and never revisited for P4a. Per the task brief ("if the ctx or spec
  construction pins Hd=128 ... you may need to pass Hd through"), this harness ROUTES AROUND both pins
  by constructing the SharedAttentionCandidateContext NamedTuple DIRECTLY (bypassing candidate_context()'s
  ctx.validate() call, which is PIN 1) with hd=<the real Hd> AND acc_blocks=Hd//16 (the Hd-correct full
  accumulator slice, fixing PIN 2). custom_kernel_attention itself never calls ctx.validate() -- it only
  reads ctx.kv_tokens/start_pos/q_tokens/output_block_base/acc_blocks duck-typed -- so a hand-built ctx
  with a non-128 hd flows through it exactly like a validated one would, and the REAL emitter/builder
  (amd_gfx1100_q16_grid_hd128_loop_attention) is exercised unmodified. This is a harness-side workaround
  of a ctx-construction guard, NOT a reimplementation of the kernel.

Run: PYTHONPATH=. DEV=AMD .venv/bin/python extra/qk/prefill_hd_sweep_numerics.py
"""
from __future__ import annotations
import os
os.environ.setdefault("DEV", "AMD")

import traceback
import numpy as np

from tinygrad import Tensor

from extra.qk.attention_harness_common import causal_mask, reference_attention
from tinygrad.llm.fused_attention import custom_kernel_attention
from tinygrad.uop.ops import SharedAttentionCandidateContext

HQ, HKV = 32, 8   # real 8B GQA grid (G=Hq/Hkv=4); the proven ADMITTED_GRIDS entry is (32,8,512)
Q_TOKENS = 512
KV_TOKENS = 512   # kv==q_tokens -> start_pos=0, matches ADMITTED_GRIDS' proven 512 shape


def make_qkv_hd(hq: int, hkv: int, hd: int, q_tokens: int, kv_tokens: int, seed: int):
  """Same dtype/layout/distribution as attention_harness_common.make_qkv (fp16, device AMD,
  normal(0,.04), shape (1,H,T,Hd)), parametrized on Hd (make_qkv itself hardcodes Hd=128)."""
  rng = np.random.default_rng(seed)
  q = rng.normal(0, .04, (1, hq, q_tokens, hd)).astype(np.float16)
  k = rng.normal(0, .04, (1, hkv, kv_tokens, hd)).astype(np.float16)
  v = rng.normal(0, .04, (1, hkv, kv_tokens, hd)).astype(np.float16)
  return tuple(Tensor(x, device="AMD") for x in (q, k, v))


def run_hd(hd: int) -> str:
  start_pos = KV_TOKENS - Q_TOKENS  # 0 for kv==q_tokens
  acc_blocks = hd // 16  # Hd-correct FULL accumulator slice (fixes PIN 2 above; must land in {1,2,4,8})
  # Hand-built ctx bypasses candidate_context()'s ctx.validate() (PIN 1: hd!=128 -> ValueError) --
  # custom_kernel_attention itself never calls ctx.validate(), only reads fields duck-typed.
  ctx = SharedAttentionCandidateContext(
    "qwen3_8b_q4k_m_gfx1100", "FULL_RESIDENT_OVERLAY", Q_TOKENS, KV_TOKENS, start_pos, HQ, HKV, hd,
    True, acc_blocks=acc_blocks, output_block_base=0)

  q, k, v = make_qkv_hd(HQ, HKV, hd, Q_TOKENS, KV_TOKENS, seed=20260724 + hd)
  mask = causal_mask(Q_TOKENS, KV_TOKENS, ctx.start_pos)

  got = custom_kernel_attention(q, k, v, scale=None, causal=True, ctx=ctx).numpy().astype(np.float32)
  ref = reference_attention(q, k, v, mask, HQ, HKV).numpy().astype(np.float32)

  max_abs = float(np.max(np.abs(got - ref)))
  ok = bool(np.allclose(got, ref, rtol=.03, atol=.006))
  return f"Hd={hd}: lowered=True max_abs_err={max_abs:.4g} {'PASS' if ok else 'FAIL'}"


if __name__ == "__main__":
  for hd in (64, 128):
    try:
      print(run_hd(hd))
    except Exception as e:
      msg = "".join(traceback.format_exception_only(type(e), e)).strip()
      print(f"Hd={hd}: LOWER_FAIL {type(e).__name__}: {msg[:160]}")
