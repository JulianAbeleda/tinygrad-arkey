#!/usr/bin/env python3
"""Decode-route Hd-generic numerics sweep.

Proves whether the live-split flash-decode emitter --
extra/qk/flash_kernels.py:flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel -- LOWERS and
produces numerically CORRECT output at head_dim (Hd) != 128, for Hd in {64, 128, 192, 256}.
Hd=128 is the anchor (must PASS, it's the shipped production shape). 64/192/256 are the real test:
the emitter is believed Hd-generic (R=Hd//LANES, RP=Hd//64, W=Hd+2, scale=1/Hd**0.5, only
constraint Hd%64==0 -- see flash_kernels.py line ~25) but that has never been run end-to-end.

REAL entry point driven (not reimplemented, not a toy):
  extra/qk/flash_decode_attention_executor.py:10 flash_decode_live_split_block_tile(...)
This is the EXACT function tinygrad/llm/decode_routes.py:157 (flash_decode_attention_route) calls
for the production 8B/14B decode graph -- see tinygrad/llm/model.py:597. It wires
Q/cache -> FlashDecodeAttentionSpec -> custom_kernel tile emit -> custom_kernel fused-combine emit,
identically to the live model path.

SHAPE-GUARD BYPASS: tinygrad/llm/decode_routes.py:_FlashDecodeCandidate.bind (~line 125-131) hard-codes
`Hd != 128 -> None`, which is a MANIFEST/ROUTE-SELECTION guard, not an emitter constraint -- it exists
so decode_routes.py can fail loud on shapes nobody proved yet. We bypass ONLY that guard by calling
flash_decode_live_split_block_tile directly (as decode_routes.py itself does once bind() succeeds),
so this harness exercises the real emitter + renderer + compiler, not a reimplementation.

Reference: extra/qk/attention_harness_common.py reference_attention (the SDPA/GQA golden used by the
existing custom-kernel-attention numerics harnesses, e.g. a4_numerics.py). NOTE: attention_harness_common
make_qkv hardcodes head_dim=128 (`...,128)).astype(np.float16)` at its line 25) -- it was NOT extended
for arbitrary Hd (touching it would affect other consumers pinned to Hd=128); instead this script's
make_qkv_hd below constructs Q/K/V directly with the identical dtype (float16), device ("AMD"), and
distribution (normal(0, .04)) that make_qkv uses, just parametrized on Hd.

Decode calling convention mirrored from tinygrad/llm/model.py (~line 569, 596-599):
  - cache_kv layout: [2, B, Hkv, MAXC, Hd] fp16 (K at index 0, V at index 1; B=1 always for decode).
  - q is a SINGLE query position per head: shape (Hq, Hd) flattened head-major (matches
    q.reshape(binding.Hq, binding.Hd) at decode_routes.py:157).
  - staging="KV_BOTH" is the production default (decode_routes.py _FlashDecodeCandidate.staging);
    the executor's own default is staging="K_ONLY", which decode_routes.py comments say "assumes the
    old g5 V layout and was verified to produce bad logits on 8B" -- so this harness passes KV_BOTH
    explicitly to match the real shipped route, not the executor's raw default.
  - A decode step's query (freshly written at cache position TC-1) attends to ALL TC live cache slots
    (mask is all-zero / no masking needed -- causal is automatically satisfied since every cached
    token is <= the query's own position).

Run: PYTHONPATH=. DEV=AMD .venv/bin/python extra/qk/decode_hd_sweep_numerics.py
"""
from __future__ import annotations
import os
os.environ.setdefault("DEV", "AMD")

import traceback
import numpy as np

from tinygrad import Tensor, dtypes

from extra.qk.attention_harness_common import causal_mask, reference_attention
from extra.qk.flash_decode_attention_executor import flash_decode_live_split_block_tile

HQ, HKV = 32, 8          # real 8B GQA shape (G=Hq/Hkv=4); Hkv/G are orthogonal to the Hd question here
MAXC = 512               # synthetic ring/cache capacity
TC = 400                 # live context length at the decode step under test (TC <= MAXC)
SPLIT_COUNT = 4          # live-split S; production uses 48 at MAXC~=32k, 4 is adequate for MAXC=512


def make_qkv_hd(hq: int, hkv: int, hd: int, kv_tokens: int, seed: int):
  """Same dtype/layout/distribution as attention_harness_common.make_qkv, parametrized on Hd
  (make_qkv itself hardcodes Hd=128 -- see module docstring)."""
  rng = np.random.default_rng(seed)
  q = rng.normal(0, .04, (1, hq, 1, hd)).astype(np.float16)
  k = rng.normal(0, .04, (1, hkv, kv_tokens, hd)).astype(np.float16)
  v = rng.normal(0, .04, (1, hkv, kv_tokens, hd)).astype(np.float16)
  return (q, k, v)


def run_hd(hd: int) -> str:
  qn, kn, vn = make_qkv_hd(HQ, HKV, hd, TC, seed=20260724 + hd)

  # cache_kv: [2, B=1, Hkv, MAXC, Hd] fp16, matching tinygrad/llm/model.py's self.cache_kv layout.
  # Slots [0:TC) hold the live K/V; slots [TC:MAXC) are unwritten (zero, never read since Tc=TC gates it).
  cache_np = np.zeros((2, 1, HKV, MAXC, hd), dtype=np.float16)
  cache_np[0, 0, :, :TC, :] = kn[0]
  cache_np[1, 0, :, :TC, :] = vn[0]
  cache = Tensor(cache_np, device="AMD")

  # q flattened head-major to (Hq*Hd,), matching q.reshape(binding.Hq, binding.Hd) at decode_routes.py:157
  # (executor itself does q.reshape(Hq*Hd) at flash_decode_attention_executor.py:14).
  q_dev = Tensor(qn, device="AMD")  # (1, Hq, 1, Hd)

  out = flash_decode_live_split_block_tile(
    q_dev, cache, TC, hd, HQ, HKV, MAXC, SPLIT_COUNT, staging="KV_BOTH", fused_combine=True)
  got = out.numpy().astype(np.float32)  # (Hq, Hd); forces compile + lowering + execution

  q_ref = Tensor(qn, device="AMD")
  k_ref = Tensor(kn, device="AMD")
  v_ref = Tensor(vn, device="AMD")
  mask = causal_mask(1, TC, TC - 1)  # query at position TC-1 attends to all TC cached tokens -> all-zero mask
  ref = reference_attention(q_ref, k_ref, v_ref, mask, HQ, HKV).numpy().astype(np.float32).reshape(HQ, hd)

  max_abs = float(np.max(np.abs(got - ref)))
  ok = bool(np.allclose(got, ref, rtol=.03, atol=.006))
  return f"Hd={hd}: lowered=True max_abs_err={max_abs:.4g} {'PASS' if ok else 'FAIL'}"


if __name__ == "__main__":
  for hd in (64, 128, 192, 256):
    try:
      print(run_hd(hd))
    except Exception as e:
      msg = "".join(traceback.format_exception_only(type(e), e)).strip()
      print(f"Hd={hd}: LOWER_FAIL {type(e).__name__}: {msg[:160]}")
