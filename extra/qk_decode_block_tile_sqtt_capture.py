#!/usr/bin/env python3
"""Minimal SQTT capture of the route-bound block tile -> profile.pkl (parse with extra/sqtt/roc.py --kernel ...).
Confirms (dynamically) whether the per-token ds_bpermute reduce latency is EXPOSED (stall-bound) -- the scheduling
thesis the static hotloop diff couldn't resolve and the cycle budget (281 vs 22 cyc/token) points at.
Run: DEV=AMD JIT=1 PROFILE=1 SQTT=1 SQTT_ITRACE_SE_MASK=1 PYTHONPATH=. python extra/qk_decode_block_tile_sqtt_capture.py"""
import os
os.environ.update({
  "DECODE_ATTN_AMDGCN_TILE": "0", "DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE": "1",
  "DECODE_ATTN_BLOCK_TILE": "1", "DECODE_STAGE_COALESCE": "4", "COALESCED_LOAD_LOWERING": "1", "SCHED_UNROLL": "8",
  "SCHED_LIST": "1", "DECODE_FAST_EXP2": "1",
})
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from extra.qk_harness_contract import DEFAULT_MODEL
from extra.llm_generate import load_model_and_tokenizer

MAXC, CTX = 2048, 1024
m, tok = load_model_and_tokenizer(DEFAULT_MODEL, MAXC, seed=20260617)
for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 400)
ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
for b in m.blk: b._use_flash, b._prefill_v2 = True, False
tokid = int(ids[CTX])
step = TinyJit(m.forward)
out = Tensor([[tokid]], dtype="int32").contiguous()
for i in range(5): out = step(out, v_sp.bind(CTX + i), temp).realize()   # warmup: compile (NOT captured)
import tinygrad.runtime.ops_amd  # noqa
with Context(SQTT=1, PROFILE=1):   # capture ONE eager forward -> profile.pkl at exit
  from tinygrad import GlobalCounters
  GlobalCounters.reset()
  m.forward(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(CTX), temp).realize()
print("SQTT_CAPTURE_DONE ctx", CTX)
