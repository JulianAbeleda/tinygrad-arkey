#!/usr/bin/env python3
"""Shared attention-harness runtime primitives, centralized from extra/qk/benchmark_shared_attention.py."""
from __future__ import annotations
import hashlib, json, statistics, time
from pathlib import Path
from typing import Any
import numpy as np
from tinygrad import Tensor, dtypes, Device

ROUTES=(("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",32,8),("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",40,8))

def content_sha(text:str) -> str: return hashlib.sha256(text.encode()).hexdigest()
def causal_mask(q_tokens:int, kv_tokens:int, start_pos:int) -> Tensor:
  return Tensor.full((1,1,q_tokens,kv_tokens),float("-inf"),dtype=dtypes.float16,buffer=False).triu(start_pos+1)
def amd_sync() -> None: Device["AMD"].synchronize()
def load_shared_attention_proof(path:Path):
  p=json.loads(path.read_text())
  schemas={"tinygrad.shared_attention_proof.v2","tinygrad.shared_attention_proof.acc_slice_v3",
           "tinygrad.shared_attention_proof.phase_v4"}
  if p.get("schema") not in schemas or p.get("status")!="PASS" or p.get("passed") is not True:
    raise ValueError("aggregate shared-attention proof is not PASS")
  return p
def make_qkv(hq:int, hkv:int, q_tokens:int, kv_tokens:int, seed:int):
  rng=np.random.default_rng(seed)
  vals=[rng.normal(0,.04,(1,h,q_tokens if n==0 else kv_tokens,128)).astype(np.float16) for n,h in enumerate((hq,hkv,hkv))]
  return tuple(Tensor(x,device="AMD") for x in vals), vals
def reference_attention(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, hq:int, hkv:int) -> Tensor:
  g=hq//hkv
  return q.scaled_dot_product_attention(k.repeat_interleave(g,dim=-3),v.repeat_interleave(g,dim=-3),attn_mask=mask)
def synced_time(fn, warmup:int, samples:int):
  for _ in range(warmup): fn().realize()
  amd_sync(); out=[]
  for _ in range(samples):
    amd_sync(); st=time.perf_counter_ns(); fn().realize(); amd_sync(); out.append((time.perf_counter_ns()-st)/1e6)
  return out
def timing_summary(values):
  x=sorted(values); return {"raw_ms":x,"median_ms":statistics.median(x),"p10_ms":np.percentile(x,10).item(),"p90_ms":np.percentile(x,90).item()}
def candidate_context(profile, strategy, hq, hkv, kv, *, q_tokens=512, hd=128, causal=True, start_pos=None):
  from tinygrad.uop.ops import SharedAttentionCandidateContext
  sp = kv - q_tokens if start_pos is None else start_pos
  return SharedAttentionCandidateContext(profile, strategy, q_tokens, kv, sp, hq, hkv, hd, causal).validate()
