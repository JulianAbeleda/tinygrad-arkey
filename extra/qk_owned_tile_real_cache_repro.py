#!/usr/bin/env python3
"""Owned-tile real-cache revalidation probe (docs/owned-amdgcn-tile-real-cache-revalidation-scope-20260623.md).
Token-correctness comparison on REAL in-model fp32 cache: gqa baseline vs owned tile (fp32) vs owned tile + fp16 cast.
  run: DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 PYTHONPATH=. .venv/bin/python extra/qk_owned_tile_real_cache_repro.py
"""
import json, os, pathlib, subprocess, sys
OUT = pathlib.Path(__file__).resolve().parents[1] / "bench/qk-owned-amdgcn-tile-real-cache"

CHILD = r'''
import os, numpy as np
from tinygrad import Tensor, UOp
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_harness_contract import DEFAULT_MODEL
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,4608,seed=20260617)
for lin in (getattr(m,'_q4k_linears',None).linears if getattr(m,'_q4k_linears',None) else []): lin.decode_enabled=True
for b in m.blk: b._use_flash,b._prefill_v2=True,False
v=UOp.variable("start_pos",0,4607); temp=Tensor([0.0])
ids=((tok.prefix() if hasattr(tok,'prefix') else [])+tok.encode("the quick brown fox jumps. "*500))[:2048]
o=None; sp=0
for st in range(0,len(ids),512): o=m.forward(Tensor([ids[st:st+512]],dtype="int32").contiguous(),sp,temp).realize(); sp+=len(ids[st:st+512])
print("CACHE_DTYPE", m.blk[0].cache_kv.dtype)
toks=[int(o.item())]
for _ in range(8): o=m.forward(Tensor([[toks[-1]]],dtype="int32").contiguous(),v.bind(sp),temp).realize(); toks.append(int(o.item())); sp+=1
print("TOKENS", toks[1:])  # decode tokens (drop prefill output)
'''

def run(label, env):
    e = dict(os.environ); e.update(env); e["PYTHONPATH"]="."
    p = subprocess.run([".venv/bin/python","-c",CHILD], capture_output=True, text=True, env=e)
    toks, dt = None, None
    for ln in p.stdout.splitlines():
        if ln.startswith("TOKENS"): toks=json.loads(ln[6:])
        if ln.startswith("CACHE_DTYPE"): dt=ln.split(None,1)[1]
    return {"label":label, "cache_dtype":dt, "decode_tokens":toks, "ok": toks is not None}

base = dict(DEV="AMD", JIT="1", Q4K_GEMV_WARP="1", Q4K_GEMV_WARP_DOWN="1")
res = {"date":"2026-06-23","phase":"OWNED_TILE_REAL_CACHE_REPRO","gpu":"gfx1100"}
res["gqa_baseline"]        = run("gqa", base)
res["owned_tile_fp32"]     = run("owned_tile_fp32_cache", {**base, "DECODE_ATTN_AMDGCN_TILE":"1"})
res["owned_tile_fp16cast"] = run("owned_tile_fp16cast",   {**base, "DECODE_ATTN_AMDGCN_TILE":"1", "DECODE_ATTN_AMDGCN_FP16CAST":"1"})
gqa = res["gqa_baseline"]["decode_tokens"]
def match(r): return r["decode_tokens"]==gqa if (r["decode_tokens"] and gqa) else None
res["owned_tile_fp32"]["matches_gqa"] = match(res["owned_tile_fp32"])
res["owned_tile_fp16cast"]["matches_gqa"] = match(res["owned_tile_fp16cast"])
res["verdict"] = ("OWNED_TILE_REAL_CACHE_FAIL_REPRODUCED" if res["owned_tile_fp32"]["matches_gqa"] is False else "OWNED_TILE_REAL_CACHE_FAIL_NOT_REPRODUCED")
res["fp16cast_fixes"] = res["owned_tile_fp16cast"]["matches_gqa"]
OUT.mkdir(parents=True, exist_ok=True); (OUT/"repro.json").write_text(json.dumps(res,indent=2))
print("gqa:        ", gqa, file=sys.stderr)
print("owned fp32: ", res["owned_tile_fp32"]["decode_tokens"], "matches_gqa=", res["owned_tile_fp32"]["matches_gqa"], file=sys.stderr)
print("owned fp16cast:", res["owned_tile_fp16cast"]["decode_tokens"], "matches_gqa=", res["owned_tile_fp16cast"]["matches_gqa"], file=sys.stderr)
print("verdict:", res["verdict"], "| fp16cast_fixes:", res["fp16cast_fixes"], file=sys.stderr)
