#!/usr/bin/env python3
"""Diagnostic pp512 Flash boundary/cache-after extractor; never changes policy."""
import argparse, hashlib, json, pathlib
import numpy as np

ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True); ap.add_argument('--max-context',type=int,default=1024); a=ap.parse_args()
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
import tinygrad.llm.fused_attention as fa
import tinygrad.llm.flash_prefill_attention as fpa
from tinygrad import Tensor
captured=[]; original_route=fa.route_prefill_attention; original_shared=fpa.shared_prefill_attention; original_sdpa=Tensor.scaled_dot_product_attention
def record(q,k,v):
  if not captured: captured.append(({'q_shape':list(q.shape),'q_dtype':str(q.dtype),'k_shape':list(k.shape),'k_dtype':str(k.dtype),
    'v_shape':list(v.shape),'v_dtype':str(v.dtype),'start_pos':0,'T':512,'B':1,'Hq':32,'Hkv':8,'Hd':128},q,k,v))
def tap_route(q,k,v,*args,**kw): record(q,k,v); return original_route(q,k,v,*args,**kw)
def tap_shared(q,k,v,*args,**kw): record(q,k,v); return original_shared(q,k,v,*args,**kw)
def tap_sdpa(q,k,v,*args,**kw): record(q,k,v); return original_sdpa(q,k,v,*args,**kw)
fa.route_prefill_attention=tap_route; fpa.shared_prefill_attention=tap_shared; Tensor.scaled_dot_product_attention=tap_sdpa
model=_load(a.model,a.max_context)
# Match the model-arm direct forward pattern: a concrete 512-token chunk at
# start_pos=0, with Flash explicitly requested at the call boundary.
tokens = __import__('tinygrad').Tensor([[(i*7)%1000 for i in range(512)]], dtype='int32').contiguous()
temp = __import__('tinygrad').Tensor([0.0])
model(tokens, 0, temp, use_flash=True, greedy=True).realize()
if not captured: raise RuntimeError('no Flash boundary captured; enable the production Flash route explicitly')
row,q,k,v=captured[0]; q.realize(); k.realize(); v.realize(); qa,ka,va=np.asarray(q.numpy()),np.asarray(k.numpy()),np.asarray(v.numpy())
row.update({'q_strides_bytes':list(qa.strides),'k_strides_bytes':list(ka.strides),'v_strides_bytes':list(va.strides),
  'q_sha256':hashlib.sha256(qa.tobytes()).hexdigest(),'k_sha256':hashlib.sha256(ka.tobytes()).hexdigest(),'v_sha256':hashlib.sha256(va.tobytes()).hexdigest(),
  'q_finite':bool(np.isfinite(qa).all()),'k_finite':bool(np.isfinite(ka).all()),'v_finite':bool(np.isfinite(va).all())}); pathlib.Path(a.out).write_text(json.dumps({'schema':'tinygrad.nv_pp512_flash_cache_after.v1','capture':row},indent=2,sort_keys=True)+'\n'); print(json.dumps(row,indent=2,sort_keys=True))
