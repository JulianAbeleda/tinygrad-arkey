#!/usr/bin/env python3
"""Attention-only, proof-gated AMD benchmark for shared prefill attention."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
from tinygrad.uop.ops import SharedAttentionCandidateContext

SCHEMA="tinygrad.shared_attention_benchmark.v1"
ROUTES=(("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",32,8),("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",40,8))
def _sha(x): return hashlib.sha256(x.encode()).hexdigest()
def _mask(q,kv,start): return Tensor.full((1,1,q,kv),float("-inf"),dtype=dtypes.float16,buffer=False).triu(start+1)
def _sync(): Device["AMD"].synchronize()
def _proof(path:Path):
  p=json.loads(path.read_text())
  if p.get("schema")!="tinygrad.shared_attention_proof.v2" or p.get("status")!="PASS" or p.get("passed") is not True: raise ValueError("aggregate shared-attention proof is not PASS")
  return p
def _inputs(hq,hkv,q,kv,seed):
  rng=np.random.default_rng(seed)
  vals=[rng.normal(0,.04,(1,h,q if n==0 else kv,128)).astype(np.float16) for n,h in enumerate((hq,hkv,hkv))]
  return tuple(Tensor(x,device="AMD") for x in vals), vals
def _candidate(q,k,v,ctx): return shared_prefill_attention(q,k,v,mask=_mask(ctx.q_tokens,ctx.kv_tokens,ctx.start_pos),candidate_context=ctx)
def _baseline(q,k,v,ctx):
  g=ctx.hq//ctx.hkv
  return q.scaled_dot_product_attention(k.repeat_interleave(g,dim=-3),v.repeat_interleave(g,dim=-3),attn_mask=_mask(ctx.q_tokens,ctx.kv_tokens,ctx.start_pos))
def _time(fn,warmup,samples):
  for _ in range(warmup): fn().realize()
  _sync(); out=[]
  for _ in range(samples):
    _sync(); st=time.perf_counter_ns(); fn().realize(); _sync(); out.append((time.perf_counter_ns()-st)/1e6)
  return out
def _summary(x):
  x=sorted(x); return {"raw_ms":x,"median_ms":statistics.median(x),"p10_ms":np.percentile(x,10).item(),"p90_ms":np.percentile(x,90).item()}
def run(proof_path:Path,out:Path,*,samples:int,warmup:int,contexts:tuple[int,...]):
  proof=_proof(proof_path); rows=[]
  for profile,strategy,hq,hkv in ROUTES:
    for kv in contexts:
      ctx=SharedAttentionCandidateContext(profile,strategy,512,kv,kv-512,hq,hkv,128,True).validate()
      (q,k,v),raw=_inputs(hq,hkv,512,kv,20260723+hq+kv)
      cand=lambda:_candidate(q,k,v,ctx); base=lambda:_baseline(q,k,v,ctx)
      # Numeric gate is deliberately before any timed iteration.
      cg,bg=cand().numpy().astype(np.float32),base().numpy().astype(np.float32)
      if not np.allclose(cg,bg,rtol=.03,atol=.006): raise RuntimeError(f"numeric gate failed for {profile} KV={kv}: max_abs={np.abs(cg-bg).max()}")
      c,b=_summary(_time(cand,warmup,samples)),_summary(_time(base,warmup,samples))
      c["tokens_s"]=512000/c["median_ms"]; b["tokens_s"]=512000/b["median_ms"]
      rows.append({"candidate_context":dict(zip(ctx._fields,ctx)),"candidate":c,"baseline":b,"speedup":b["median_ms"]/c["median_ms"],
        "numeric_max_abs":float(np.abs(cg-bg).max()),"input_sha256":_sha("".join(str(x.shape)+str(x.sum()) for x in raw))})
  artifact={"schema":SCHEMA,"proof_path":str(proof_path),"proof_sha256":_sha(proof_path.read_text()),"config":{"samples":samples,"warmup":warmup,"contexts":contexts,"q_tokens":512,"dtype":"float16","device":"AMD"},"hardware":{"platform":platform.platform(),"rocm_visible_devices":os.getenv("ROCR_VISIBLE_DEVICES")},"assumptions":{"flops":"4*Hq*Q*KV*Hd","bytes":"fp16 QKV/output; excludes cache and compiler"},"rows":rows}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(artifact,sort_keys=True,indent=2)+"\n"); return artifact
def main():
  p=argparse.ArgumentParser(); p.add_argument("--proof",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--samples",type=int,default=10); p.add_argument("--warmup",type=int,default=3); p.add_argument("--contexts",default="512,1024,2048,4096")
  a=p.parse_args();
  if a.samples<10 or a.warmup<1: raise ValueError("require >=10 samples and >=1 warmup")
  run(a.proof,a.output,samples=a.samples,warmup=a.warmup,contexts=tuple(int(x) for x in a.contexts.split(",")))
if __name__=="__main__": main()
