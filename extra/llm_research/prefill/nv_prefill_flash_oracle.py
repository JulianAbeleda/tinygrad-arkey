#!/usr/bin/env python3
"""Independent pp512 Flash score/reduction oracle (no decode route reuse)."""
import json, pathlib, numpy as np

def reference(q,k,v,hq=32,hkv=8,start_pos=0):
  # Explicit GQA expansion, causal lower-right mask, fp32 score/reduction.
  g=hq//hkv; q=q.astype(np.float32); k=np.repeat(k.astype(np.float32),g,axis=1)
  v=np.repeat(v.astype(np.float32),g,axis=1); scores=q@k.transpose(0,1,3,2)/np.sqrt(q.shape[-1])
  tq,tk=q.shape[2],k.shape[2]; allowed=np.arange(tk)[None,:] <= (start_pos+np.arange(tq))[:,None]
  scores=np.where(allowed[None,None],scores,-np.inf); scores-=np.max(scores,axis=-1,keepdims=True)
  p=np.exp(scores); p/=np.sum(p,axis=-1,keepdims=True); return p@v

def main():
  rng=np.random.default_rng(20260829); q=rng.normal(0,.04,(1,32,512,128)).astype(np.float16)
  k=rng.normal(0,.04,(1,8,512,128)).astype(np.float16); v=rng.normal(0,.04,(1,8,512,128)).astype(np.float16)
  out=reference(q,k,v); p=pathlib.Path('docs/task_workflow/evidence/nv-prefill-flash-20260829'); p.mkdir(parents=True,exist_ok=True)
  np.savez(p/'oracle.npz',q=q,k=k,v=v,out=out); rec={'schema':'tinygrad.nv_prefill_flash_oracle.v1','status':'PASS','shape':{'q_tokens':512,'kv_tokens':512,'hq':32,'hkv':8,'hd':128},'causal':True,'start_pos':0,'gqa_group':4,'score_dtype':'fp32','reduction_dtype':'fp32','output_shape':list(out.shape),'finite':bool(np.isfinite(out).all()),'oracle':str(p/'oracle.npz')}; (p/'oracle.json').write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec))
if __name__=='__main__': main()
