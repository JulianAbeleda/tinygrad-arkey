#!/usr/bin/env python3
"""Capture representative diagonal activation energy at one dense FFN entry."""
from __future__ import annotations
import argparse,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt

def install(rows,block):
  import tinygrad.runtime.ops_nv as nv
  orig=nv.NVProgram.__call__;seen=0
  def call(self,*bufs,global_size=(1,1,1),local_size=(1,1,1),vals=(),wait=False,timeout=None):
    nonlocal seen
    sizes=[int(getattr(b,"size",getattr(getattr(b,"_buf",None),"size",0))) for b in bufs]
    family=len(sizes)>=4 and sizes[:4]==[24576,28311552,28311552,8192]
    selected=family and seen==block
    if family:seen+=1
    ret=orig(self,*bufs,global_size=global_size,local_size=local_size,vals=vals,wait=wait or selected,timeout=timeout)
    if selected:
      host=memoryview(bytearray(8192));self.dev.allocator._copyout(host,bufs[3]);rows.append(np.frombuffer(host,dtype=np.float16).astype(np.float32))
    return ret
  nv.NVProgram.__call__=call
  return nv,orig

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf");ap.add_argument("--block",type=int,default=0);ap.add_argument("--depth",type=int,default=128);ap.add_argument("--max-context",type=int,default=512);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  model=_load(a.model,a.max_context);rows=[];nv,orig=install(rows,a.block)
  gen=model.generate(_prompt(a.model,a.depth),chunk_size=32,temperature=0.0)
  try:
    for _ in range(4):next(gen)
  finally:gen.close();nv.NVProgram.__call__=orig
  if not rows:raise RuntimeError("no FFN activation captured")
  x=np.concatenate([z.reshape(-1,z.shape[-1]) for z in rows],axis=0);h=np.mean(x*x,axis=0);a.out.parent.mkdir(parents=True,exist_ok=True);np.save(a.out,h)
  print(json.dumps({"block":a.block,"samples":len(x),"features":len(h),"finite":bool(np.isfinite(h).all()),"min":float(h.min()),"median":float(np.median(h)),"max":float(h.max())}))
if __name__=="__main__":raise SystemExit(main())
