#!/usr/bin/env python3
"""Exact Q4-down (block 4/type 12) FP16->Q8 producer gate."""
import argparse, json, pathlib, statistics
import numpy as np

from extra.llm_research.prefill.nv_q8_saved_z_helper import quantize_saved_z
from extra.llm_research.prefill.nv_q8_k12288_source import source_k12288, GRID, BLOCK

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--z', required=True); ap.add_argument('--out', required=True); ap.add_argument('--rounds',type=int,default=9); a=ap.parse_args()
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  z=np.load(a.z)
  if z.shape != (1,512,12288) or z.dtype != np.float16: raise RuntimeError(f'unexpected saved-z fixture {z.shape} {z.dtype}')
  # Match the CUDA producer's literal reciprocal (not host 1/127), while
  # retaining the shared helper's exact FP16-rounded sum contract.
  _,_,ru=quantize_saved_z(z)
  g=z.reshape(512,384,32).astype(np.float32); rs=(np.max(np.abs(g),axis=2)*np.float32(float.fromhex('0x1.020408p-7'))).astype(np.float32)
  # Independent host expression of the CUDA `v*(1.0f/d)` path.
  y=g*(np.float32(1.0)/rs[...,None])
  rq=np.rint(y)
  rq=np.clip(rq,-127,127).astype(np.int8).reshape(-1); rs=rs.reshape(-1)
  x=Tensor(z.reshape(-1),device='NV').contiguous().realize()
  q=Tensor.empty(512*12288,dtype=dtypes.int8,device='NV').realize(); s=Tensor.empty(512*384,dtype=dtypes.float32,device='NV').realize(); u=Tensor.empty(512*384,dtype=dtypes.float32,device='NV').realize()
  dev=Device['NV']; src=source_k12288(); lib=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_q8_saved_z_k12288_v1').compile(src)
  p=NVProgram(dev,'q8_compact_record_fp16_k12288',lib); bufs=tuple(t.uop.buffer.get_buf('NV') for t in (x,q,s,u))
  for _ in range(3): p(*bufs,global_size=GRID,local_size=BLOCK,wait=True)
  ts=[p(*bufs,global_size=GRID,local_size=BLOCK,wait=True)*1e6 for _ in range(a.rounds)]
  aq,as_,au=q.numpy(),s.numpy(),u.numpy(); badq=np.flatnonzero(aq!=rq); bads=np.flatnonzero(as_!=rs); badu=np.flatnonzero(au!=ru)
  rec={'schema':'tinygrad.nv_q8_saved_z_k12288_gate.v1','role':'blk.4.ffn_down.weight','ggml_type':12,'shape':{'M':512,'K':12288},'grid':GRID,'block':BLOCK,
   'correctness':{'finite':bool(np.isfinite(as_).all() and np.isfinite(au).all()),'q_exact':not len(badq),'q_mismatch':len(badq),'scale_exact':not len(bads),'scale_mismatch':len(bads),'sum_exact':not len(badu),'sum_mismatch':len(badu),'q_examples':badq[:8].tolist(),'scale_examples':bads[:8].tolist(),'sum_examples':badu[:8].tolist(),'scale_first':[(int(i),float(as_[i]),float(rs[i])) for i in bads[:8]]},
   'timing_us':{'min':min(ts),'median':statistics.median(ts),'max':max(ts),'samples':ts},'resources':{'registers':p.regs_usage,'shared_bytes':p.shmem_usage,'local_bytes':p.lcmem_usage},'source':'source_k12288 float4/thread, 128 threads, 24 K-segment CTAs per row'}
  rec['passed']=bool(rec['correctness']['finite'] and rec['correctness']['q_exact'] and rec['correctness']['scale_exact'] and rec['correctness']['sum_exact'])
  out=pathlib.Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec,indent=2))
  if not rec['passed']: raise SystemExit(1)
if __name__=='__main__': main()
