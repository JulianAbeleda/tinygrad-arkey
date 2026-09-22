#!/usr/bin/env python3
"""Bitwise fp16-input compact-Q8 qualification against fp32 producer."""
import hashlib, json, statistics
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_q8_compact_producer_gate import M,K,SRC,SRC_FP16

def sha(x): return hashlib.sha256(x.tobytes()).hexdigest()
def main():
  rng=np.random.default_rng(20260828); a=(rng.standard_normal((M,K))*0.2).astype(np.float16)
  # Adversarial exact half values, zeros and extrema.
  a.reshape(-1)[:8]=np.array([0,-0.,1,-1,2**-14,-2**-14,31.5,-31.5],np.float16)
  xh=Tensor(a,device="NV").contiguous().realize(); xf=xh.cast(dtypes.float32).contiguous().realize()
  dev=Device["NV"]
  l0=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_compact_fp32_control_v2").compile(SRC)
  l1=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_compact_fp16_input_v1").compile(SRC_FP16)
  p0,p1=NVProgram(dev,"q8_compact",l0),NVProgram(dev,"q8_compact_fp16",l1)
  def outs(): return (Tensor.full(M*K,77,dtype=dtypes.int8,device="NV").realize(),
    Tensor.full(M*K//32,np.nan,dtype=dtypes.float32,device="NV").realize(),
    Tensor.full(M*K//32,np.nan,dtype=dtypes.float32,device="NV").realize())
  o0,o1=outs(),outs(); buf=lambda t:t.uop.buffer.get_buf("NV"); grid=(M,8,1);block=(128,1,1)
  p0(buf(xf),*[buf(x) for x in o0],global_size=grid,local_size=block,wait=True)
  before=sha(xh.numpy()); ts=[]
  for _ in range(12): ts.append(p1(buf(xh),*[buf(x) for x in o1],global_size=grid,local_size=block,wait=True)*1e6)
  aa=[x.numpy() for x in o0];bb=[x.numpy() for x in o1]
  rec={"passed":all(np.array_equal(x,y) for x,y in zip(aa,bb)) and before==sha(xh.numpy()),
       "exact":{"q":bool(np.array_equal(aa[0],bb[0])),"scales":bool(np.array_equal(aa[1],bb[1])),"sums":bool(np.array_equal(aa[2],bb[2]))},
       "readonly":before==sha(xh.numpy()),"finite":bool(np.isfinite(bb[1]).all() and np.isfinite(bb[2]).all()),
       "r9_us":ts[3:],"min_us":min(ts[3:]),"median_us":statistics.median(ts[3:]),
       "resources":{"regs":p1.regs_usage,"shared":p1.shmem_usage,"local":p1.lcmem_usage}}
  print(json.dumps(rec,sort_keys=True));
  if not rec["passed"]: raise SystemExit(1)
if __name__=="__main__":main()
