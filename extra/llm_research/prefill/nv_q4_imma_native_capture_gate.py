#!/usr/bin/env python3
"""Full producer/main/fixup TinyJit capture/replay gate for the fixed pp512 ABI."""
from __future__ import annotations
import json, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import call_native, native_nv_program
from extra.llm_research.prefill.nv_q4_imma_provider import M,N,K,PARTIAL_SLOTS,compile_provider,provider_programs
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC as Q8_SOURCE

def main():
  dev=Device["NV"]
  provider=compile_provider(dev); mp,fp=provider_programs(provider)
  qlib=NVRTCCompiler(dev.arch,ptx=False,cache_key="q8_pp512_capture_v1").compile(Q8_SOURCE)
  # producer ABI is x,q8,scales,sums.
  qp=native_nv_program("q8_compact",qlib,global_size=(M,8,1),local_size=(128,1,1),
    globals=(0,1,2,3),outs=(1,2,3),ins=(0,))
  x=Tensor.zeros(M,K,device="NV").contiguous().realize()
  words=Tensor.zeros(N,K//256,36,dtype=dtypes.uint32,device="NV").contiguous().realize()
  q8=Tensor.empty(M*K,dtype=dtypes.int8,device="NV").realize()
  scales=Tensor.empty(M*(K//32),dtype=dtypes.float32,device="NV").realize()
  sums=Tensor.empty(M*(K//32),dtype=dtypes.float32,device="NV").realize()
  out=Tensor.empty(M*N,dtype=dtypes.float32,device="NV").realize()
  partials=Tensor.empty(PARTIAL_SLOTS*16384,dtype=dtypes.float32,device="NV").realize()
  ids=Tensor.empty(PARTIAL_SLOTS,dtype=dtypes.int32,device="NV").realize()
  slotmap=Tensor(provider.slotmap,device="NV").contiguous().realize()

  @TinyJit
  def chain(x,words,q8,scales,sums,out,partials,ids,slotmap):
    call_native(qp,x,q8,scales,sums)
    call_native(mp,out,partials,ids,words,q8,scales,sums)
    call_native(fp,out,partials,slotmap)
    return out

  times=[]
  for _ in range(12):
    st=time.perf_counter(); ret=chain(x,words,q8,scales,sums,out,partials,ids,slotmap);dev.synchronize();times.append((time.perf_counter()-st)*1e3)
  got=ret.numpy()
  result={"passed":bool(np.isfinite(got).all() and np.count_nonzero(got)==0),"finite":bool(np.isfinite(got).all()),
          "nonzero":int(np.count_nonzero(got)),"r9_ms":times[3:],"min_ms":float(min(times[3:])),
          "median_ms":float(np.median(times[3:])),"main_shared":provider.main.shmem_usage,
          "one_symbol_cubins":True}
  print(json.dumps(result,sort_keys=True))
  if not result["passed"]: raise SystemExit(1)

if __name__ == "__main__": main()
