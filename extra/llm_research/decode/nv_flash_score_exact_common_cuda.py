#!/usr/bin/env python3
"""Common CUDA-driver timing gate for exact tinygrad and llama Flash cubins."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, math, pathlib, statistics
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check

LLAMA_SYMBOL = "_Z18flash_attn_ext_vecILi128ELi1EL9ggml_type1ELS0_1ELb0EEvPKcS2_S2_S2_S2_PKiPfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil"

class UInt3(ctypes.Structure): _fields_ = [("x",ctypes.c_uint32),("y",ctypes.c_uint32),("z",ctypes.c_uint32)]

def holder(v, typ): return typ(v)
def params(values): return (ctypes.c_void_p*len(values))(*[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in values])
def alloc(size):
  p=cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(p),size)); check(cuda.cuMemsetD8_v2(p,0,size)); return p
def load(blob, symbol):
  m=cuda.CUmodule(); f=cuda.CUfunction(); check(cuda.cuModuleLoadData(ctypes.byref(m),blob)); check(cuda.cuModuleGetFunction(ctypes.byref(f),m,symbol.encode())); return m,f
def launch(f,grid,block,p,stream): check(cuda.cuLaunchKernel(f,*grid,*block,0,stream,p,None))

def event_batch(fn, stream, warmup, reps):
  for _ in range(warmup): fn()
  check(cuda.cuStreamSynchronize(stream)); a,b=cuda.CUevent(),cuda.CUevent()
  check(cuda.cuEventCreate(ctypes.byref(a),0));check(cuda.cuEventCreate(ctypes.byref(b),0));check(cuda.cuEventRecord(a,stream))
  for _ in range(reps): fn()
  check(cuda.cuEventRecord(b,stream));check(cuda.cuEventSynchronize(b)); ms=ctypes.c_float();check(cuda.cuEventElapsedTime(ctypes.byref(ms),a,b))
  cuda.cuEventDestroy_v2(a);cuda.cuEventDestroy_v2(b);return 1000*ms.value/reps

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--tiny-cubin",type=pathlib.Path,required=True);ap.add_argument("--llama-cubin",type=pathlib.Path,required=True)
  ap.add_argument("--tiny-symbol",required=True);ap.add_argument("--reps",type=int,default=2000);ap.add_argument("--warmup",type=int,default=200)
  ap.add_argument("--condition-mib",type=int,default=0)
  ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  check(cuda.cuInit(0));dev=ctypes.c_int();check(cuda.cuDeviceGet(ctypes.byref(dev),0));ctx=cuda.CUcontext();check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx),dev));check(cuda.cuCtxSetCurrent(ctx))
  tm,tf=load(a.tiny_cubin.read_bytes(),a.tiny_symbol);lm,lf=load(a.llama_cubin.read_bytes(),LLAMA_SYMBOL)
  q,cache,tout,ldst,lmeta=alloc(32*128*4),alloc(2*8*1024*128*2),alloc(102400),alloc(32*6*128*4),alloc(32*6*8)
  zero=cuda.CUdeviceptr(0); v=cuda.CUdeviceptr(int(cache.value)+8*1024*128*2); tc=holder(768,ctypes.c_int32)
  tp=params([tout,q,cache,tc])
  # Exact release ABI. Physical extent is 768, so no tail mask is required.
  vals=[q,cache,v,zero,zero,zero,ldst,lmeta,
    holder(1/math.sqrt(128),ctypes.c_float),holder(0,ctypes.c_float),holder(1,ctypes.c_float),holder(1,ctypes.c_float),
    holder(32,ctypes.c_uint32),holder(0,ctypes.c_float),holder(128,ctypes.c_int32),UInt3(1,0,1),holder(32,ctypes.c_int32),holder(1,ctypes.c_int32),
    holder(512,ctypes.c_int32),holder(512,ctypes.c_int32),holder(16384,ctypes.c_int32),
    holder(128,ctypes.c_int32),holder(768,ctypes.c_int32),holder(8,ctypes.c_int32),holder(1,ctypes.c_int32),
    holder(256,ctypes.c_int32),holder(262144,ctypes.c_int64),holder(2097152,ctypes.c_int32),
    holder(256,ctypes.c_int32),holder(262144,ctypes.c_int64),holder(2097152,ctypes.c_int32),
    holder(0,ctypes.c_int32),holder(0,ctypes.c_int32),holder(0,ctypes.c_int32),holder(0,ctypes.c_int32),holder(0,ctypes.c_int32),holder(0,ctypes.c_int64)]
  lp=params(vals);stream=cuda.CUstream();check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
  cm=cf=None; condition_bufs=[]; cp=None
  if a.condition_mib:
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
    cb=NVRTCCompiler("sm_120a",ptx=False,cache_key="flash_exact_condition").compile(r'''extern "C" __global__ void condition(const float *s, unsigned long long n, float *o) { unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float v=s[i];if(v<0)o[0]=v;} }''')
    cm,cf=load(cb,"condition");condition_bufs=[alloc(a.condition_mib<<20),alloc(4)];cp=params([condition_bufs[0],holder((a.condition_mib<<20)//4,ctypes.c_uint64),condition_bufs[1]])
  targets={"tiny":lambda:launch(tf,(6,32,1),(32,4,1),tp,stream),"llama":lambda:launch(lf,(1,6,32),(32,4,1),lp,stream)}
  def seq(name):
    targets[name]()
    if cf is not None:
      launch(cf,(((a.condition_mib<<20)//4+255)//256,1,1),(256,1,1),cp,stream);targets[name]()
  fns={name:(lambda n=name:seq(n)) for name in targets}
  rows=[]
  for order in (("tiny","llama"),("llama","tiny"),("tiny","llama"),("llama","tiny")):
    for name in order: rows.append({"arm":name,"event_us":event_batch(fns[name],stream,a.warmup,a.reps)})
  check(cuda.cuStreamSynchronize(stream))
  raw=(ctypes.c_float*(32*6*128))();check(cuda.cuMemcpyDtoH_v2(ctypes.byref(raw),ldst,ctypes.sizeof(raw)))
  finite=all(math.isfinite(x) for x in raw);zero_output=all(x==0 for x in raw)
  traw=(ctypes.c_float*(32*6*130))();check(cuda.cuMemcpyDtoH_v2(ctypes.byref(traw),tout,ctypes.sizeof(traw)))
  tiny_finite=all(math.isfinite(x) for x in traw)
  summary={name:{"median_event_us":statistics.median(r["event_us"] for r in rows if r["arm"]==name),"samples":[r["event_us"] for r in rows if r["arm"]==name]} for name in fns}
  result={"schema":"tinygrad.nv_flash_score_exact_common_cuda.v1","logical_tc":768,"condition_mib":a.condition_mib,"zero_inputs":True,
    "tiny_partial_finite":tiny_finite,"llama_output_finite":finite,"llama_output_zero":zero_output,
    "tiny_cubin_sha256":hashlib.sha256(a.tiny_cubin.read_bytes()).hexdigest(),"llama_cubin_sha256":hashlib.sha256(a.llama_cubin.read_bytes()).hexdigest(),"rows":rows,"summary":summary}
  result["tiny_minus_llama_us"]=summary["tiny"]["median_event_us"]-summary["llama"]["median_event_us"]
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
  for p in (q,cache,tout,ldst,lmeta,*condition_bufs):cuda.cuMemFree_v2(p)
  if cm is not None:cuda.cuModuleUnload(cm)
  cuda.cuStreamDestroy_v2(stream);cuda.cuModuleUnload(tm);cuda.cuModuleUnload(lm);cuda.cuDevicePrimaryCtxRelease(dev)
if __name__=="__main__":main()
