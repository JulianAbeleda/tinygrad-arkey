#!/usr/bin/env python3
"""Exact synthetic oracle for llama's fused Q4_K attention-O residual epilogue."""
from __future__ import annotations
import argparse, ctypes, json, pathlib, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from scratchpad.llama_cuda_quantized_live_oracle import (_cpu_quantizers, pack_q4, decode_q4, decode_q8, device_pointer,
  ENTRY_Q8, FusionArgs, UInt3, fastdiv_values)
from scratchpad.cuda_decode_q4_attention_o_llama_graph_ab import ENTRY_Q4_FUSED, Q4, Q8

BASE=pathlib.Path("/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0")
def params(args): return (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in args])
def copy_f32(b):
  raw=bytearray(b.nbytes); b.copyout(memoryview(raw)); return np.frombuffer(raw,dtype=np.float32).copy()
def run(rows=4096,k=4096):
  q4,_,_= _cpu_quantizers(BASE); rng=np.random.default_rng(20260804)
  wf=rng.normal(0,.2,size=(rows,k)).astype(np.float32); xf=rng.normal(0,.2,size=k).astype(np.float32); residual=rng.normal(0,.2,size=rows).astype(np.float32)
  wq=pack_q4(wf,q4)
  weight=Buffer("CUDA",len(wq),dtypes.uint8,initial_value=bytearray(wq)); x=Buffer("CUDA",k,dtypes.float,initial_value=bytearray(xf.tobytes()))
  q8=Buffer("CUDA",k//32*36,dtypes.uint8); out=Buffer("CUDA",rows,dtypes.float); bias=Buffer("CUDA",rows,dtypes.float,initial_value=bytearray(residual.tobytes()))
  for b in (weight,x,q8,out,bias): b.ensure_allocated()
  qm,qf,mm,mf,stream=cuda.CUmodule(),cuda.CUfunction(),cuda.CUmodule(),cuda.CUfunction(),cuda.CUstream()
  try:
    check(cuda.cuModuleLoad(ctypes.byref(qm),str(Q8).encode())); check(cuda.cuModuleGetFunction(ctypes.byref(qf),qm,ENTRY_Q8.encode()))
    check(cuda.cuModuleLoad(ctypes.byref(mm),str(Q4).encode())); check(cuda.cuModuleGetFunction(ctypes.byref(mf),mm,ENTRY_Q4_FUSED.encode())); check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
    qa=[device_pointer(x),device_pointer(q8),ctypes.c_int64(k),ctypes.c_int64(k),ctypes.c_int64(k),ctypes.c_int64(k),ctypes.c_int64(k),ctypes.c_uint32(1),fastdiv_values(1)]
    z,one=UInt3(0,0,0),fastdiv_values(1)
    ma=[device_pointer(weight),device_pointer(q8),ctypes.c_void_p(),FusionArgs(device_pointer(bias),None,None,0),device_pointer(out),ctypes.c_uint32(k),z,
      ctypes.c_uint32(k//256),ctypes.c_uint32(k//32),ctypes.c_uint32(rows),one,ctypes.c_uint32(rows*(k//256)),ctypes.c_uint32(k//32),ctypes.c_uint32(rows),one,
      ctypes.c_uint32(rows*(k//256)),ctypes.c_uint32(k//32),ctypes.c_uint32(rows),ctypes.c_uint32(0)]
    check(cuda.cuLaunchKernel(qf,(k+255)//256,1,1,256,1,1,0,stream,params(qa),None)); check(cuda.cuLaunchKernel(mf,rows,1,1,32,4,1,0,stream,params(ma),None)); check(cuda.cuStreamSynchronize(stream))
    q8raw=bytearray(q8.nbytes); q8.copyout(memoryview(q8raw))
    reference=decode_q4(wq).reshape(rows,k) @ decode_q8(q8raw) + residual
    got=copy_f32(out); err=np.abs(got-reference)
    return {"schema":"tinygrad.llama_q4_attention_o_oracle.v1","shape":{"rows":rows,"k":k},"semantic":{"formula":"Q4_K(W) @ x + residual","fusion":{"x_bias":"f32[rows]","gate":None,"gate_bias":None,"glu_op":0}},"abi":{"entry":ENTRY_Q4_FUSED,"grid":[rows,1,1],"block":[32,4,1],"dynamic_shared":0},"correctness":{"max_abs":float(err.max()),"rmse":float(np.sqrt(np.mean(err.astype(np.float64)**2))),"argmax_equal":bool(got.argmax()==reference.argmax()),"verdict":"PASS" if err.max()<1e-3 else "FAIL"}}
  finally:
    for m in (qm,mm):
      if m: cuda.cuModuleUnload(m)
def main():
  p=argparse.ArgumentParser(); p.add_argument("--out",required=True); a=p.parse_args(); o=run(); pathlib.Path(a.out).write_text(json.dumps(o,indent=2)+"\n"); print(json.dumps(o,sort_keys=True))
if __name__=="__main__": main()
