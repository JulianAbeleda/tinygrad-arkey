#!/usr/bin/env python3
"""Exact extracted llama fused-residual FFN-down MMVQ oracle (diagnostic only)."""
from __future__ import annotations
import argparse, ctypes, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from scratchpad.llama_cuda_quantized_live_oracle import (FusionArgs, UInt3, device_pointer, pack_q4, pack_q6, pack_q8, decode_q8)

ROOT=pathlib.Path(__file__).resolve().parents[1]
CUBIN=ROOT/"scratchpad/llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin"
BASE=pathlib.Path("/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0")
ENTRIES={
  "Q4_K":"_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
  "Q6_K":"_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
}

def cpu_fns():
  lib=ctypes.CDLL(str(BASE),mode=ctypes.RTLD_LOCAL)
  out={}
  for kind,suffix in (("Q4_K","q4_K"),("Q6_K","q6_K")):
    q=getattr(lib,"quantize_row_"+suffix+"_ref"); dq=getattr(lib,"dequantize_row_"+suffix)
    q.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64];q.restype=None
    dq.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_int64];dq.restype=None
    out[kind]=(q,dq)
  q8=lib.quantize_row_q8_1_ref;q8.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64];q8.restype=None
  return out,q8

def dequant_rows(payload:bytes,rows:int,k:int,block_bytes:int,fn):
  out=np.empty((rows,k),np.float32); row_bytes=k//256*block_bytes
  for r in range(rows):
    src=(ctypes.c_uint8*row_bytes).from_buffer_copy(payload[r*row_bytes:(r+1)*row_bytes])
    fn(src,out[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),k)
  return out

def ptrs(args): return (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in args])

def run(kind="Q4_K",rows=4096,k=12288):
  fns,q8fn=cpu_fns(); qfn,dqfn=fns[kind]; rng=np.random.default_rng(20260804+(0 if kind=="Q4_K" else 1))
  wf=rng.normal(0,.1,(rows,k)).astype(np.float32); xf=rng.normal(0,.1,k).astype(np.float32); bias=rng.normal(0,.1,rows).astype(np.float32)
  wp=(pack_q4(wf,qfn) if kind=="Q4_K" else pack_q6(wf,qfn)); xp=pack_q8(xf,q8fn)
  w=dequant_rows(wp,rows,k,144 if kind=="Q4_K" else 210,dqfn)
  ref=w@decode_q8(xp)+bias
  wb=Buffer("CUDA",len(wp),dtypes.uint8,initial_value=bytearray(wp)); xb=Buffer("CUDA",len(xp),dtypes.uint8,initial_value=bytearray(xp))
  bb=Buffer("CUDA",rows,dtypes.float32,initial_value=bytearray(bias.tobytes())); ob=Buffer("CUDA",rows,dtypes.float32,initial_value=bytearray(np.zeros(rows,np.float32).tobytes()))
  Device["CUDA"].synchronize(); mod,fn,stream=cuda.CUmodule(),cuda.CUfunction(),cuda.CUstream()
  try:
    check(cuda.cuModuleLoad(ctypes.byref(mod),str(CUBIN).encode()));check(cuda.cuModuleGetFunction(ctypes.byref(fn),mod,ENTRIES[kind].encode()));check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
    z,one=UInt3(0,0,0),UInt3(1,0,1); rb=k//256; qb=k//32
    keep=[device_pointer(wb),device_pointer(xb),ctypes.c_void_p(),FusionArgs(device_pointer(bb),None,None,0),device_pointer(ob),ctypes.c_uint32(k),z,
      ctypes.c_uint32(rb),ctypes.c_uint32(qb),ctypes.c_uint32(rows),one,ctypes.c_uint32(rows*rb),ctypes.c_uint32(qb),ctypes.c_uint32(rows),one,
      ctypes.c_uint32(rows*rb),ctypes.c_uint32(qb),ctypes.c_uint32(rows),ctypes.c_uint32(0)]
    p=ptrs(keep);check(cuda.cuLaunchKernel(fn,rows,1,1,32,4,1,0,stream,p,None));check(cuda.cuStreamSynchronize(stream))
    raw=bytearray(ob.nbytes);ob.copyout(memoryview(raw));got=np.frombuffer(raw,np.float32).copy()
    err=np.abs(got-ref)
    return {"schema":"tinygrad.llama_cuda_ffn_down_oracle.v1","kind":kind,"shape":{"rows":rows,"k":k},
      "semantic":{"formula":"quantized_matvec(x) + x_bias","glu_op":0},
      "abi":{"entry":ENTRIES[kind],"grid":[rows,1,1],"block":[32,4,1],"ncols_x":k,"stride_row_x":rb,"stride_col_y":qb,"stride_col_dst":rows,"fastdiv":[1,0,1]},
      "correctness":{"max_abs":float(err.max()),"max_rel":float((err/np.maximum(np.abs(ref),1e-3)).max()),"atol":1e-3,"verdict":"PASS" if float(err.max())<1e-3 else "FAIL"},
      "samples":{"got":got[:4].tolist(),"reference":ref[:4].tolist()}}
  finally:
    if stream: cuda.cuStreamDestroy_v2(stream)
    if mod: cuda.cuModuleUnload(mod)

if __name__=="__main__":
  p=argparse.ArgumentParser();p.add_argument("--kind",choices=tuple(ENTRIES),required=True);p.add_argument("--rows",type=int,default=4096);p.add_argument("--k",type=int,default=12288);p.add_argument("--out");a=p.parse_args()
  result=run(a.kind,a.rows,a.k);print(json.dumps(result,sort_keys=True));
  if a.out:pathlib.Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
