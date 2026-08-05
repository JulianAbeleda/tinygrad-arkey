#!/usr/bin/env python3
"""Exact llama Q4_K fused-gate/up MMVQ diagnostic oracle.

This is deliberately diagnostic-only.  It launches the extracted llama CUDA
``mul_mat_vec_q<Q4_K,1,true,false>`` entry on tinygrad-owned buffers, with two
independently packed Q4_K matrices.  Its output contract is Qwen's FFN GLU:
``up(x) * silu(gate(x))`` as f32.  It is not a production route.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, statistics, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from scratchpad.llama_cuda_quantized_live_oracle import (FusionArgs, UInt3, fastdiv_values, device_pointer, pack_q8, decode_q8)

QK_K, Q4_BYTES, Q8_BYTES = 256, 144, 36
ENTRY = "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"
ENTRY_PLAIN = "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"
ENTRY_Q8 = "_Z13quantize_q8_1PKfPvlllllj5uint3"
CUBIN = pathlib.Path(__file__).with_name("llama_cuda_quantized_oracle_dump") / "libggml-cuda.so.0.14.36.sm_120a.cubin"
Q8_CUBIN = pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")
BASE = pathlib.Path("/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0")

def digest(p):
  h=hashlib.sha256()
  with open(p,'rb') as f:
    for b in iter(lambda:f.read(1<<20),b''): h.update(b)
  return h.hexdigest()

def q4_ref_lib():
  lib=ctypes.CDLL(str(BASE), mode=ctypes.RTLD_LOCAL)
  q4,q8,dq4=lib.quantize_row_q4_K_ref,lib.quantize_row_q8_1_ref,lib.dequantize_row_q4_K
  for fn in (q4,q8): fn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64]; fn.restype=None
  dq4.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_int64]; dq4.restype=None
  return q4,q8,dq4

def pack_q4(x, fn):
  x=np.ascontiguousarray(x,dtype=np.float32).reshape(-1)
  out=(ctypes.c_uint8*(x.size//QK_K*Q4_BYTES))(); fn(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),out,x.size)
  return bytes(out)

def dequant_q4(payload, rows, k, fn):
  out=np.empty((rows,k),np.float32)
  # This is a pinned llama CPU implementation used only as an independent GPU
  # output check; packed inputs and launch ABI are separately pinned below.
  for r in range(rows):
    block=payload[r*(k//QK_K)*Q4_BYTES:(r+1)*(k//QK_K)*Q4_BYTES]
    c=(ctypes.c_uint8*len(block)).from_buffer_copy(block)
    fn(c,out[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),k)
  return out

def params(up, q8, gate, out, nrows, k, fused=True):
  # Captured live llama graph proves fastdiv(1) is (1,0,1), not the
  # mathematically tempting (0,0,1). The multiplier field is consumed by the
  # device fastdiv helper even on the unit divisor path.
  z,one=UInt3(0,0,0),UInt3(1,0,1)
  fusion = FusionArgs(None,device_pointer(gate),None,2) if fused else FusionArgs(None,None,None,0)
  args=[device_pointer(up),device_pointer(q8),ctypes.c_void_p(),fusion,device_pointer(out),ctypes.c_uint32(k),z,
        ctypes.c_uint32(k//QK_K),ctypes.c_uint32(k//32),ctypes.c_uint32(nrows),one,ctypes.c_uint32(nrows*(k//QK_K)),
        ctypes.c_uint32(k//32),ctypes.c_uint32(nrows),one,ctypes.c_uint32(nrows*(k//QK_K)),ctypes.c_uint32(k//32),ctypes.c_uint32(nrows),ctypes.c_uint32(0)]
  return args,(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in args])

def run(rows=12288,k=4096,iters=100,reps=5, plain=False, same_gate=False):
  if not CUBIN.is_file() or not Q8_CUBIN.is_file() or not BASE.is_file(): raise FileNotFoundError("pinned cubin/base artifact missing")
  if ctypes.sizeof(FusionArgs)!=32 or ctypes.sizeof(UInt3)!=12: raise RuntimeError("fusion ABI size mismatch")
  q4,q8,dq4=q4_ref_lib(); rng=np.random.default_rng(20260804)
  up_f=rng.normal(0,.2,(rows,k)).astype(np.float32); gate_f=rng.normal(0,.2,(rows,k)).astype(np.float32); x=rng.normal(0,.2,k).astype(np.float32)
  up_p,gate_p,x_p=pack_q4(up_f,q4),pack_q4(gate_f,q4),pack_q8(x,q8)
  # q8 decode is independent Python; q4 decode is llama CPU reference and is
  # explicitly recorded as such rather than misrepresented as independent.
  ref_up,ref_gate=dequant_q4(up_p,rows,k,dq4)@decode_q8(x_p),dequant_q4(gate_p,rows,k,dq4)@decode_q8(x_p)
  # GGML_GLU_OP_SWIGLU is `up * silu(gate)`, not merely `silu(gate)`.
  # Keeping the leading up factor explicit is the key independent semantic
  # check for the fused entry.
  ref=ref_up if plain else ref_up*((ref_up if same_gate else ref_gate)/(1+np.exp(-(ref_up if same_gate else ref_gate))))
  u=Buffer("CUDA",len(up_p),dtypes.uint8,initial_value=bytearray(up_p)); g=Buffer("CUDA",len(gate_p),dtypes.uint8,initial_value=bytearray(gate_p))
  q=Buffer("CUDA",len(x_p),dtypes.uint8,initial_value=bytearray(x_p)); out=Buffer("CUDA",rows,dtypes.float32,initial_value=bytearray(np.zeros(rows,np.float32).tobytes()))
  Device["CUDA"].synchronize(); mod,qmod,fn,qfn,stream=cuda.CUmodule(),cuda.CUmodule(),cuda.CUfunction(),cuda.CUfunction(),cuda.CUstream(); graph,exe=cuda.CUgraph(),cuda.CUgraphExec(); start,end=cuda.CUevent(),cuda.CUevent()
  try:
    entry = ENTRY_PLAIN if plain else ENTRY
    check(cuda.cuModuleLoad(ctypes.byref(mod),str(CUBIN).encode())); check(cuda.cuModuleGetFunction(ctypes.byref(fn),mod,entry.encode())); check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
    keep,pp=params(u,q,u if same_gate else g,out,rows,k,not plain)
    def launch():
      # 768 B is compiler-reported static shared memory for this specialization;
      # host dispatcher passes zero dynamic shared bytes.
      check(cuda.cuLaunchKernel(fn,rows,1,1,32,4,1,0,stream,pp,None))
    launch(); check(cuda.cuStreamSynchronize(stream)); got=np.frombuffer(bytearray(out.nbytes),dtype=np.float32); out.copyout(memoryview(got))
    check(cuda.cuStreamBeginCapture_v2(stream,cuda.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)); launch(); check(cuda.cuStreamEndCapture(stream,ctypes.byref(graph))); nn=ctypes.c_size_t(); check(cuda.cuGraphGetNodes(graph,None,ctypes.byref(nn))); check(cuda.cuGraphInstantiate_v2(ctypes.byref(exe),graph,None,None,0))
    check(cuda.cuEventCreate(ctypes.byref(start),0));check(cuda.cuEventCreate(ctypes.byref(end),0)); times=[]
    for _ in range(reps):
      check(cuda.cuEventRecord(start,stream)); [check(cuda.cuGraphLaunch(exe,stream)) for _ in range(iters)];check(cuda.cuEventRecord(end,stream));check(cuda.cuEventSynchronize(end)); ms=ctypes.c_float();check(cuda.cuEventElapsedTime(ctypes.byref(ms),start,end));times.append(ms.value*1000/iters)
    err=float(np.max(np.abs(got-ref))); rel=float(np.max(np.abs(got-ref)/np.maximum(np.abs(ref),1e-3)))
    # Relative error is not a decision criterion here: the GLU result crosses
    # zero, where a tiny correct absolute difference has an arbitrary ratio.
    return {"schema":"tinygrad.llama_q4k_fused_gate_up_oracle.v1","evidence":"CONSTRUCTION_RECORD","shape":{"rows":rows,"k":k},"semantic":{"formula":"up(x)" if plain else "up(x) * silu(gate(x))","glu_op":None if plain else "GGML_GLU_OP_SWIGLU=2","output":"f32[rows]"},"abi":{"entry":entry,"argument_count":len(keep),"fusion_args_bytes":32,"gate_field":None if plain else "Q4_K weight pointer","x_bias":None,"gate_bias":None,"grid":[rows,1,1],"block":[32,4,1],"static_shared_bytes":768,"dynamic_shared_bytes":0},"artifacts":{"cubin_sha256":digest(CUBIN),"base_sha256":digest(BASE)},"correctness":{"max_abs":err,"max_rel":rel,"absolute_tolerance":1e-3,"verdict":"PASS" if err < 1e-3 else "FAIL: construction ABI not yet exact; no timing claim"},"graph":{"nodes":nn.value},"observed_output_samples":{"got":got[:4].tolist(),"reference":ref[:4].tolist()},"next_experiment":"Only attempt a real-token construction after this exact synthetic arm passes.","reference":"q4 dequant: pinned llama CPU; q8 decode: independent Python"}
  finally:
    if exe: cuda.cuGraphExecDestroy(exe)
    if graph: cuda.cuGraphDestroy(graph)
    if stream: cuda.cuStreamDestroy_v2(stream)
    if qmod: cuda.cuModuleUnload(qmod)
    if mod: cuda.cuModuleUnload(mod)

if __name__ == '__main__':
  ap=argparse.ArgumentParser();ap.add_argument('--rows',type=int,default=12288);ap.add_argument('--k',type=int,default=4096);ap.add_argument('--iters',type=int,default=100);ap.add_argument('--reps',type=int,default=5);ap.add_argument('--plain',action='store_true');ap.add_argument('--same-gate',action='store_true');ap.add_argument('--out');a=ap.parse_args()
  r=run(a.rows,a.k,a.iters,a.reps,a.plain,a.same_gate);print(json.dumps(r,sort_keys=True));
  if a.out: pathlib.Path(a.out).write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
