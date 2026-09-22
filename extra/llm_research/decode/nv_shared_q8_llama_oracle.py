#!/usr/bin/env python3
"""Byte-exact native-NV provider check against llama's extracted CUDA Q8_1 kernel."""
from __future__ import annotations
import argparse, ctypes, json, pathlib
import numpy as np

from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Buffer
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.llm.shared_q8_attention import _emit_q4, _emit_q6, _emit_q8_provider
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from scratchpad.llama_cuda_quantized_live_oracle import (DEFAULT_BASE, DEFAULT_CUBIN as DEFAULT_MMVQ_CUBIN, ENTRY_Q4, ENTRY_Q6, ENTRY_Q8,
  FusionArgs, _copy_f32, _cpu_quantizers, _kernel_params, device_pointer, fastdiv_values, pack_q4, pack_q6)

DEFAULT_Q8_CUBIN=pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")

def run(cubin:pathlib.Path, mmvq_cubin:pathlib.Path=DEFAULT_MMVQ_CUBIN, seed:int=123) -> dict:
  rng=np.random.default_rng(seed)
  # The production tinygrad consumer rounds its activation to fp16 before
  # quantization. Feed those same values, widened to fp32, to llama CUDA.
  x16=rng.normal(0,.3,4096).astype(np.float16); x32=x16.astype(np.float32)
  inp=Buffer("CUDA",4096,dtypes.float32,initial_value=bytearray(x32.tobytes()))
  out=Buffer("CUDA",128*36,dtypes.uint8,preallocate=True)
  module,function,mmvq_module,stream=cuda.CUmodule(),cuda.CUfunction(),cuda.CUmodule(),cuda.CUstream()
  llama_outputs, packed_weights = {}, {}
  try:
    check(cuda.cuModuleLoad(ctypes.byref(module),str(cubin).encode()))
    check(cuda.cuModuleGetFunction(ctypes.byref(function),module,ENTRY_Q8.encode()))
    check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
    args=[device_pointer(inp),device_pointer(out),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),
          ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_uint32(1),fastdiv_values(1)]
    params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(a),ctypes.c_void_p) for a in args])
    check(cuda.cuLaunchKernel(function,16,1,1,256,1,1,0,stream,params,None));check(cuda.cuStreamSynchronize(stream))
    llama=bytearray(out.nbytes);out.copyout(memoryview(llama))
    # Run the exact extracted llama Q4_K/Q6_K MMVQ consumers against that live
    # Q8 buffer. The native consumers below receive identical packed bytes.
    check(cuda.cuModuleLoad(ctypes.byref(mmvq_module),str(mmvq_cubin).encode()))
    q4fn,q6fn=cuda.CUfunction(),cuda.CUfunction();check(cuda.cuModuleGetFunction(ctypes.byref(q4fn),mmvq_module,ENTRY_Q4.encode()));check(cuda.cuModuleGetFunction(ctypes.byref(q6fn),mmvq_module,ENTRY_Q6.encode()))
    q4quant,q6quant,_q8quant=_cpu_quantizers(DEFAULT_BASE)
    for label,fn,packer,quantizer in (("q4",q4fn,pack_q4,q4quant),("q6",q6fn,pack_q6,q6quant)):
      weights=rng.normal(0,.2,(64,4096)).astype(np.float32);payload=packer(weights,quantizer);packed_weights[label]=payload
      wb=Buffer("CUDA",len(payload),dtypes.uint8,initial_value=bytearray(payload));ob=Buffer("CUDA",64,dtypes.float32,initial_value=bytearray(np.zeros(64,dtype=np.float32).tobytes()))
      keep,kparams=_kernel_params(wb,out,ob,64,4096,0)
      check(cuda.cuLaunchKernel(fn,64,1,1,32,4,1,0,stream,kparams,None));check(cuda.cuStreamSynchronize(stream));llama_outputs[label]=_copy_f32(ob)
  finally:
    if stream:cuda.cuStreamDestroy_v2(stream)
    if mmvq_module:cuda.cuModuleUnload(mmvq_module)
    if module:cuda.cuModuleUnload(module)
  program=KernelProgram("decode_shared_q8_attention","llama_oracle",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    _emit_q8_provider(),output_spec=OutputSpec((1152,),dtypes.uint32))
  native=execute_promoted_program(None,Tensor(x16,device="NV").contiguous(),program=program).numpy().astype(np.uint32,copy=False)
  native_blocks=bytearray()
  qbytes=native[:1024].tobytes();meta=native[1024:].tobytes()
  for group in range(128):native_blocks.extend(meta[group*4:(group+1)*4]+qbytes[group*32:(group+1)*32])
  mismatch=np.frombuffer(native_blocks,dtype=np.uint8)!=np.frombuffer(llama,dtype=np.uint8)
  fields={"d":0,"s":0,"qs":0}
  for off in np.flatnonzero(mismatch):fields["d" if off%36<2 else "s" if off%36<4 else "qs"]+=1
  consumers={}
  for label,dtype,emitter in (("q4",dtypes.uint32,_emit_q4(64)),("q6",dtypes.uint16,_emit_q6(64))):
    words=np.frombuffer(packed_weights[label],dtype=np.uint32 if label=="q4" else np.uint16).copy()
    cp=KernelProgram("decode_shared_q8_attention",f"llama_oracle_{label}",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      emitter,output_spec=OutputSpec((64,),dtypes.float32))
    native_out=execute_promoted_program(None,Tensor(words,dtype=dtype,device="NV").contiguous(),Tensor(native,device="NV").contiguous(),program=cp).numpy()
    delta=np.abs(native_out-llama_outputs[label]);consumers[label]={"max_abs":float(delta.max()),"mean_abs":float(delta.mean()),"atol_2e_2_pass":bool(np.all(delta<=.02))}
  provider_pass=not bool(mismatch.any())
  return {"schema":"tinygrad.nv_shared_q8_llama_oracle.v1","seed":seed,"bytes":len(llama),
          "mismatch_bytes":int(mismatch.sum()),"mismatch_fields":fields,"provider_pass":provider_pass,
          "consumers":consumers,"pass":provider_pass and all(row["atol_2e_2_pass"] for row in consumers.values())}

if __name__=="__main__":
  p=argparse.ArgumentParser();p.add_argument("--cubin",type=pathlib.Path,default=DEFAULT_Q8_CUBIN);p.add_argument("--mmvq-cubin",type=pathlib.Path,default=DEFAULT_MMVQ_CUBIN);p.add_argument("--seed",type=int,default=123);p.add_argument("--out",type=pathlib.Path);a=p.parse_args()
  result=run(a.cubin,a.mmvq_cubin,a.seed);encoded=json.dumps(result,indent=2,sort_keys=True);print(encoded,flush=True)
  if a.out:a.out.write_text(encoded+"\n")
