"""Exact llama.cpp Q6_K vocabulary MMVQ binding contract (NV, pp512).

The cubins are unmodified llama CUDA artifacts. This module only admits the
terminal M=1 shape; callers must provide the post-output_norm row and retain
the Q8 producer -> Q6 consumer dependency.
"""
from pathlib import Path
import struct
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

ARTIFACTS = Path(__file__).resolve().parents[3] / "scratchpad/llama_cuda_quantized_oracle_dump"
Q8_SYMBOL = "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout2EEvPKfPKiPvlllllii"
Q6_SYMBOL = "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"

# CUDA parameter-bank offsets recovered from the real llama graph node.
Q6_ARG_OFFSETS = (0, 8, 16, 24, 56, 64, 68, 80, 84, 88, 92, 104, 108, 112, 116, 128, 132, 136, 140)
Q6_ARG_SIZES = (8, 8, 8, 32, 8, 4, 12, 4, 4, 4, 12, 4, 4, 4, 12, 4, 4, 4, 4)

Q8_SOURCE = r'''
#include <cuda_fp16.h>
struct __align__(4) block_q8_1 { __half d, s; signed char qs[32]; };
extern "C" __global__ void q8_vocab_fp32(const float *x, block_q8_1 *out) {
  const int lane=threadIdx.x, block=blockIdx.x;
  const float value=x[block*32+lane];
  float a=fabsf(value);
  for (int off=16;off;off>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,off));
  const float d=__shfl_sync(0xffffffff,a,0)/127.0f;
  int q=d==0.0f ? 0 : (int)roundf(value/d);
  q=max(-127,min(127,q));
  out[block].qs[lane]=(signed char)q;
  int sum=q;
  for (int off=16;off;off>>=1) sum+=__shfl_xor_sync(0xffffffff,sum,off);
  if (lane==0) { out[block].d=__float2half(d); out[block].s=__float2half(sum*d); }
}
'''

def enabled(config) -> bool:
  return bool(getattr(config, "prefill_ubatch", None) == 512 and __import__("os").environ.get("NV_LLAMA_Q6_VOCAB_PP512") == "1")

def artifacts() -> dict:
  return {"q8": ARTIFACTS / "libggml-cuda.q8_1.sm_120a.cubin", "q6": Path(__file__).resolve().parents[3] / "docs/task_workflow/evidence/nv-llama-q6k-vocab-standalone-20260830/q6k-vocab-hcq.sm_120a.cubin"}

def validate() -> None:
  a = artifacts()
  if not all(p.is_file() and p.read_bytes()[:4] == b"\x7fELF" for p in a.values()): raise FileNotFoundError("exact llama vocabulary cubins missing")
  if Q6_ARG_OFFSETS[-1] + Q6_ARG_SIZES[-1] != 144: raise ValueError("invalid captured Q6 parameter-bank layout")

def programs():
  validate()
  q8_cubin=NVRTCCompiler(Device["NV"].arch,ptx=False,cache_key="nv_q8_vocab_fp32_v1").compile(Q8_SOURCE)
  q8 = native_nv_program("q8_vocab_fp32",q8_cubin,global_size=(128,1,1),local_size=(32,1,1),globals=(0,1),outs=(1,),ins=(0,))
  import os
  rows=int(os.environ.get("Q6_TEST_ROWS", "151936"))
  q6 = native_nv_program("q6k_vocab_hcq", artifacts()["q6"].read_bytes(), global_size=(rows,1,1), local_size=(32,4,1),
                         globals=(0,1,2), outs=(2,), ins=(0,1), vals=(rows,))
  return q8, q6

class Binding:
  def __init__(self): self.q8,self.q6=programs()
  def project(self,x:Tensor,weights:Tensor)->Tensor:
    if tuple(x.shape)!=(4096,) or x.dtype!=dtypes.float32 or x.device!="NV": raise ValueError("Q6 vocabulary input must be NV fp32[4096]")
    if weights.device!="NV" or weights.nbytes()!=151936*16*210: raise ValueError("Q6 vocabulary weight contract mismatch")
    packet=Tensor.empty((1152,),dtype=dtypes.uint32,device="NV")
    out=Tensor.empty((151936,),dtype=dtypes.float32,device="NV")
    _,packet=x.uop_program(packet,fxn=lambda *_:self.q8)
    weights,packet,out=weights.uop_program(packet,out,fxn=lambda *_:self.q6)
    return out

_BINDING=None
def binding_for(device="NV"):
  global _BINDING
  if device!="NV": raise ValueError("Q6 vocabulary binding is NV-only")
  if _BINDING is None:_BINDING=Binding()
  return _BINDING
