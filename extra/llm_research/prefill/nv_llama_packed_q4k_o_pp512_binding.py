"""Default-off graph-owned Q4_K x Q8_1 attention-output projection for NV pp512."""
from __future__ import annotations
from dataclasses import dataclass
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_packed_q4k_q8_llama_candidate import (
  ARTIFACTS, MAIN_GRID, MAIN_BLOCK, FIXUP_GRID, FIXUP_BLOCK, MMQ_X,
  SHARED_BYTES, fastdiv,
)

M, N, K = 512, 4096, 4096
# The exact llama host trace retains this persistent geometry for O even at
# N=4096; scaling it down underfills the Blackwell service.
MAIN_GRID, FIXUP_GRID = (170, 1, 1), (170, 4, 1)
PROJECTIONS_PER_MODEL = 36
Q8_RECORD_BYTES = M * (K // 128) * 144 + MMQ_X * 144
SCRATCH_FLOATS = MAIN_GRID[0] * MMQ_X * MMQ_X
_BINDINGS = {}

def _single_owner_main(words:Tensor, record:Tensor, out:Tensor, workspace:Tensor, main) -> tuple[Tensor,Tensor]:
  # Keep one lazy AFTER owner. ProgramInfo.outs still declares both physical
  # writes, so dependency/resource tracking owns the raw workspace correctly.
  _,_,out,_=words.uop_program(record,out,workspace,fxn=lambda *_:main)
  return out,workspace

MAIN_SYMBOL = "_Z15dense_mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_"
FIXUP_SYMBOL = "_Z30dense_mul_mat_q_stream_k_fixupIL9ggml_type12ELi128ELb0EEvPfS1_5uint3iiiS2_iS2_iS2_"

FP16_DS4_SOURCE = r'''
#include <cuda_fp16.h>
struct __align__(4) block_q8_1_ds4 { __half2 ds[4]; signed char qs[128]; };
extern "C" __global__ void q8_ds4_fp16_o_pp512(const half*x,block_q8_1_ds4*y){
 const int row=blockIdx.x,i0=(blockIdx.y*128+threadIdx.x)*4;
 const half2 a=*(const half2*)(x+row*4096+i0),b=*(const half2*)(x+row*4096+i0+2);
 const float4 v=make_float4(__half2float(__low2half(a)),__half2float(__high2half(a)),__half2float(__low2half(b)),__half2float(__high2half(b)));
 float am=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w))),sum=v.x+v.y+v.z+v.w;
 #pragma unroll
 for(int off=4;off>0;off>>=1){am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,off));sum+=__shfl_xor_sync(0xffffffff,sum,off);}
 const float di=127.0f/am; char4 q=make_char4(roundf(v.x*di),roundf(v.y*di),roundf(v.z*di),roundf(v.w*di));
 const int iq=i0&127,ib=(i0>>7)*512+row; ((char4*)y[ib].qs)[iq>>2]=q;
 if((iq&31)==0)y[ib].ds[iq>>5]=__floats2half2_rn(1.0f/di,sum);
}
'''

def supports(*, model_family, role, weight_type, m, n, k, device):
  return model_family == "qwen3_8b" and role == "attn_output" and weight_type == "Q4_K" and (m,n,k) == (M,N,K) and device == "NV"

def _main_vals():
  f1,f16,f4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X); sx,sy,sd=N*(K//256),M*(K//32)*9,M*N
  return (*f16,N,M,K//256,M,N,*f1,*f1,sx,sy,sd,*f1,*f1,sx,sy,sd,*f4)

def _fixup_vals():
  f1,f16,f4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X)
  return (*f16,N,M,N,*f1,M*N,*f1,M*N,*f4)

@dataclass(frozen=True)
class Binding:
  producer: object; main: object; fixup: object
  @classmethod
  def compile(cls, dev):
    lib=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_ds4_fp16_o_pp512_v1").compile(FP16_DS4_SOURCE)
    return cls(native_nv_program("q8_ds4_fp16_o_pp512",lib,global_size=(M,K//512,1),local_size=(128,1,1),globals=(0,1),outs=(1,),ins=(0,)),
      native_nv_program(MAIN_SYMBOL,(ARTIFACTS/"q4k-mmq-dense.sm_120a.cubin").read_bytes(),global_size=MAIN_GRID,local_size=MAIN_BLOCK,globals=(0,1,2,3),outs=(2,3),ins=(0,1),vals=_main_vals(),shared_mem=SHARED_BYTES),
      native_nv_program(FIXUP_SYMBOL,(ARTIFACTS/"q4k-fixup-dense.sm_120a.cubin").read_bytes(),global_size=FIXUP_GRID,local_size=FIXUP_BLOCK,globals=(0,1),outs=(0,),ins=(0,1),vals=_fixup_vals()))
  def new_capture(self): return Capture(self)
  def prepare_records(self,count):
    if count != PROJECTIONS_PER_MODEL: raise ValueError("exact route requires 36 attention-output projections")

@dataclass
class Capture:
  asset: Binding; trace_epoch: int = 0; cursor: int = 0
  def begin_trace(self): self.trace_epoch,self.cursor=self.trace_epoch+1,0
  def project(self,x,words,residual=None,*,model_family,role,weight_type="Q4_K"):
    if self.trace_epoch == 0: raise RuntimeError("begin_trace required")
    if self.cursor >= PROJECTIONS_PER_MODEL: raise RuntimeError("Q4 attention-output census exceeded")
    if not supports(model_family=model_family,role=role,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device): raise ValueError("unsupported Q4 attention-output route")
    if x.dtype != dtypes.float16 or words.dtype != dtypes.uint32: raise ValueError("Q4 attention-output requires fp16 and uint32")
    self.cursor += 1
    r=Tensor.empty(Q8_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device); o=Tensor.empty(M*N,dtype=dtypes.float32,device=x.device); s=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=x.device)
    _,r=x.uop_program(r,fxn=lambda *_:self.asset.producer)
    if getenv("NV_LLAMA_O_SINGLE_OWNER_PP512", getenv("NV_LLAMA_FULL_PACKED_PP512", 1)):
      # The PROGRAM declares o/s as writes. Retain one AFTER owner for the main
      # and pass the raw workspace allocation to fixup; retaining both AFTERs
      # can materialize the same opaque main once per result owner.
      o,s=_single_owner_main(words,r,o,s,self.asset.main)
    else: words,r,o,s=words.uop_program(r,o,s,fxn=lambda *_:self.asset.main)
    o,s=o.uop_program(s,fxn=lambda *_:self.asset.fixup)
    out=o.reshape(M,N)
    return out if residual is None else out+residual

def binding_for(device="NV"):
  if device != "NV": raise ValueError("Q4 attention-output binding is NV-only")
  if device not in _BINDINGS: _BINDINGS[device]=Binding.compile(Device[device])
  return _BINDINGS[device]
