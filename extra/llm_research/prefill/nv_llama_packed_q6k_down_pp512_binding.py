"""Default-off graph-owned llama Q6_K x Q8_1 FFN-down primitive for NV pp512."""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad import Device,Tensor,dtypes
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_packed_q4k_q8_llama_candidate import ARTIFACTS,MAIN_GRID,MAIN_BLOCK,FIXUP_GRID,FIXUP_BLOCK,MMQ_X,SHARED_BYTES,SCRATCH_FLOATS,fastdiv

M,N,K=512,4096,12288
PROJECTIONS_PER_MODEL=18
Q8_RECORD_BYTES=M*(K//128)*144+MMQ_X*144
_BINDINGS={}
MAIN_SYMBOL="_Z15dense_mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_"
FIXUP_SYMBOL="_Z30dense_mul_mat_q_stream_k_fixupIL9ggml_type14ELi128ELb0EEvPfS1_5uint3iiiS2_iS2_iS2_"

FP16_D4_SOURCE=r'''
#include <cuda_fp16.h>
struct __align__(4) block_q8_1_d4 { float d[4]; signed char qs[128]; };
extern "C" __global__ void q8_d4_fp16_down_pp512(const half *x, block_q8_1_d4 *y) {
  const int row=blockIdx.x,i0=(blockIdx.y*128+threadIdx.x)*4;
  const half2 a=*(const half2 *)(x+row*12288+i0),b=*(const half2 *)(x+row*12288+i0+2);
  const float4 v=make_float4(__half2float(__low2half(a)),__half2float(__high2half(a)),__half2float(__low2half(b)),__half2float(__high2half(b)));
  float amax=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
  #pragma unroll
  for(int off=4;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off));
  const float dinv=127.0f/amax;char4 q=make_char4(roundf(v.x*dinv),roundf(v.y*dinv),roundf(v.z*dinv),roundf(v.w*dinv));
  const int iqs=i0&127,ib=(i0>>7)*512+row;((char4 *)y[ib].qs)[iqs>>2]=q;
  if((iqs&31)==0)y[ib].d[iqs>>5]=1.0f/dinv;
}
'''

def supports(*,model_family:str,role:str,weight_type:str,m:int,n:int,k:int,device:str)->bool:
  return model_family=="qwen3_8b" and role=="ffn_down" and weight_type=="Q6_K" and (m,n,k)==(M,N,K) and device=="NV"

def _main_vals():
  fd1,fd48,fd4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X);sx,sy,sd=N*(K//256),M*(K//32)*9,M*N
  return (*fd48,N,M,K//256,M,N,*fd1,*fd1,sx,sy,sd,*fd1,*fd1,sx,sy,sd,*fd4)

def _fixup_vals():
  fd1,fd48,fd4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X)
  return (*fd48,N,M,N,*fd1,M*N,*fd1,M*N,*fd4)

@dataclass(frozen=True)
class LlamaPackedQ6KDownBinding:
  producer:object;main:object;fixup:object
  @classmethod
  def compile(cls,dev):
    lib=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_d4_fp16_down_pp512_v1").compile(FP16_D4_SOURCE)
    producer=native_nv_program("q8_d4_fp16_down_pp512",lib,global_size=(M,K//512,1),local_size=(128,1,1),globals=(0,1),outs=(1,),ins=(0,))
    main=native_nv_program(MAIN_SYMBOL,(ARTIFACTS/"q6k-mmq-dense.sm_120a.cubin").read_bytes(),global_size=MAIN_GRID,local_size=MAIN_BLOCK,
      globals=(0,1,2,3),outs=(2,3),ins=(0,1),vals=_main_vals(),shared_mem=SHARED_BYTES)
    fixup=native_nv_program(FIXUP_SYMBOL,(ARTIFACTS/"q6k-fixup-dense.sm_120a.cubin").read_bytes(),global_size=FIXUP_GRID,local_size=FIXUP_BLOCK,
      globals=(0,1),outs=(0,),ins=(0,1),vals=_fixup_vals())
    return cls(producer,main,fixup)
  def new_capture(self):return LlamaPackedQ6KDownCapture(self)
  def prepare_records(self,count:int):
    if count!=PROJECTIONS_PER_MODEL:raise ValueError(f"exact route requires {PROJECTIONS_PER_MODEL} Q6 down projections")

@dataclass
class LlamaPackedQ6KDownCapture:
  asset:LlamaPackedQ6KDownBinding;trace_epoch:int=0;cursor:int=0
  def begin_trace(self):self.trace_epoch,self.cursor=self.trace_epoch+1,0
  def project(self,x:Tensor,halfs:Tensor,*,model_family:str,role:str,weight_type:str="Q6_K"):
    if self.trace_epoch==0:raise RuntimeError("begin_trace must establish a capture-local epoch before projection")
    if self.cursor>=PROJECTIONS_PER_MODEL:raise RuntimeError("llama packed Q6_K trace exceeded exact 18-projection census")
    if not supports(model_family=model_family,role=role,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device):raise ValueError("unsupported llama Q6_K down route")
    if x.dtype!=dtypes.float16 or halfs.dtype!=dtypes.uint16:raise ValueError("llama Q6_K down requires fp16 activation and canonical uint16 weights")
    self.cursor+=1;record=Tensor.empty(Q8_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device);out=Tensor.empty(M*N,dtype=dtypes.float32,device=x.device);workspace=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=x.device)
    _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
    halfs,record,out,workspace=halfs.uop_program(record,out,workspace,fxn=lambda *_:self.asset.main)
    out,workspace=out.uop_program(workspace,fxn=lambda *_:self.asset.fixup)
    return out.reshape(M,N)

def binding_for(device="NV"):
  if device!="NV":raise ValueError("llama packed Q6_K down binding is NV-only")
  if device not in _BINDINGS:_BINDINGS[device]=LlamaPackedQ6KDownBinding.compile(Device[device])
  return _BINDINGS[device]
