"""Default-off graph-owned llama Q4_K x Q8_1 gate/up pair for NV pp512."""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_packed_q4k_q8_llama_candidate import (
  ARTIFACTS, FIXUP_BLOCK, FIXUP_GRID, K, M, MAIN_BLOCK, MAIN_GRID, MMQ_X, N,
  Q8_RECORD_BYTES, SCRATCH_FLOATS, SHARED_BYTES, fastdiv,
)

PAIRS_PER_MODEL = 36
_BINDINGS: dict[str, "LlamaPackedQ4KPP512Binding"] = {}

FP16_DS4_SOURCE = r'''
#include <cuda_fp16.h>
struct __align__(4) block_q8_1_ds4 { __half2 ds[4]; signed char qs[128]; };
extern "C" __global__ void q8_ds4_fp16_pp512(const half *x, block_q8_1_ds4 *y) {
  const int row=blockIdx.x, i0=(blockIdx.y*128 + threadIdx.x)*4;
  const half2 a=*(const half2 *)(x + row*4096 + i0), b=*(const half2 *)(x + row*4096 + i0 + 2);
  const float4 v=make_float4(__half2float(__low2half(a)),__half2float(__high2half(a)),
                             __half2float(__low2half(b)),__half2float(__high2half(b)));
  float amax=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
  float sum=v.x+v.y+v.z+v.w;
  #pragma unroll
  for (int off=4;off>0;off>>=1) {
    amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off));
    sum+=__shfl_xor_sync(0xffffffff,sum,off);
  }
  const float dinv=127.0f/amax;
  char4 q=make_char4(roundf(v.x*dinv),roundf(v.y*dinv),roundf(v.z*dinv),roundf(v.w*dinv));
  const int iqs=i0&127, ib=(i0>>7)*512+row;
  ((char4 *)y[ib].qs)[iqs>>2]=q;
  if ((iqs&31)==0) y[ib].ds[iqs>>5]=__floats2half2_rn(1.0f/dinv,sum);
}
'''

EPILOGUE_SOURCE = r'''
#include <cuda_fp16.h>
extern "C" __global__ void nv_llama_gate_up_silu_mul_fp16(half *out, const float *gate, const float *up) {
  const unsigned int i0=blockIdx.x*256u+threadIdx.x;
  for (unsigned int i=i0; i<6291456u; i+=65536u) {
    const float g=gate[i];
    out[i]=__float2half_rn((g/(1.0f+exp2f(-1.4426950408889634f*g)))*up[i]);
  }
}
'''

MAIN_SYMBOL = "_Z15dense_mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_"
FIXUP_SYMBOL = "_Z30dense_mul_mat_q_stream_k_fixupIL9ggml_type12ELi128ELb0EEvPfS1_5uint3iiiS2_iS2_iS2_"


def supports(*, model_family:str, weight_type:str, m:int, n:int, k:int, device:str) -> bool:
  return model_family == "qwen3_8b" and weight_type == "Q4_K" and (m,n,k) == (M,N,K) and device == "NV"


def _main_vals() -> tuple[int, ...]:
  fd1,fd16,fd4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X)
  sx,sy,sd=N*(K//256),M*(K//32)*9,M*N
  return (*fd16,N,M,K//256,M,N,*fd1,*fd1,sx,sy,sd,*fd1,*fd1,sx,sy,sd,*fd4)


def _fixup_vals() -> tuple[int, ...]:
  fd1,fd16,fd4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X)
  return (*fd16,N,M,N,*fd1,M*N,*fd1,M*N,*fd4)


@dataclass(frozen=True)
class LlamaPackedQ4KPP512Binding:
  producer: object
  main: object
  fixup: object
  epilogue: object | None = None

  @classmethod
  def compile(cls, dev) -> "LlamaPackedQ4KPP512Binding":
    producer_lib=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_ds4_fp16_pp512_v1").compile(FP16_DS4_SOURCE)
    producer=native_nv_program("q8_ds4_fp16_pp512",producer_lib,global_size=(M,8,1),local_size=(128,1,1),
                               globals=(0,1),outs=(1,),ins=(0,))
    main=native_nv_program(MAIN_SYMBOL,(ARTIFACTS/"q4k-mmq-dense.sm_120a.cubin").read_bytes(),global_size=MAIN_GRID,
      local_size=MAIN_BLOCK,globals=(0,1,2,3),outs=(2,3),ins=(0,1),vals=_main_vals(),shared_mem=SHARED_BYTES)
    fixup=native_nv_program(FIXUP_SYMBOL,(ARTIFACTS/"q4k-fixup-dense.sm_120a.cubin").read_bytes(),global_size=FIXUP_GRID,
      local_size=FIXUP_BLOCK,globals=(0,1),outs=(0,),ins=(0,1),vals=_fixup_vals())
    epilogue_lib=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_llama_gate_up_epilogue_pp512_v1").compile(EPILOGUE_SOURCE)
    epilogue=native_nv_program("nv_llama_gate_up_silu_mul_fp16",epilogue_lib,global_size=(256,1,1),local_size=(256,1,1),
      globals=(0,1,2),outs=(0,),ins=(1,2))
    return cls(producer,main,fixup,epilogue)

  def new_capture(self) -> "LlamaPackedQ4KPP512Capture": return LlamaPackedQ4KPP512Capture(self)
  def prepare_pairs(self,count:int) -> None:
    if count != PAIRS_PER_MODEL: raise ValueError(f"exact route requires {PAIRS_PER_MODEL} gate/up pairs")


@dataclass
class LlamaPackedQ4KPP512Capture:
  asset: LlamaPackedQ4KPP512Binding
  trace_epoch: int = 0
  cursor: int = 0

  def begin_trace(self) -> None: self.trace_epoch,self.cursor=self.trace_epoch+1,0

  def produce(self, x:Tensor) -> Tensor:
    """Research-only producer entry exposing the canonical Q8 record."""
    if self.trace_epoch == 0: raise RuntimeError("begin_trace must establish a capture-local epoch before projection")
    record=Tensor.empty(Q8_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device)
    _, record=x.uop_program(record,fxn=lambda *_: self.asset.producer)
    return record

  def project_pair(self,x:Tensor,gate_words:Tensor,up_words:Tensor,*,model_family:str,weight_type:str="Q4_K") -> tuple[Tensor,Tensor]:
    if self.trace_epoch == 0: raise RuntimeError("begin_trace must establish a capture-local epoch before projection")
    if self.cursor >= PAIRS_PER_MODEL: raise RuntimeError("llama packed Q4_K trace exceeded exact 36-pair census")
    if not supports(model_family=model_family,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device):
      raise ValueError("unsupported llama packed Q4_K gate/up route")
    if x.dtype != dtypes.float16 or gate_words.dtype != dtypes.uint32 or up_words.dtype != dtypes.uint32:
      raise ValueError("llama packed Q4_K route requires fp16 activation and canonical uint32 weights")
    self.cursor += 1
    record=Tensor.empty(Q8_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device)
    gate=Tensor.empty(M*N,dtype=dtypes.float32,device=x.device)
    up=Tensor.empty(M*N,dtype=dtypes.float32,device=x.device)
    workspace=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=x.device)
    _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
    gate_words,record,gate,workspace=gate_words.uop_program(record,gate,workspace,fxn=lambda *_:self.asset.main)
    gate,workspace=gate.uop_program(workspace,fxn=lambda *_:self.asset.fixup)
    up_words,record,up,workspace=up_words.uop_program(record,up,workspace,fxn=lambda *_:self.asset.main)
    up,workspace=up.uop_program(workspace,fxn=lambda *_:self.asset.fixup)
    return gate.reshape(M,N),up.reshape(M,N)

  def project(self,x:Tensor,words:Tensor,*,model_family:str,role:str,weight_type:str="Q4_K") -> Tensor:
    """Single-role adapter over the existing llama main/fixup ABI."""
    if self.trace_epoch == 0: raise RuntimeError("begin_trace must establish a capture-local epoch before projection")
    if self.cursor >= 2*PAIRS_PER_MODEL: raise RuntimeError("llama packed Q4_K trace exceeded exact 72-projection census")
    if role not in ("ffn_gate", "ffn_up") or not supports(model_family=model_family,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device):
      raise ValueError("unsupported llama packed Q4_K single-role route")
    if x.dtype != dtypes.float16 or words.dtype != dtypes.uint32: raise ValueError("llama packed Q4_K route requires fp16 activation and canonical uint32 weights")
    self.cursor += 1
    record=Tensor.empty(Q8_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device)
    out=Tensor.empty(M*N,dtype=dtypes.float32,device=x.device)
    workspace=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=x.device)
    _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
    words,record,out,workspace=words.uop_program(record,out,workspace,fxn=lambda *_:self.asset.main)
    out,workspace=out.uop_program(workspace,fxn=lambda *_:self.asset.fixup)
    return out.reshape(M,N)

  def project_from_record(self, record:Tensor, words:Tensor, *, model_family:str, role:str, weight_type:str="Q4_K") -> Tensor:
    """Research-only llama main/fixup consumer for an already-realized DS4 record."""
    if self.trace_epoch == 0: raise RuntimeError("begin_trace must establish a capture-local epoch before projection")
    if role not in ("ffn_gate", "ffn_up"): raise ValueError("unsupported llama Q4 role")
    self.cursor += 1
    out=Tensor.empty(M*N,dtype=dtypes.float32,device=words.device); workspace=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=words.device)
    words,record,out,workspace=words.uop_program(record,out,workspace,fxn=lambda *_:self.asset.main)
    out,workspace=out.uop_program(workspace,fxn=lambda *_:self.asset.fixup)
    return out.reshape(M,N)

  def project_pair_epilogue(self,x:Tensor,gate_words:Tensor,up_words:Tensor,*,model_family:str,weight_type:str="Q4_K") -> Tensor:
    gate,up=self.project_pair(x,gate_words,up_words,model_family=model_family,weight_type=weight_type)
    if self.asset.epilogue is None: raise RuntimeError("gate/up epilogue is unavailable")
    out=Tensor.empty(M*N,dtype=dtypes.float16,device=x.device)
    out,gate,up=out.uop_program(gate,up,fxn=lambda *_:self.asset.epilogue)
    return out.reshape(M,N)


def binding_for(device:str="NV") -> LlamaPackedQ4KPP512Binding:
  if device != "NV": raise ValueError("llama packed Q4_K binding is NV-only")
  if device not in _BINDINGS: _BINDINGS[device]=LlamaPackedQ4KPP512Binding.compile(Device[device])
  return _BINDINGS[device]
