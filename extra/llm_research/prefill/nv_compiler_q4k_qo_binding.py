"""Default-off compiler-owned Q/O Q4_K/Q8_1 IMMA research binding."""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import warmstart_key
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import _record_source
from extra.llm_research.prefill.nv_compiler_q4k_production_gate import _activation_carrier, _weight_carrier
from extra.llm_research.prefill.nv_compiler_q4k_qo_gate import M,N,K,_context
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program

RECORD_BYTES=M*K+2*M*(K//32)*4
RECORD_U32=RECORD_BYTES//4
LEGAL_ROLES=frozenset(("attn_q","attn_output"))
_BINDINGS={}


def supports(*,model_family:str,role:str,weight_type:str,m:int,n:int,k:int,device:str)->bool:
  return model_family=="qwen3_8b" and role in LEGAL_ROLES and weight_type=="Q4_K" and \
    (m,n,k)==(M,N,K) and device=="NV"


@dataclass
class CompilerQOBinding:
  producer:object
  main_program:UOp
  transform:object
  activation:object
  context:object
  warmstart:dict
  warmstart_contexts:dict
  records:list[Tensor]
  outputs:list[Tensor]
  cursor:int=0

  @classmethod
  def compile(cls,dev):
    wt,at,_,context=_context();key=warmstart_key({M,N},K,wt.storage_dtype)
    lib=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_compact_record_fp16_v1").compile(_record_source())
    producer=native_nv_program("q8_compact_record_fp16",lib,global_size=(M,8,1),local_size=(128,1,1),
                               globals=(0,1),outs=(1,),ins=(0,))
    warmstart,warmstart_contexts={key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}
    from tinygrad.codegen import to_program_cache
    from tinygrad.codegen.opt.postrange import warmstart_candidate_state
    record_probe=Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device="NV").realize()
    words_probe=Tensor.empty(wt.packed_bytes//4,dtype=dtypes.uint32,device="NV").realize()
    residual_probe=Tensor.empty((M,N),dtype=dtypes.float32,device="NV").realize()
    with warmstart_candidate_state(warmstart,warmstart_contexts):
      (_activation_carrier(record_probe,at).matmul(_weight_carrier(words_probe,wt).transpose(),dtype=dtypes.int) \
        .cast(dtypes.float) + residual_probe).contiguous().realize()
    matching=[program for program in to_program_cache.values() if program.op is Ops.PROGRAM and program.src and
      getattr(program.src[0].arg,"candidate_context",None) is not None and
      program.src[0].arg.candidate_context.canonical_identity==context.canonical_identity]
    if len(set(matching))!=1:raise RuntimeError(f"expected one compiler Q/O PROGRAM, found {len(set(matching))}")
    compiled=matching[0]
    main=compiled.replace(src=(UOp(Ops.SINK,arg=compiled.src[0].arg),compiled.src[1],UOp(Ops.LINEAR),*compiled.src[3:]))
    if main.arg.outs!=(0,) or main.arg.ins!=(1,2,3):raise RuntimeError(f"unexpected Q/O residual PROGRAM ABI {main.arg}")
    return cls(producer,main,wt,at,context,warmstart,warmstart_contexts,[],[])

  @property
  def candidate_identity(self):return self.context.canonical_identity

  def install_warmstart(self,model):
    opts,contexts=dict(model._packed_wmma_warmstart or {}),dict(model._packed_wmma_warmstart_contexts or {})
    for key,value in self.warmstart.items():
      if key in opts and opts[key]!=value:raise RuntimeError("Q/O warmstart collision")
      opts[key]=value
    for key,value in self.warmstart_contexts.items():
      if key in contexts and contexts[key]!=value:raise RuntimeError("Q/O context collision")
      contexts[key]=value
    model._packed_wmma_warmstart,model._packed_wmma_warmstart_contexts=opts,contexts

  def prepare(self,count:int):
    while len(self.records)<count:
      self.records.append(Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device="NV").realize())
      self.outputs.append(Tensor.empty(M*N,dtype=dtypes.float32,device="NV").realize())

  def begin_trace(self):self.cursor=0

  def new_capture(self):
    # The compiler O research route owns graph-local output tensors; retain the
    # existing binding object as its capture state, matching the single-model
    # pp512 harness lifecycle.
    self.cursor=0
    return self

  def project(self,x:Tensor,words:Tensor,residual:Tensor|None=None,*,model_family:str,role:str,weight_type:str="Q4_K"):
    if not supports(model_family=model_family,role=role,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device):
      raise ValueError("unsupported compiler Q/O research route")
    if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32:raise ValueError("Q/O route requires fp16 x and canonical uint32 Q4_K")
    if self.cursor>=len(self.records):self.prepare(self.cursor+1)
    record,out=self.records[self.cursor],self.outputs[self.cursor];self.cursor+=1
    _,record=x.uop_program(record,fxn=lambda *_:self.producer)
    if residual is None: raise ValueError("compiler Q/O residual route requires residual")
    out,record,words,residual=out.uop_program(record,words,residual,fxn=lambda *_:self.main_program)
    return out.reshape(M,N)


def binding_for(device="NV"):
  if device!="NV":raise ValueError("compiler Q/O research binding is NV-only")
  if device not in _BINDINGS:_BINDINGS[device]=CompilerQOBinding.compile(Device[device])
  return _BINDINGS[device]
