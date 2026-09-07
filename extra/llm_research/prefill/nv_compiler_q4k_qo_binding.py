"""Default-off compiler-owned Q/O Q4_K/Q8_1 IMMA research binding."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib, os
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
  q_program:UOp
  plain_program:UOp
  transform:object
  activation:object
  o_context:object
  q_context:object
  conventional_context:object
  warmstart:dict
  warmstart_contexts:dict
  records:list[Tensor]
  outputs:list[Tensor]
  cursor:int=0

  @classmethod
  def compile(cls,dev,*,native_q_x4:bool=False):
    wt,at,_,base_context=_context();key=warmstart_key({M,N},K,wt.storage_dtype)
    base_context=replace(base_context,operand_order="activation_a_weight_b",native_weight_fragment=None)
    q_base=replace(base_context,operand_order="weight_a_activation_b",native_weight_fragment="q4_a_x4") if native_q_x4 else base_context
    lib=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_compact_record_fp16_v1").compile(_record_source())
    producer=native_nv_program("q8_compact_record_fp16",lib,global_size=(M,8,1),local_size=(128,1,1),
                               globals=(0,1),outs=(1,),ins=(0,))
    def compile_contract(with_residual, salt, candidate_context):
      context=replace(candidate_context, canonical_identity=hashlib.sha256((candidate_context.canonical_identity+salt+repr((candidate_context.operand_order,candidate_context.native_weight_fragment))).encode()).hexdigest())
      warmstart,warmstart_contexts={key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}
      from tinygrad.codegen import to_program_cache
      from tinygrad.codegen.opt.postrange import warmstart_candidate_state
      record_probe=Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device="NV").realize()
      words_probe=Tensor.empty(wt.packed_bytes//4,dtype=dtypes.uint32,device="NV").realize()
      residual_probe=Tensor.empty((M,N),dtype=dtypes.float32,device="NV").realize()
      activation,weight=_activation_carrier(record_probe,at),_weight_carrier(words_probe,wt)
      expr=(weight.matmul(activation.transpose(),dtype=dtypes.int) if context.operand_order=="weight_a_activation_b" else
            activation.matmul(weight.transpose(),dtype=dtypes.int)).cast(dtypes.float)
      if context.operand_order=="weight_a_activation_b": expr=expr.transpose().contiguous()
      if with_residual: expr=(expr+residual_probe).contiguous()
      else: expr=expr.contiguous()
      with warmstart_candidate_state(warmstart,warmstart_contexts): expr.realize()
      matching=[program for program in to_program_cache.values() if program.op is Ops.PROGRAM and program.src and
        getattr(program.src[0].arg,"candidate_context",None) is not None and
        program.src[0].arg.candidate_context.canonical_identity==context.canonical_identity]
      if len(set(matching))!=1:raise RuntimeError(f"expected one compiler Q/O PROGRAM, found {len(set(matching))}")
      compiled=matching[0]
      main=compiled.replace(src=(UOp(Ops.SINK,arg=compiled.src[0].arg),compiled.src[1],UOp(Ops.LINEAR),*compiled.src[3:]))
      expected=(1,2,3) if with_residual else (1,2)
      if main.arg.outs!=(0,) or main.arg.ins!=expected:raise RuntimeError(f"unexpected Q/O PROGRAM ABI {main.arg}")
      return main,context
    plain_program,plain_context=compile_contract(False,"plain",base_context)
    q_program,q_context=(compile_contract(False,"q_x4",q_base) if native_q_x4 else (plain_program,plain_context))
    o_program,o_context=compile_contract(True,"o",base_context)
    return cls(producer=producer,main_program=o_program,q_program=q_program,plain_program=plain_program,transform=wt,activation=at,
      o_context=o_context,q_context=q_context,conventional_context=plain_context,warmstart={},warmstart_contexts={},records=[],outputs=[])

  @property
  def candidate_identity(self):return self.o_context.canonical_identity

  @property
  def q_output_transposed(self): return False

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
    if role == "attn_q":
      if residual is not None: raise ValueError("plain Q projection does not accept residual")
      if self.q_context.operand_order=="weight_a_activation_b": out,words,record=out.uop_program(words,record,fxn=lambda *_:self.q_program)
      else: out,record,words=out.uop_program(record,words,fxn=lambda *_:self.q_program)
    else:
      if residual is None:
        out,record,words=out.uop_program(record,words,fxn=lambda *_:self.q_program)
      else:
        if residual.shape != (M,N) or residual.dtype != dtypes.float32: raise ValueError("O residual must be float32 (512,4096)")
        out,record,words,residual=out.uop_program(record,words,residual,fxn=lambda *_:self.main_program)
    return out.reshape(M,N)


@dataclass
class CompilerQ4StreamKCapture:
  """Research capture with independent storage for each projection."""
  producer: UOp
  q_program: UOp
  fixup_program: UOp
  slots: Tensor
  active: Tensor
  candidate_identity: str
  transform: object
  record_u32: int
  cursor: int = 0
  n: int = 4096
  population: int = 36
  roles: tuple[str,...] = ("attn_q","attn_output")
  pair_q8_reuse: bool = False
  pair_record: object = None

  @classmethod
  def compile(cls, dev, base, *, n=4096, pair_q8_reuse=False):
    if n not in (4096,12288): raise ValueError("unsupported Q4 Stream-K shape")
    population,roles=(72,("attn_q","attn_output")) if n==4096 else (72,("ffn_gate","ffn_up"))
    from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import transform_compiler_q4k_to_streamk, active_fixup_source
    from extra.llm_research.prefill.nv_compiler_streamk_codegen import q4_down_fixup_map
    plain=base.q_program if n==4096 else base.main_program
    sources=[u.arg for u in plain.src if u.op is Ops.SOURCE]
    if len(sources)!=1: raise ValueError("plain Q/O must retain one compiler source")
    unroll=int(os.environ.get("NV_COMPILER_Q4_STREAMK_UNROLL", "8"))
    if unroll not in (1,2,4,6,8,10,12,16,32): raise ValueError("unsupported NV_COMPILER_Q4_STREAMK_UNROLL")
    kernel_name="q4_qo_streamk_n4096" if n==4096 else "q4_qo_streamk"
    double_buffer=bool(int(os.environ.get("NV_COMPILER_Q4_STREAMK_DOUBLE_BUFFER", "0")))
    # Retained default-off: its full bracket did not clear the frozen 0.5 ms
    # population investment threshold.
    fragment_load_to_use=bool(int(os.environ.get("NV_COMPILER_Q4_STREAMK_FRAGMENT_LOAD_TO_USE", "0")))
    shared_load_to_pack=os.environ.get("NV_COMPILER_Q4_STREAMK_SHARED_LOAD_TO_PACK", "0")
    if shared_load_to_pack=="0": shared_load_to_pack=False
    elif shared_load_to_pack=="1": shared_load_to_pack=True
    elif shared_load_to_pack not in ("all","fragments","scales"):
      raise ValueError("NV_COMPILER_Q4_STREAMK_SHARED_LOAD_TO_PACK must be 0, 1, all, fragments, or scales")
    interleave_wmma_updates=bool(int(os.environ.get("NV_COMPILER_Q4_STREAMK_INTERLEAVE_WMMA_UPDATES", "0")))
    source=transform_compiler_q4k_to_streamk(sources[0],unroll=unroll,tiles_n=n//128,k_blocks=64,
      output_stride=n,kernel_name=kernel_name,double_buffer=double_buffer,fragment_load_to_use=fragment_load_to_use,
      shared_load_to_pack=shared_load_to_pack,interleave_wmma_updates=interleave_wmma_updates)
    fixup_source=active_fixup_source(max_contributors=3,sliced=True)
    rows,active=q4_down_fixup_map(k=K,n=n)
    if len(rows)!=4*(n//128) or not active or any(len(row)>3 for row in rows):
      raise ValueError("Q4 Stream-K map must cover every tile")
    compiler=NVRTCCompiler(dev.arch,ptx=False,
      cache_key=f"q4_qo_streamk_u{unroll}_d{int(double_buffer)}_f{int(fragment_load_to_use)}_s{shared_load_to_pack}_i{int(interleave_wmma_updates)}_v1")
    def program(name,source,grid,block,globals,outs,ins,vals=()):
      p=native_nv_program(name,compiler.compile(source),global_size=grid,local_size=block,
        globals=globals,outs=outs,ins=ins,vals=vals)
      return p.replace(src=tuple(u.replace(arg=source) if u.op is Ops.SOURCE else u for u in p.src))
    main=program(kernel_name,source,(170,1,1),(32,2,4),(0,1,2,3,4),(0,1,2),(3,4))
    fix=program("q4k_imma_fixup_active",fixup_source,(len(active),4,1),(128,1,1),(0,1,2,3),(0,),(1,2,3),(M,n))
    slots=Tensor([v for row in rows for v in (*row,*([-1]*(3-len(row))))],dtype=dtypes.int32,device="NV").realize()
    active_tensor=Tensor(active,dtype=dtypes.int32,device="NV").realize()
    identity=hashlib.sha256((source+fixup_source+repr(rows)).encode()).hexdigest()
    return cls(base.producer,main,fix,slots,active_tensor,identity,base.transform,base.activation.storage_units,n=n,population=population,roles=roles,
               pair_q8_reuse=pair_q8_reuse)

  def prepare(self,count):
    if count!=self.population: raise ValueError("Q/O research capture has the wrong projection population")
  def prepare_records(self,count): self.prepare(count)
  def install_warmstart(self,model):
    # This route invokes its already-compiled main/fixup PROGRAMs directly.
    # It does not claim an ordinary compiler schedule key.
    del model
  @property
  def main_program(self): return self.q_program
  def begin_trace(self): self.cursor,self.pair_record=0,None
  def new_capture(self): return replace(self,cursor=0,pair_record=None)
  def project(self,x,words,residual=None,*,model_family,role,weight_type="Q4_K"):
    if (model_family!="qwen3_8b" or role not in self.roles or weight_type!="Q4_K" or x.shape!=(M,K) or x.device!="NV"
        or x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32 or words.numel()!=self.n*(K//256)*36):
      raise ValueError("unsupported Q/O Stream-K input contract")
    if residual is not None: raise ValueError("Stream-K Q/O research arm is projection-only")
    if self.cursor>=self.population: raise ValueError("Q4 Stream-K capture exceeds its admitted projection population")
    expected_role=self.roles[self.cursor%2]
    if self.pair_q8_reuse and role!=expected_role: raise ValueError("Q8 reuse requires ordered gate/up projection pairs")
    if not self.pair_q8_reuse or self.cursor%2==0:
      record=Tensor.empty(self.record_u32,dtype=dtypes.uint32,device=x.device)
      _,record=x.uop_program(record,fxn=lambda *_:self.producer)
      if self.pair_q8_reuse:self.pair_record=record
    else:
      if self.pair_record is None: raise RuntimeError("up projection has no preceding gate Q8 record")
      record=self.pair_record
    self.cursor+=1
    out=Tensor.empty(M*self.n,dtype=dtypes.float32,device=x.device)
    partial=Tensor.empty(340*128*128,dtype=dtypes.float32,device=x.device)
    ids=Tensor.empty(340,dtype=dtypes.int32,device=x.device)
    out,partial,ids,words,record=out.uop_program(partial,ids,words,record,fxn=lambda *_:self.q_program)
    out,partial,slots,active=out.uop_program(partial,self.slots,self.active,fxn=lambda *_:self.fixup_program)
    return out.reshape(M,self.n)


# Compatibility name for the original Q/O-only research capture.
StreamKQOCapture = CompilerQ4StreamKCapture


def binding_for(device="NV", *, variant="wide", native_q_x4:bool=False):
  if device!="NV": raise ValueError("compiler Q/O research binding is NV-only")
  if variant not in ("wide","streamk"): raise ValueError("unknown Q/O candidate variant")
  if native_q_x4 and variant!="wide": raise ValueError("native Q x4 is only admitted on the wide compiler route")
  key=(device,variant,native_q_x4)
  if key not in _BINDINGS:
    _BINDINGS[key]=(CompilerQOBinding.compile(Device[device],native_q_x4=native_q_x4) if variant=="wide" else
                    StreamKQOCapture.compile(Device[device],binding_for(device)))
  return _BINDINGS[key]
