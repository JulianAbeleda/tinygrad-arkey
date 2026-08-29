"""Default-off compiler-owned Qwen3-8B pp512 Q4_K K-projection binding.

This is intentionally separate from the gate/up binding.  It admits exactly
36 ``attn_k`` projections at (512,1024,4096), uses the occupancy-qualified
64x32x64/256-CTA geometry, and has no Q4 V or Q6 V admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_key
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import _record_source

M, N, K, TILE_K = 512, 1024, 4096, 64
PROJECTIONS_PER_MODEL = 36
LEGAL_ROLES = frozenset(("attn_k",))
RECORD_BYTES = M*K + 2*M*(K//32)*4
RECORD_U32 = RECORD_BYTES//4
_BINDINGS: dict[str, "CompilerKPP512Binding"] = {}


def supports(*, model_family:str, role:str, weight_type:str, m:int, n:int, k:int, device:str) -> bool:
  return (model_family == "qwen3_8b" and role in LEGAL_ROLES and weight_type == "Q4_K" and
          (m,n,k) == (M,N,K) and device == "NV")


@dataclass(frozen=True)
class _Context:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  packed_weight: PackedWeightTransform
  packed_fragment_provider: Q4KInt8FragmentProvider
  packed_activation: Q8ActivationRecordTransform
  packed_activation_provider: Q8Int8FragmentProvider
  group_accumulator: Q4KQ8GroupAccumulatorContract
  pipeline: None = None


def _weight_carrier(words:Tensor, transform:PackedWeightTransform) -> Tensor:
  blocks, halfwords = transform.rows*transform.blocks_per_row, int(transform.block_bytes)//2
  return words.bitcast(dtypes.uint16).reshape(blocks,halfwords).pad(((0,0),(0,128-halfwords))) \
    .reshape(blocks,128,1).expand(blocks,128,2).reshape(transform.rows,transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _activation_carrier(record:Tensor, transform:Q8ActivationRecordTransform) -> Tensor:
  return record.bitcast(dtypes.uint16)[:transform.values_bytes//2].reshape(transform.rows,transform.k//2) \
    .reshape(transform.rows,transform.k//2,1).expand(transform.rows,transform.k//2,2) \
    .reshape(transform.rows,transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


@dataclass(frozen=True)
class CompilerKPP512Binding:
  producer: object
  main_program: object
  transform: PackedWeightTransform
  activation: Q8ActivationRecordTransform
  context: _Context
  warmstart: Mapping
  warmstart_contexts: Mapping

  @classmethod
  def compile(cls, dev, role:str="attn_k") -> "CompilerKPP512Binding":
    wt, at = PackedWeightTransform("Q4_K",N,K), Q8ActivationRecordTransform(M,K)
    wp, ap = Q4KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
    accumulator = Q4KQ8GroupAccumulatorContract(wp,ap)
    stride = TILE_K+(TILE_K//16)*4
    geometry = KernelTileGeometry((64,32,TILE_K),(2,2),128,32,
      (KernelLDSWindow("A",0,64*stride,stride), KernelLDSWindow("B",64*stride,96*stride,stride)))
    if role not in LEGAL_ROLES: raise ValueError(f"unsupported Q4 K role {role}")
    identity = hashlib.sha256(repr((geometry,wp.identity,ap.identity,accumulator.abi)).encode()).hexdigest()
    context = _Context("boltbeam.full_kernel_candidate.v1",identity,geometry,wt,wp,at,ap,accumulator)
    # Role-qualified key prevents V compilation from reusing K's ambient
    # candidate context while retaining the same ProgramInfo ABI.
    # The optimizer's canonical lookup key is shape-based; role isolation is
    # carried by the context identity and compilation order, not by changing
    # this established key ABI.
    key = warmstart_key({M,N},K,wt.storage_dtype)
    lib = NVRTCCompiler(dev.arch,ptx=False,cache_key=f"nv_q8_compact_record_fp16_{role}_v1").compile(_record_source())
    producer = native_nv_program("q8_compact_record_fp16",lib,global_size=(M,8,1),local_size=(128,1,1),
                                 globals=(0,1),outs=(1,),ins=(0,))
    warmstart, contexts = {key:(Opt(OptOps.TC,0,(-1,2,1)),)}, {key:context}

    # Compile once through the ordinary packed carrier matmul and retain the
    # exact compiler PROGRAM as an immutable reusable asset.
    from tinygrad.codegen import to_program_cache
    from tinygrad.codegen.opt.postrange import warmstart_candidate_state
    record_probe = Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device="NV").realize()
    words_probe = Tensor.empty(wt.packed_bytes//4,dtype=dtypes.uint32,device="NV").realize()
    with warmstart_candidate_state(warmstart,contexts):
      _activation_carrier(record_probe,at).matmul(_weight_carrier(words_probe,wt).transpose(),dtype=dtypes.int) \
        .cast(dtypes.float).contiguous().realize()
    matching = [program for program in to_program_cache.values() if program.op is Ops.PROGRAM and program.src and
                getattr(program.src[0].arg,"candidate_context",None) is not None and
                program.src[0].arg.candidate_context.canonical_identity == identity]
    if len(set(matching)) != 1: raise RuntimeError(f"expected one compiler K PROGRAM, found {len(set(matching))}")
    compiled = matching[0]
    main_program = compiled.replace(src=(UOp(Ops.SINK,arg=compiled.src[0].arg),compiled.src[1],UOp(Ops.LINEAR),*compiled.src[3:]))
    if main_program.arg.outs != (0,) or main_program.arg.ins != (1,2):
      raise RuntimeError(f"compiler K PROGRAM has unexpected ABI {main_program.arg}")
    if (main_program.arg.global_size,main_program.arg.local_size) != ((32,8,1),(32,2,2)):
      raise RuntimeError(f"compiler K PROGRAM lost qualified 256-CTA geometry: {main_program.arg}")
    return cls(producer,main_program,wt,at,context,MappingProxyType(warmstart),MappingProxyType(contexts))

  @property
  def candidate_identity(self) -> str: return self.context.canonical_identity

  def install_warmstart(self, model) -> None:
    opts, contexts = dict(model._packed_wmma_warmstart or {}), dict(model._packed_wmma_warmstart_contexts or {})
    for key,value in self.warmstart.items():
      if key in opts and opts[key] != value: raise RuntimeError("compiler K warmstart key collides with another route")
      opts[key] = value
    for key,value in self.warmstart_contexts.items():
      if key in contexts and contexts[key] != value: raise RuntimeError("compiler K context key collides with another route")
      contexts[key] = value
    model._packed_wmma_warmstart, model._packed_wmma_warmstart_contexts = opts, contexts

  def new_capture(self) -> "CompilerKPP512Capture": return CompilerKPP512Capture(self)

  def prepare_records(self,count:int) -> None:
    if count != PROJECTIONS_PER_MODEL: raise ValueError(f"exact K route requires {PROJECTIONS_PER_MODEL} projections")

  @property
  def records(self) -> range: return range(PROJECTIONS_PER_MODEL)

  @property
  def outputs(self) -> range: return range(PROJECTIONS_PER_MODEL)

  def begin_trace(self) -> None: pass

  def project(self,x:Tensor,words:Tensor,*,model_family:str,role:str,weight_type:str="Q4_K",wait:bool=False) -> Tensor:
    return _project(self,x,words,model_family=model_family,role=role,weight_type=weight_type,wait=wait)


@dataclass
class CompilerKPP512Capture:
  asset: CompilerKPP512Binding
  trace_epoch: int = 0
  cursor: int = 0

  @property
  def candidate_identity(self) -> str: return self.asset.candidate_identity
  @property
  def transform(self) -> PackedWeightTransform: return self.asset.transform
  @property
  def producer(self): return self.asset.producer
  @property
  def records(self) -> range: return range(self.cursor)
  @property
  def outputs(self) -> range: return range(self.cursor)

  def prepare_records(self,count:int) -> None: self.asset.prepare_records(count)

  def begin_trace(self) -> None: self.trace_epoch, self.cursor = self.trace_epoch+1, 0

  def project(self,x:Tensor,words:Tensor,*,model_family:str,role:str,weight_type:str="Q4_K",wait:bool=False) -> Tensor:
    if self.trace_epoch == 0: raise RuntimeError("begin_trace must establish a K capture-local epoch")
    if self.cursor >= PROJECTIONS_PER_MODEL: raise RuntimeError("compiler K trace exceeded exact 36-projection census")
    self.cursor += 1
    return _project(self.asset,x,words,model_family=model_family,role=role,weight_type=weight_type,wait=wait)


def _project(binding:CompilerKPP512Binding,x:Tensor,words:Tensor,*,model_family:str,role:str,
             weight_type:str="Q4_K",wait:bool=False) -> Tensor:
  del wait
  if not supports(model_family=model_family,role=role,weight_type=weight_type,
                  m=x.shape[0],n=N,k=x.shape[1],device=x.device):
    raise ValueError("unsupported compiler Q4 K research route")
  if x.dtype != dtypes.float16 or words.dtype != dtypes.uint32:
    raise ValueError("compiler Q4 K route requires fp16 activation and canonical uint32 Q4_K words")
  record = Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device=x.device)
  _, record = x.uop_program(record,fxn=lambda *_:binding.producer)
  # Keep the main as an ordinary compiler-owned matmul inside the attention
  # function.  The immutable asset above pins its exact candidate identity and
  # launch; normal scheduling then owns the output lifetime instead of routing
  # an opaque preallocated output through the nested block FUNCTION.
  return _activation_carrier(record,binding.activation).matmul(
    _weight_carrier(words,binding.transform).transpose(),dtype=dtypes.int).cast(dtypes.float).contiguous()


def binding_for(device:str="NV", role:str="attn_k") -> CompilerKPP512Binding:
  if device != "NV": raise ValueError("compiler Q4 K research binding is NV-only")
  key=f"{device}:{role}"
  if key not in _BINDINGS:
    _BINDINGS[key] = CompilerKPP512Binding.compile(Device[device], role)
  return _BINDINGS[key]
