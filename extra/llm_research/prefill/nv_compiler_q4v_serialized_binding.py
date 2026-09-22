"""Default-off, immutable serialized Q4-K attention-V pp512 asset.

The V main PROGRAM is loaded from the finalized cubin produced by
``nv_compiler_q4v_asset_build``.  In particular this module never compiles a
V matmul in the process which owns the established K route.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, os
from pathlib import Path
from types import MappingProxyType
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt.packed_weight import PackedWeightTransform, Q8ActivationRecordTransform
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import _record_source, _activation_carrier, _weight_carrier, RECORD_U32

M,N,K=512,1024,4096
PROJECTIONS_PER_MODEL=18
DEFAULT_ASSET=Path('/tmp/q4v-asset')
_BINDINGS={}

def supports(*, model_family, role, weight_type, m, n, k, device):
  return model_family=='qwen3_8b' and role=='attn_v' and weight_type=='Q4_K' and (m,n,k)==(M,N,K) and device=='NV'

@dataclass(frozen=True)
class SerializedVAssetBinding:
  producer: object
  main_program: object
  transform: PackedWeightTransform
  activation: Q8ActivationRecordTransform
  candidate_identity: str
  manifest: object

  @classmethod
  def load(cls, dev, asset_dir=DEFAULT_ASSET):
    root=Path(asset_dir); manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema')!='tinygrad.nv.q4v.asset.v1': raise ValueError('unsupported Q4 V asset schema')
    binary=(root/'program.cubin').read_bytes()
    if hashlib.sha256(binary).hexdigest()!=manifest.get('binary_sha256'): raise ValueError('Q4 V cubin digest mismatch')
    if binary[:4]!=b'\x7fELF' or tuple(manifest.get('global_size',()))!=(32,8,1) or tuple(manifest.get('local_size',()))!=(32,2,2): raise ValueError('invalid Q4 V launch ABI')
    if tuple(manifest.get('globals',()))!=(0,1,2) or tuple(manifest.get('outs',()))!=(0,) or tuple(manifest.get('ins',()))!=(1,2): raise ValueError('invalid Q4 V buffer ABI')
    wt,at=PackedWeightTransform('Q4_K',N,K),Q8ActivationRecordTransform(M,K)
    lib=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_q8_compact_record_fp16_v_serialized_v1').compile(_record_source())
    # `_record_source()` exports the ordinary symbol.  The native PROGRAM
    # name is part of the CUDA function ABI; inventing a role-specific name
    # here leaves the launcher pointing at a non-existent entry point.
    producer=native_nv_program('q8_compact_record_fp16',lib,global_size=(M,8,1),local_size=(128,1,1),globals=(0,1),outs=(1,),ins=(0,))
    main=native_nv_program(manifest['name'],binary,global_size=tuple(manifest['global_size']),local_size=tuple(manifest['local_size']),globals=tuple(manifest['globals']),outs=tuple(manifest['outs']),ins=tuple(manifest['ins']),vals=tuple(manifest.get('vals',())),shared_mem=(manifest.get('aux') or [0])[0])
    return cls(producer,main,wt,at,manifest['identity'],MappingProxyType(manifest))

  def prepare_records(self,count):
    if count!=18: raise ValueError('serialized Q4 V route requires exactly 18 projections')
  def new_capture(self): return SerializedVCapture(self)
  def project(self,x,words,**kw): return _project(self,x,words,**kw)

@dataclass
class SerializedVCapture:
  asset: SerializedVAssetBinding
  cursor:int=0
  def begin_trace(self): self.cursor=0
  @property
  def candidate_identity(self): return self.asset.candidate_identity
  @property
  def transform(self): return self.asset.transform
  @property
  def producer(self): return self.asset.producer
  def project(self,x,words,**kw):
    if self.cursor>=18: raise RuntimeError('serialized Q4 V trace exceeded exact 18-projection census')
    self.cursor+=1; return _project(self.asset,x,words,**kw)

def _project(b,x,words,*,model_family,role,weight_type='Q4_K',wait=False):
  del wait
  if not supports(model_family=model_family,role=role,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device): raise ValueError('unsupported serialized Q4 V route')
  if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32: raise ValueError('serialized Q4 V requires fp16 activation and uint32 weights')
  record=Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device=x.device); _,record=x.uop_program(record,fxn=lambda *_:b.producer)
  # Keep both stages opaque and graph-owned: the serialized asset's V main
  # PROGRAM is the only admitted consumer.  Rebuilding a carrier matmul here
  # would silently compile a fresh V kernel in the parent process.
  out=Tensor.empty(M*N,dtype=dtypes.float32,device=x.device)
  out,record,words=out.uop_program(record,words,fxn=lambda *_:b.main_program)
  return out.reshape(M,N)

def binding_for(device='NV', asset_dir=None):
  if device!='NV': raise ValueError('serialized Q4 V binding is NV-only')
  key=f'{device}:{asset_dir or DEFAULT_ASSET}'
  if key not in _BINDINGS: _BINDINGS[key]=SerializedVAssetBinding.load(Device[device],asset_dir or DEFAULT_ASSET)
  return _BINDINGS[key]
