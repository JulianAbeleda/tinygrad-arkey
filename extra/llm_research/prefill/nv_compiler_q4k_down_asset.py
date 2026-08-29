"""Immutable compiler-owned Q4_K FFN-down asset (M512,N4096,K12288).

This is isolated from the K asset: distinct geometry, identity and cache key.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_key, warmstart_candidate_state
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_q8_k12288_source import source_k12288_record

M,N,K,TILE_K=512,4096,12288,64
PROJECTIONS_PER_MODEL=18
RECORD_U32=(M*K+2*M*(K//32)*4)//4

def supports(*,model_family,role,weight_type,m,n,k,device,ggml_type=12):
  return model_family=="qwen3_8b" and role=="ffn_down" and weight_type=="Q4_K" and ggml_type==12 and (m,n,k)==(M,N,K) and device=="NV"

def _wc(words,wt):
  blocks,hw=wt.rows*wt.blocks_per_row,int(wt.block_bytes)//2
  return words.bitcast(dtypes.uint16).reshape(blocks,hw).pad(((0,0),(0,128-hw))).reshape(blocks,128,1).expand(blocks,128,2).reshape(wt.rows,wt.k).bitcast(dtypes.float16).cast(dtypes.int8)
def _ac(record,at):
  return record.bitcast(dtypes.uint16)[:at.values_bytes//2].reshape(at.rows,at.k//2).reshape(at.rows,at.k//2,1).expand(at.rows,at.k//2,2).reshape(at.rows,at.k).bitcast(dtypes.float16).cast(dtypes.int8)

@dataclass(frozen=True)
class DownAsset:
  producer: object; main_program: object; transform: object; activation: object; candidate_identity: str
  warmstart: object; warmstart_contexts: object
  @classmethod
  def compile(cls,dev):
    wt,at=PackedWeightTransform("Q4_K",N,K),Q8ActivationRecordTransform(M,K)
    wp,ap=Q4KInt8FragmentProvider(wt),Q8Int8FragmentProvider(at)
    acc=Q4KQ8GroupAccumulatorContract(wp,ap)
    stride=TILE_K+(TILE_K//16)*4
    geom=KernelTileGeometry((64,32,TILE_K),(2,2),128,32,(KernelLDSWindow("A",0,64*stride,stride),KernelLDSWindow("B",64*stride,96*stride,stride)))
    ident=hashlib.sha256(repr(("ffn_down",geom,wp.identity,ap.identity,acc.abi)).encode()).hexdigest()
    key=warmstart_key({M,N},K,wt.storage_dtype); context=type("DownContext",(),{"schema_version":"boltbeam.full_kernel_candidate.v1","canonical_identity":ident,"geometry":geom,"packed_weight":wt,"packed_fragment_provider":wp,"packed_activation":at,"packed_activation_provider":ap,"group_accumulator":acc})()
    # Reuse the proven compact-record ABI, specializing both the input row
    # stride and scale/sum group stride for K=12288.
    # Use the independently qualified K=12288 producer source.  The former
    # string-specialized copy drifted from the saved-Z gate's ABI/rounding.
    record_src=source_k12288_record()
    lib=NVRTCCompiler(dev.arch,ptx=False,cache_key="nv_q8_compact_record_fp16_down_k12288_v3").compile(record_src)
    prod=native_nv_program("q8_compact_record_fp16_k12288",lib,global_size=(M,24,1),local_size=(128,1,1),globals=(0,1),outs=(1,),ins=(0,))
    opts,ctxs={key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}
    from tinygrad.codegen import to_program_cache
    rp=Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device="NV").realize(); wpb=Tensor.empty(wt.packed_bytes//4,dtype=dtypes.uint32,device="NV").realize()
    with warmstart_candidate_state(opts,ctxs): _ac(rp,at).matmul(_wc(wpb,wt).transpose(),dtype=dtypes.int).cast(dtypes.float).contiguous().realize()
    ms=[p for p in to_program_cache.values() if p.op is Ops.PROGRAM and p.src and getattr(p.src[0].arg,"candidate_context",None) is not None and p.src[0].arg.candidate_context.canonical_identity==ident]
    if len(set(ms))!=1: raise RuntimeError(f"expected one down tileK64 PROGRAM, found {len(set(ms))}")
    p=ms[0]; main=p.replace(src=(UOp(Ops.SINK,arg=p.src[0].arg),p.src[1],UOp(Ops.LINEAR),*p.src[3:]))
    if main.arg.outs!=(0,) or main.arg.ins!=(1,2): raise RuntimeError(f"unexpected down ABI {main.arg}")
    return cls(prod,main,wt,at,ident,MappingProxyType(opts),MappingProxyType(ctxs))

_CACHE={}
def binding_for(device="NV"):
  if device!="NV": raise ValueError("Q4 down asset is NV-only")
  if device not in _CACHE: _CACHE[device]=DownAsset.compile(Device[device])
  return _CACHE[device]
