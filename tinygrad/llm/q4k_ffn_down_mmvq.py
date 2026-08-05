"""Closed-default native Q4_K FFN-down MMVQ admission candidate.

The route is deliberately unreachable unless a research harness attaches an
explicit ``Q4KFFNDownMMVQAdmission`` to one concrete Q4_K FFN-down linear. It
keeps the installed fp16 activation boundary, packs that buffer to llama's
Q8_1 CUDA ABI, and consumes it with four warps per output row. There is no
W1/W3 producer redesign and no generic environment-variable selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import _staged_shfl, _warp_reduce_sum_staged, warp_reduce_max
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import Q4K_WORDS_PER_BLOCK, _f16_word
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

ROWS, K, QK = 4096, 12288, 256
WARP, WARPS_PER_ROW = 32, 4
Q4_BLOCKS, BLOCKS_PER_WARP = K//QK, (K//QK)//WARPS_PER_ROW
Q8_PAYLOAD_WORDS, Q8_GROUPS = K//4, K//32
Q8_WORDS = Q8_PAYLOAD_WORDS + Q8_GROUPS


@dataclass(frozen=True)
class Q4KFFNDownMMVQAdmission:
  block_index: int
  owned_input_boundary: bool = False
  def __post_init__(self):
    if not isinstance(self.block_index,int) or isinstance(self.block_index,bool) or self.block_index < 0:
      raise ValueError("Q4_K FFN-down MMVQ block index must be a non-negative integer")
    if not isinstance(self.owned_input_boundary,bool): raise ValueError("owned_input_boundary must be bool")


def _pack4(values:list[UOp]) -> UOp:
  out=UOp.const(dtypes.uint32,0)
  for i,value in enumerate(values): out=out.bitwise_or(value.cast(dtypes.uint8).cast(dtypes.uint32).lshift(8*i))
  return out


def _i8lane(packet:UOp, lane:int) -> UOp:
  return packet.rshift(8*lane).bitwise_and(255).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32)


def _q8_d(packed:UOp, group:UOp) -> UOp:
  return packed[Q8_PAYLOAD_WORDS+group].bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def ownership_coordinates() -> list[tuple[int,int,int,int,int]]:
  return [(warp,lane,warp*BLOCKS_PER_WARP+block_rel,lane//4,(lane%4)*2+slot)
          for warp in range(WARPS_PER_ROW) for lane in range(WARP)
          for block_rel in range(BLOCKS_PER_WARP) for slot in range(2)]


def emit_q8_provider(source_dtype=dtypes.float16) -> callable:
  """Production Q8_1 provider with the fp16 boundary explicit or owned."""
  if source_dtype not in (dtypes.float16,dtypes.float32): raise ValueError("Q8 provider source must be fp16 or fp32")
  def kernel(out:UOp, x:UOp) -> UOp:
    block=UOp.special(Q8_GROUPS//8,"gidx0"); lid=UOp.special(8*WARP,"lidx0")
    warp,lane=lid//WARP,lid%WARP; group=block*8+warp
    rounded=x[group*WARP+lane].cast(dtypes.float16).cast(dtypes.float32)
    amax=warp_reduce_max(rounded.abs(),lane,WARP,100); d=amax/UOp.const(dtypes.float32,127.)
    # Keep llama CUDA's x/d spelling and roundf ties-away rule. UOp.round is
    # ties-to-even and reciprocal multiplication moves a live fp16 tie.
    scaled=d.eq(0.).where(UOp.const(dtypes.float32,0.),rounded/d)
    roundf=(scaled>=0.).where(scaled+UOp.const(dtypes.float32,.5),scaled-UOp.const(dtypes.float32,.5)).cast(dtypes.int32)
    q=roundf.maximum(UOp.const(dtypes.int32,-128)).minimum(UOp.const(dtypes.int32,127)).cast(dtypes.int8)
    q1=_staged_shfl(q,1,lane,110); q2=_staged_shfl(q,2,lane,111); q3=_staged_shfl(q,3,lane,112)
    payload=out[group*8+lane//4].store(_pack4([q,q1,q2,q3]),lane.bitwise_and(3).eq(0))
    raw_sum=_warp_reduce_sum_staged(rounded,lane,WARP,120)
    dh=d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    sh=raw_sum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    lane0=lane.eq(0); metadata_index=lane0.where(Q8_PAYLOAD_WORDS+group,UOp.const(dtypes.weakint,0))
    metadata=out[metadata_index].store(dh.bitwise_or(sh.lshift(16)),lane0)
    return UOp.group(payload,metadata).sink(arg=KernelInfo(name="q8_1_llama_provider_12288",opts_to_apply=()))
  return kernel


def owned_boundary_topology() -> dict:
  """Static one-to-one topology/lifetime contract for the dormant successor."""
  return {"schema":"tinygrad.q4k_ffn_down_mmvq_owned_boundary.v1",
    "control_nodes":["w1w3_fused_fp32","fp16_materialize","installed_q4_ffn_down"],
    "candidate_nodes":["w1w3_fused_fp32","fp32_to_fp16_q8_provider","q4_q8_direct_consumer"],
    "control_node_count":3,"candidate_node_count":3,"net_graph_members":0,
    "ownership":{"w1w3_fused_fp32":"produced once; consumed once by Q8 provider",
      "packed_q8":"provider-owned fresh buffer; consumed once by direct Q4/Q8 kernel",
      "ffn_down_fp32":"consumer-owned fresh output; consumed by unchanged residual chain"},
    "removed_intermediate":"fp16 activation materialization",
    "semantic_rounding":"provider casts fp32 input to fp16 and back to fp32 before live-llama x/d quantization",
    "gpu_gate":"candidate graph must be exactly 875 calls with control partitions before timing"}


def emit_four_warp_direct(block_count:UOp, *, sum_dp4a:bool=False) -> callable:
  """Four-warp Q4/Q8 DP4A consumer with shared partial staging and direct output.

  ``sum_dp4a`` is an opt-in research spelling. It replaces the four signed
  byte extracts used for the Q8 correction sum with llama's exact
  ``dp4a(0x01010101, q8)`` form. It remains false for every production caller.
  """
  def kernel(out:UOp, words:UOp, packed:UOp) -> UOp:
    row,lid=UOp.special(ROWS,"gidx0"),UOp.special(WARP*WARPS_PER_ROW,"lidx0")
    warp,lane=lid//WARP,lid%WARP; group,word_base=lane//4,(lane%4)*2
    block_rel=UOp.range(block_count,2,axis_type=AxisType.LOOP); block=warp*BLOCKS_PER_WARP+block_rel
    base=(row*Q4_BLOCKS+block)*Q4K_WORDS_PER_BLOCK
    w0,w1,w2,w3=words[base],words[base+1],words[base+2],words[base+3]
    d,dmin=_f16_word(w0,False),_f16_word(w0,True); g4=group%4
    b1=w1.rshift(g4*8).bitwise_and(0xff); b2=w2.rshift(g4*8).bitwise_and(0xff); hb=w3.rshift(g4*8).bitwise_and(0xff)
    scale=(group<4).where(b1.bitwise_and(63),hb.bitwise_and(0xf).bitwise_or(b1.rshift(6).lshift(4)))
    minimum=(group<4).where(b2.bitwise_and(63),hb.rshift(4).bitwise_or(b2.rshift(6).lshift(4)))
    contribution=UOp.const(dtypes.float32,0.)
    for slot in range(2):
      word=word_base+slot
      qw=words[base+4+(group//2)*8+word].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
      xv=packed[block*64+group*8+word]
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xv).cast(dtypes.float32)
      xsum=(int8x4_dot(UOp.const(dtypes.int32,0),UOp.const(dtypes.uint32,0x01010101),xv)
            if sum_dp4a else _i8lane(xv,0)+_i8lane(xv,1)+_i8lane(xv,2)+_i8lane(xv,3))
      contribution=contribution+_q8_d(packed,block*8+group)*(
        d*scale.cast(dtypes.float32)*dot-dmin*minimum.cast(dtypes.float32)*xsum.cast(dtypes.float32))
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); acc=acc.after(acc[0].store(0.))
    acc=acc.after(acc[0].store(acc.after(block_rel)[0]+contribution).end(block_rel)); partial=acc[0]
    shared=UOp.placeholder(((WARPS_PER_ROW-1)*WARP,),dtypes.float32,40,addrspace=AddrSpace.LOCAL)
    other_warp=warp>0; shared_index=other_warp.where((warp-1)*WARP+lane,UOp.const(dtypes.weakint,0))
    publish=shared[shared_index].store(partial,other_warp); ready=UOp.barrier(UOp.group(publish)); total=partial
    for other in range(WARPS_PER_ROW-1): total=total+shared.after(ready)[other*WARP+lane]
    for slot,offset in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,offset,lane,slot)
    return out[row].store(total,warp.eq(0)&lane.eq(0)).sink(
      arg=KernelInfo(name="q4k_q8_mmvq_direct_4096_12288",opts_to_apply=()))
  return kernel


def q4k_ffn_down_mmvq_call(admission:object, linear:Any, x:Tensor, binding:Any,
                            epilogue_inputs:dict[str,Tensor]) -> Tensor|None:
  """Return the leased candidate, or None without changing the installed path."""
  if not isinstance(admission,Q4KFFNDownMMVQAdmission): return None
  capability=getattr(getattr(linear,"route_admission",None),"capability",None)
  if (getattr(capability,"backend",None),getattr(capability,"architecture",None)) != ("NV","sm_120"): return None
  if (getattr(linear,"route_role",None),binding.N,binding.K) != ("ffn_down",ROWS,K): return None
  if epilogue_inputs or getattr(linear,"bias",None) is not None or not str(x.device).startswith("NV"): return None
  words=linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
  if admission.owned_input_boundary:
    if x.dtype != dtypes.float32: return None
    xv=x[:,0,:].reshape(K)
  else: xv=x[:,0,:].reshape(K).cast(dtypes.float16).contiguous()
  provider=KernelProgram("decode_q4k_ffn_down_mmvq",f"blk{admission.block_index}.q8_provider",
    KernelProgramProvenance.RESEARCH_ONLY,emit_q8_provider(dtypes.float32 if admission.owned_input_boundary else dtypes.float16))
  packed=execute_research_program(Tensor.empty((Q8_WORDS,),dtype=dtypes.uint32,device=x.device),xv,program=provider)
  consumer=KernelProgram("decode_q4k_ffn_down_mmvq",f"blk{admission.block_index}.consumer",
    KernelProgramProvenance.RESEARCH_ONLY,emit_four_warp_direct(UOp.const(dtypes.weakint,BLOCKS_PER_WARP)))
  out=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device),words,packed,program=consumer)
  return out.reshape(1,1,ROWS)


__all__=["Q4KFFNDownMMVQAdmission","ROWS","K","Q4_BLOCKS","BLOCKS_PER_WARP","Q8_PAYLOAD_WORDS",
         "Q8_GROUPS","Q8_WORDS","ownership_coordinates","owned_boundary_topology","emit_q8_provider","emit_four_warp_direct",
         "q4k_ffn_down_mmvq_call"]
