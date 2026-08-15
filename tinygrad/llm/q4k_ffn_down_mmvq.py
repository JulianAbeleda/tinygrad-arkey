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
from tinygrad.codegen.late.warp_reduce import WARP, _staged_shfl, _warp_reduce_sum_staged, warp_reduce_max
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGateUpLaneMap, Q4K_WORDS_PER_BLOCK, _f16_word,
  _q4k_block_dot_packed_load, _silu_uop)
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance,
  OutputSpec, ResidualViewRequest, TypedLayout, TypedViewRequest, execute_promoted_program,
  execute_research_program)
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

ROWS, K, QK = 4096, 12288, 256
WARP, WARPS_PER_ROW = 32, 4
Q4_BLOCKS, BLOCKS_PER_WARP = K//QK, (K//QK)//WARPS_PER_ROW
SUB_BLOCKS = BLOCKS_PER_WARP // WARPS_PER_ROW
Q8_PAYLOAD_WORDS, Q8_GROUPS = K//4, K//32
Q8_WORDS = Q8_PAYLOAD_WORDS + Q8_GROUPS


@dataclass(frozen=True)
class Q4KFFNDownMMVQAdmission:
  block_index: int
  owned_input_boundary: bool = False
  fp16_fma: bool = False
  scalar_q8_packet: bool = False
  def __post_init__(self):
    if not isinstance(self.block_index,int) or isinstance(self.block_index,bool) or self.block_index < 0:
      raise ValueError("Q4_K FFN-down MMVQ block index must be a non-negative integer")
    if not isinstance(self.owned_input_boundary,bool): raise ValueError("owned_input_boundary must be bool")
    if not isinstance(self.fp16_fma,bool): raise ValueError("fp16_fma must be bool")
    if not isinstance(self.scalar_q8_packet,bool): raise ValueError("scalar_q8_packet must be bool")
    if self.owned_input_boundary and self.fp16_fma:
      raise ValueError("owned_input_boundary and fp16_fma are mutually exclusive research spellings")
    if self.scalar_q8_packet and (self.owned_input_boundary or self.fp16_fma):
      raise ValueError("scalar_q8_packet is mutually exclusive with owned_input_boundary and fp16_fma")


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


def emit_ffn_w1w3_q8_scalar_packet() -> callable:
  """Fold llama Q8_1 quantization into the W1/W3 producer epilogue.

  One 1024-thread CTA owns a 32-row Q8_1 packet. Every warp reduces one GLU row
  with the same 32-lane scalar map as ``q4k_g3_lanemap_gemv_w1w3_kernel``, so
  the gate/up arithmetic is bitwise identical to the fused16 producer. Lane zero
  of each warp publishes its fp16 result to a 64-byte LOCAL array, then warp
  zero performs the established ``quantize_q8_1`` CUDA spelling after the CTA
  barrier. The output is the existing 3072+384-word packed ABI; there is no
  fp16 activation buffer and no separate ``q8_1_llama_provider_12288`` node.
  """
  if (K, ROWS) != (12288, 4096): raise ValueError("scalar-packet producer is fixed to Qwen 12288x4096")
  lm = Q4KGateUpLaneMap(k=ROWS, n=K); lm.validate()
  grid_x, threads, pack = K // 32, 32 * 32, 32
  blocks_per_group, k_blocks = lm.blocks_per_group, lm.k_blocks

  def kernel(out:UOp, gate_words:UOp, up_words:UOp, x:UOp) -> UOp:
    packet = UOp.special(grid_x, "gidx0")
    lid = UOp.special(threads, "lidx0")
    warp, lane = lid // WARP, lid % WARP
    row = packet * pack + warp
    part = LanePartition(lane, lane_extent=WARP, words_per_group=8)
    lblk = UOp.range(blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * blocks_per_group + lblk
    base_g = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    base_u = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib_g = _q4k_block_dot_packed_load(gate_words, x, base_g, blk, part.word_col)
    contrib_u = _q4k_block_dot_packed_load(up_words, x, base_u, blk, part.word_col)

    acc_g = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc_u = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG)
    init = acc_g[0].store(0.0)
    init = acc_u.after(init)[0].store(0.0)
    acc_g, acc_u = acc_g.after(init), acc_u.after(init)
    upd_g = acc_g[0].store(acc_g.after(lblk)[0] + contrib_g)
    upd_u = acc_u.after(upd_g)[0].store(acc_u.after(lblk)[0] + contrib_u).end(lblk)
    total_g = _warp_reduce_sum_staged(acc_g.after(upd_u)[0], part.lane, part.lane_extent, 90)
    total_u = _warp_reduce_sum_staged(acc_u.after(upd_u)[0], part.lane, part.lane_extent, 95)
    z = _silu_uop(total_g).mul(total_u).cast(dtypes.float16)

    zsh = UOp.placeholder((pack,), dtypes.float16, 30, addrspace=AddrSpace.LOCAL)
    published = zsh[warp].store(z, lane.eq(0))
    ready = UOp.barrier(UOp.group(published))
    consumer = ready.post_barrier_region(warp.eq(0))
    rounded = zsh.after(consumer)[lane].cast(dtypes.float32)
    amax = warp_reduce_max(rounded.abs(), lane, WARP, 100)
    d = amax / UOp.const(dtypes.float32, 127.0)
    scaled = d.eq(0.0).where(UOp.const(dtypes.float32, 0.0), rounded / d)
    roundf = (scaled >= 0.0).where(scaled + UOp.const(dtypes.float32, 0.5),
      scaled - UOp.const(dtypes.float32, 0.5)).cast(dtypes.int32)
    q = roundf.maximum(UOp.const(dtypes.int32, -128)).minimum(UOp.const(dtypes.int32, 127)).cast(dtypes.int8)
    q1 = _staged_shfl(q, 1, lane, 110)
    q2 = _staged_shfl(q, 2, lane, 111)
    q3 = _staged_shfl(q, 3, lane, 112)
    payload = out[packet * 8 + lane // 4].store(_pack4([q, q1, q2, q3]), lane.bitwise_and(3).eq(0))
    xsum = _warp_reduce_sum_staged(rounded, lane, WARP, 120)
    dh = d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    sh = xsum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    metadata_idx = lane.eq(0).where(UOp.const(dtypes.weakint, Q8_PAYLOAD_WORDS) + packet, UOp.const(dtypes.weakint, 0))
    metadata = out[metadata_idx].store(dh.bitwise_or(sh.lshift(16)), lane.eq(0))
    return consumer.end_region(UOp.group(payload, metadata)).sink(
      arg=KernelInfo(name="ffn_w1w3_q8_scalar_packet_12288_4096", opts_to_apply=()))
  return kernel


def emit_four_warp_fp16_direct(block_count:UOp, *, resadd:bool=False) -> callable:
  """Four-warp fp16-FMA Q4_K FFN-down consumer (occupancy/geometry research spelling).

  This is the isolated geometry lever from the occupancy proof: 128 threads/row
  across 4 warps like llama's Q4 MMQ, while the datapath stays the installed
  scalar fp16 route -- the same packed weight loads/dequant and fp32 FMA,
  consuming the ``w1w3fused16`` fp16 activation directly. There is no Q8 provider
  node and no activation quantization, so the only numeric change versus the
  installed 1-warp kernel is fp32 reduction reorder (four warp partials), not a
  quantization. Each warp owns 12 of the 48 Q4 blocks; inside a warp the 32 lanes
  keep the installed (word_col=lane%8, sub_group=lane//8) partition over 3 blocks
  each. Cross-warp partials combine through shared memory, then a staged shuffle
  reduces the 32 lanes. ``resadd`` absorbs M2b ``h + ffn_out`` in-kernel.
  """
  def kernel(out:UOp, words:UOp, x:UOp, *extra:UOp) -> UOp:
    row,lid=UOp.special(ROWS,"gidx0"),UOp.special(WARP*WARPS_PER_ROW,"lidx0")
    warp,lane=lid//WARP,lid%WARP
    word_col=lane%8; sub_group=lane//8
    block_rel=UOp.range(block_count,2,axis_type=AxisType.LOOP)
    block=warp*BLOCKS_PER_WARP+sub_group*block_count+block_rel
    base=(row*Q4_BLOCKS+block)*Q4K_WORDS_PER_BLOCK
    contribution=_q4k_block_dot_packed_load(words,x,base,block,word_col)
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); acc=acc.after(acc[0].store(0.))
    acc=acc.after(acc[0].store(acc.after(block_rel)[0]+contribution).end(block_rel)); partial=acc[0]
    shared=UOp.placeholder(((WARPS_PER_ROW-1)*WARP,),dtypes.float32,40,addrspace=AddrSpace.LOCAL)
    other_warp=warp>0; shared_index=other_warp.where((warp-1)*WARP+lane,UOp.const(dtypes.weakint,0))
    publish=shared[shared_index].store(partial,other_warp); ready=UOp.barrier(UOp.group(publish)); total=partial
    for other in range(WARPS_PER_ROW-1): total=total+shared.after(ready)[other*WARP+lane]
    for slot,offset in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,offset,lane,slot)
    result=total+extra[0][row] if resadd else total
    name="q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd" if resadd else "q4k_fp16_mmvq_direct_4096_12288"
    return out[row].store(result,warp.eq(0)&lane.eq(0)).sink(arg=KernelInfo(name=name,opts_to_apply=()))
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


def emit_four_warp_direct(block_count:UOp, *, sum_dp4a:bool=False, resadd:bool=False) -> callable:
  """Four-warp Q4/Q8 DP4A consumer with shared partial staging and direct output.

  ``sum_dp4a`` is an opt-in research spelling. It replaces the four signed
  byte extracts used for the Q8 correction sum with llama's exact
  ``dp4a(0x01010101, q8)`` form. It remains false for every production caller.

  ``resadd`` absorbs the M2b ``h + ffn_out`` residual add in-kernel: the kernel
  takes the fp32 ``normed_h`` residual as a fourth slot and stores
  ``total + residual[row]``, matching the installed ``_epi_ffnresadd`` control
  epilogue bit-for-bit. The consumer name gains the ``_epi_ffnresadd`` suffix.
  """
  def kernel(out:UOp, words:UOp, packed:UOp, *extra:UOp) -> UOp:
    row,lid=UOp.special(ROWS,"gidx0"),UOp.special(WARP*WARPS_PER_ROW,"lidx0")
    warp,lane=lid//WARP,lid%WARP; group,word_base=lane//4,(lane%4)*2
    block_rel=UOp.range(block_count,2,axis_type=AxisType.LOOP); block=warp*BLOCKS_PER_WARP+block_rel
    base=(row*Q4_BLOCKS+block)*Q4K_WORDS_PER_BLOCK
    hdr=words.index(base).load(dtype=dtypes.uint32.vec(4))
    w0,w1,w2,w3=hdr.gep(0),hdr.gep(1),hdr.gep(2),hdr.gep(3)
    d,dmin=_f16_word(w0,False),_f16_word(w0,True); g4=group%4
    b1=w1.rshift(g4*8).bitwise_and(0xff); b2=w2.rshift(g4*8).bitwise_and(0xff); hb=w3.rshift(g4*8).bitwise_and(0xff)
    scale=(group<4).where(b1.bitwise_and(63),hb.bitwise_and(0xf).bitwise_or(b1.rshift(6).lshift(4)))
    minimum=(group<4).where(b2.bitwise_and(63),hb.rshift(4).bitwise_or(b2.rshift(6).lshift(4)))
    contribution=UOp.const(dtypes.float32,0.)
    q8d=_q8_d(packed,block*8+group)
    qw_pair=words.index(base+4+(group//2)*8+word_base).load(dtype=dtypes.uint32.vec(2))
    xv_pair=packed.index(block*64+group*8+word_base).load(dtype=dtypes.uint32.vec(2))
    for slot in range(2):
      qw=qw_pair.gep(slot).rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
      xv=xv_pair.gep(slot)
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xv)
      xsum=(int8x4_dot(UOp.const(dtypes.int32,0),UOp.const(dtypes.uint32,0x01010101),xv)
            if sum_dp4a else _i8lane(xv,0)+_i8lane(xv,1)+_i8lane(xv,2)+_i8lane(xv,3))
      # llama keeps dot*scale and xsum*min in int32 (IMAD) and converts once;
      # that removes the float-pipe FMUL/FFMA this kernel was paying per DP4A.
      scale_dot=(dot*scale.cast(dtypes.int32)).cast(dtypes.float32)
      min_sum=(xsum*minimum.cast(dtypes.int32)).cast(dtypes.float32)
      contribution=contribution+q8d*(d*scale_dot-dmin*min_sum)
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); acc=acc.after(acc[0].store(0.))
    acc=acc.after(acc[0].store(acc.after(block_rel)[0]+contribution).end(block_rel)); partial=acc[0]
    shared=UOp.placeholder(((WARPS_PER_ROW-1)*WARP,),dtypes.float32,40,addrspace=AddrSpace.LOCAL)
    other_warp=warp>0; shared_index=other_warp.where((warp-1)*WARP+lane,UOp.const(dtypes.weakint,0))
    publish=shared[shared_index].store(partial,other_warp); ready=UOp.barrier(UOp.group(publish)); total=partial
    for other in range(WARPS_PER_ROW-1): total=total+shared.after(ready)[other*WARP+lane]
    for slot,offset in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,offset,lane,slot)
    result=total+extra[0][row] if resadd else total
    name="q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd" if resadd else "q4k_q8_mmvq_direct_4096_12288"
    return out[row].store(result,warp.eq(0)&lane.eq(0)).sink(arg=KernelInfo(name=name,opts_to_apply=()))
  return kernel


def q4k_ffn_down_mmvq_call(admission:object, linear:Any, x:Tensor, binding:Any,
                            epilogue_inputs:dict[str,Tensor]) -> Tensor|None:
  """Return the leased candidate, or None without changing the installed path."""
  if not isinstance(admission,Q4KFFNDownMMVQAdmission): return None
  capability=getattr(getattr(linear,"route_admission",None),"capability",None)
  if (getattr(capability,"backend",None),getattr(capability,"architecture",None)) != ("NV","sm_120"): return None
  if (getattr(linear,"route_role",None),binding.N,binding.K) != ("ffn_down",ROWS,K): return None
  if getattr(linear,"bias",None) is not None or not str(x.device).startswith("NV"): return None
  if any(key != "normed_h" for key in epilogue_inputs): return None
  words=linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
  if admission.owned_input_boundary:
    if x.dtype != dtypes.float32: return None
    # The owned boundary consumes the fused w1w3 output WITHOUT a materialize.
    # ``x`` is the producer's exact result wrapped in equal-span reshapes; hand
    # the opaque provider the AFTER itself so custom_kernel keeps it (an input
    # that is only a view would be conservatively copied into a fp32 materialize,
    # silently replacing the fp16 cast this route exists to remove).
    owned_uop=x.uop
    owned_expected=owned_uop.numel()
    while owned_uop.op is Ops.RESHAPE and len(owned_uop.src) and owned_uop.src[0].numel()==owned_expected:
      owned_uop=owned_uop.src[0]
    if owned_uop.op is not Ops.AFTER or owned_uop.shape != (K,) or owned_uop.dtype != x.dtype: return None
    xv=Tensor(owned_uop)
  else:
    xv=x[:,0,:].reshape(K).cast(dtypes.float16).contiguous()
  resadd="normed_h" in epilogue_inputs
  residual=epilogue_inputs["normed_h"][:,0,:].reshape(ROWS).cast(dtypes.float32) if resadd else None
  if admission.fp16_fma:
    # The program_id must end in .gemv (not a bespoke suffix): the M5 epilogue-
    # absorption validator accepts only .gemv/.q8_provider and the M2b residual
    # validator accepts only .gemv/.consumer. A mismatched id silently falls back
    # to the materializing flat-buffer ABI (two extra transport kernels per block).
    consumer=KernelProgram("decode_q4k_ffn_down_mmvq",f"blk{admission.block_index}.gemv",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      emit_four_warp_fp16_direct(UOp.const(dtypes.weakint,SUB_BLOCKS),resadd=resadd),
      output_spec=OutputSpec((ROWS,),dtypes.float32,
        typed_output=(DeclaredTypedOutput(TypedLayout(dtypes.float32,(ROWS,),(1,1,ROWS)),
          combine_fusion_admitted=False,epilogue_absorption_admitted=True) if resadd else None)),
      typed_input_views=(TypedViewRequest(slot=1,dtype=dtypes.float16,flat_shape=(K,),route_role="ffn_down",
        requires_combine_fusion=False,requires_epilogue_absorption=True),),
      residual_input_views=((ResidualViewRequest(slot=2,dtype=dtypes.float32,flat_shape=(ROWS,),
        route_role="ffn_down",kind="residual_add"),) if resadd else ()))
    out=execute_promoted_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device),
      words,xv,*((residual,) if resadd else ()),program=consumer)
    return out.reshape(1,1,ROWS)
  provider_typed_views = () if admission.owned_input_boundary else (
    TypedViewRequest(slot=0,dtype=dtypes.float16,flat_shape=(K,),route_role="ffn_down",
      requires_combine_fusion=False,requires_epilogue_absorption=True),)
  provider=KernelProgram("decode_q4k_ffn_down_mmvq",f"blk{admission.block_index}.q8_provider",
    KernelProgramProvenance.RESEARCH_ONLY,emit_q8_provider(dtypes.float32 if admission.owned_input_boundary else dtypes.float16),
    typed_input_views=provider_typed_views)
  packed=execute_research_program(Tensor.empty((Q8_WORDS,),dtype=dtypes.uint32,device=x.device),xv,program=provider)
  consumer=KernelProgram("decode_q4k_ffn_down_mmvq",f"blk{admission.block_index}.consumer",
    KernelProgramProvenance.RESEARCH_ONLY,emit_four_warp_direct(UOp.const(dtypes.weakint,BLOCKS_PER_WARP),resadd=resadd),
    output_spec=OutputSpec((ROWS,),dtypes.float32,
      typed_output=(DeclaredTypedOutput(TypedLayout(dtypes.float32,(ROWS,),(1,1,ROWS)),
        combine_fusion_admitted=False,epilogue_absorption_admitted=True) if resadd else None)),
    residual_input_views=((ResidualViewRequest(slot=2,dtype=dtypes.float32,flat_shape=(ROWS,),
                                               route_role="ffn_down",kind="residual_add"),) if resadd else ()))
  out=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device),
    words,packed,*((residual,) if resadd else ()),program=consumer)
  return out.reshape(1,1,ROWS)


def q4k_ffn_down_mmvq_scalar_packet_call(gate:Any, up:Any, linear:Any, x:Tensor, residual:Tensor|None,
                                         admission:Q4KFFNDownMMVQAdmission) -> Tensor|None:
  """Two-program fold: scalar-packet W1/W3 Q8 producer + four-warp DP4A resadd.

  This is the M2a/M2b-compatible successor to the stale owned-boundary route.
  It replaces the promoted fused16 W1/W3 producer and the installed Q4-down
  resadd consumer with one packed-Q8 producer and the direct Q4/Q8 consumer,
  staying net-zero programs. Every shape/lease mismatch returns ``None`` so the
  ordinary block path is unchanged.
  """
  if not isinstance(admission, Q4KFFNDownMMVQAdmission) or not admission.scalar_q8_packet: return None
  capability = getattr(getattr(linear, "route_admission", None), "capability", None)
  if (getattr(capability, "backend", None), getattr(capability, "architecture", None)) != ("NV", "sm_120"): return None
  if (getattr(linear, "route_role", None), getattr(linear, "out_features", None), getattr(linear, "in_features", None)) != \
      ("ffn_down", ROWS, K): return None
  if residual is None: return None
  if getattr(linear, "bias", None) is not None or getattr(gate, "bias", None) is not None or getattr(up, "bias", None) is not None: return None
  if not str(x.device).startswith("NV"): return None
  if (getattr(gate, "out_features", None), getattr(gate, "in_features", None)) != (K, ROWS): return None
  if (getattr(up, "out_features", None), getattr(up, "in_features", None)) != (K, ROWS): return None
  if not all(hasattr(obj, "q4k_storage") for obj in (gate, up, linear)): return None

  def words(obj:Any) -> Tensor:
    storage = obj.q4k_storage
    return storage.words.to(x.device).contiguous() if storage.mode == "q4_ondemand" else storage.words.to(x.device)

  xv = x[:, 0, :].reshape(ROWS).cast(dtypes.float16).contiguous()
  residual_flat = residual[:, 0, :].reshape(ROWS).cast(dtypes.float32)
  producer = KernelProgram("decode_q4k_ffn_down_mmvq", f"blk{admission.block_index}.scalar_packet",
    KernelProgramProvenance.RESEARCH_ONLY, emit_ffn_w1w3_q8_scalar_packet(),
    output_spec=OutputSpec((Q8_WORDS,), dtypes.uint32))
  packed = execute_research_program(Tensor.empty((Q8_WORDS,), dtype=dtypes.uint32, device=x.device),
    words(gate), words(up), xv, program=producer)
  consumer = KernelProgram("decode_q4k_ffn_down_mmvq", f"blk{admission.block_index}.consumer",
    KernelProgramProvenance.RESEARCH_ONLY, emit_four_warp_direct(UOp.const(dtypes.weakint, BLOCKS_PER_WARP), resadd=True),
    output_spec=OutputSpec((ROWS,), dtypes.float32,
      typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32, (ROWS,), (1, 1, ROWS)),
        combine_fusion_admitted=False, epilogue_absorption_admitted=True)),
    residual_input_views=(ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(ROWS,),
      route_role="ffn_down", kind="residual_add"),))
  out = execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=x.device),
    words(linear), packed, residual_flat, program=consumer)
  return out.reshape(1, 1, ROWS)


__all__=["Q4KFFNDownMMVQAdmission","ROWS","K","Q4_BLOCKS","BLOCKS_PER_WARP","SUB_BLOCKS","Q8_PAYLOAD_WORDS",
         "Q8_GROUPS","Q8_WORDS","ownership_coordinates","owned_boundary_topology","emit_q8_provider",
         "emit_ffn_w1w3_q8_scalar_packet","emit_four_warp_direct","emit_four_warp_fp16_direct",
         "q4k_ffn_down_mmvq_call","q4k_ffn_down_mmvq_scalar_packet_call"]
