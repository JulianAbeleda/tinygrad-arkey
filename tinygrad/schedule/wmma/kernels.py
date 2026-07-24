"""Fixed-geometry hand kernels, assembled from the fragment/softmax primitives."""
from __future__ import annotations

import math
from tinygrad.dtype import dtypes, PtrDType, AddrSpace
from tinygrad.uop.ops import Ops, UOp, AMDRowSoftmaxRepackSpec
from tinygrad.schedule.wmma.softmax import amd_gfx1100_row_softmax_initial, amd_gfx1100_row_softmax_state
from tinygrad.schedule.wmma.loop_state import loop_state_write, loop_state_read, packed_fragment_load

def amd_gfx1100_q16_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info,
                             causal:bool=False, valid_kv:int=16, query_start:int=0) -> UOp:
  """Build the exact live-owner q16 native attention kernel graph."""
  owners = (q, k, v, out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype, PtrDType) or x.ptrdtype.size != 256 for x in owners):
    raise ValueError("q16 native attention requires four live 256-element PARAM owners")
  if tuple(x.ptrdtype.base for x in owners) != (dtypes.half,)*4:
    raise ValueError("q16 native attention requires fp16 Q/K/V/output owners")
  if tuple(x.arg.slot for x in owners) != (1, 2, 3, 0):
    raise ValueError("q16 native attention requires PARAM slots Q=1 K=2 V=3 output=0")
  if not isinstance(scale, float) or not math.isfinite(scale) or scale <= 0:
    raise ValueError("q16 native attention requires one positive finite score scale")
  if not isinstance(causal,bool) or not isinstance(valid_kv,int) or not 0 <= valid_kv <= 16 or not isinstance(query_start,int):
    raise ValueError("q16 native attention requires typed causal/KV validity metadata")
  lane = UOp.special(32, "lidx0")
  col, halfwave = lane & 15, lane >> 4
  qfrag = UOp(Ops.STACK, dtypes.half.vec(16), tuple(q.index(col*16+i).load() for i in range(16)),
    tag=("amd_gfx1100_fragment_load_v1","Q",0,q,lane,col))
  kfrag = UOp(Ops.STACK, dtypes.half.vec(16), tuple(k.index(col*16+i).load() for i in range(16)),
    tag=("amd_gfx1100_fragment_load_v1","K",0,k,lane,col))
  zero = UOp.const(dtypes.float.vec(8), (0.0,)*8)
  # A/B are already physical half16 fragments and must not be permuted by
  # logical upcast-axis rewriting. C retains its exact three binary axes so
  # eight native accumulator lanes can be projected with GEP.
  fragment_axes = ((), (), tuple((-120-i, 2) for i in range(3)))
  warg = ("WMMA_16_16_16_half_float", (16,16,16), dtypes.half, dtypes.float, "AMD:gfx1100", 32, fragment_axes, ())
  qk = UOp(Ops.WMMA, dtypes.float.vec(8), (qfrag, kfrag, zero), warg)
  weights,sm,sl,_ = amd_gfx1100_row_softmax_initial(qk, spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),
    mode="initial_state_v1",validity_mode="causal_v1" if causal else "all_v1",query_start=query_start,
    kv_start=0,valid_kv=valid_kv))
  vfrag = UOp(Ops.STACK, dtypes.half.vec(16), tuple(v.index(i*16+col).load() for i in range(16)),
    tag=("amd_gfx1100_fragment_load_v1","V",0,v,lane,col))
  pv = UOp(Ops.WMMA, dtypes.float.vec(8), (weights, vfrag, zero), warg)
  stores=[]
  for e in range(8):
    value=pv.gep(e)
    if stores: value=value.bitcast(dtypes.uint).after(UOp.group(stores[-1])).bitcast(dtypes.float)
    den=sl.gep(e); recip=den.ne(UOp.const(dtypes.float,0)).where(UOp.const(dtypes.float,1)/den,UOp.const(dtypes.float,0))
    stores.append(out.index((UOp.const(dtypes.weakint,2*e)+halfwave)*16+col).store((value*recip).cast(dtypes.half)))
  return UOp.sink(*stores, arg=kernel_info)

def amd_gfx1100_q16_kv32_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info) -> UOp:
  owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q16-kv32 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(256,512,512,256):
    raise ValueError("q16-kv32 requires Q1/K2/V3/out0 sized 256/512/512/256")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q16-kv32 requires fp16 owners and positive finite scale")
  lane=UOp.special(32,"lidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); half=lane.alu(Ops.SHR,UOp.const(dtypes.weakint,4))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  sm=UOp.const(dtypes.float.vec(8),(-float("inf"),)*8); sl=UOp.const(dtypes.float.vec(8),(0.0,)*8); acc=zero
  for tile in range(2):
    base=UOp.const(dtypes.weakint,tile*256)
    q_owner,k_owner,v_owner=(q,k,v) if tile == 0 else (q.after(acc),k.after(acc),v.after(acc))
    qf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(q_owner.index(col*16+i).load() for i in range(16)),
      tag=("amd_gfx1100_fragment_load_v1","Q",tile,q,lane,col))
    kf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(k_owner.index(base+col*16+i).load() for i in range(16)),
      tag=("amd_gfx1100_fragment_load_v1","K",tile,k,lane,col))
    qk=UOp(Ops.WMMA,dtypes.float.vec(8),(qf,kf,zero),warg)
    if tile == 0:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_initial(qk,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="initial_state_v1"))
    else:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_state(qk,sm,sl,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="stateful_unnormalized_v1"))
    corrected=zero
    if tile:
      corrected=acc.alu(Ops.MUL,alpha)
    v_ready=v_owner.after(corrected if tile else p)
    vf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(v_ready.index(base+i*16+col).load() for i in range(16)),
      tag=("amd_gfx1100_fragment_load_v1","V",tile,v,lane,col))
    acc=UOp(Ops.WMMA,dtypes.float.vec(8),(p,vf,corrected),warg)
  stores=[]
  for e in range(8):
    value=acc.gep(e)
    if stores: value=value.bitcast(dtypes.uint).after(UOp.group(stores[-1])).bitcast(dtypes.float)
    den=sl.gep(e)
    recip=den.ne(UOp.const(dtypes.float,0)).where(UOp.const(dtypes.float,1)/den,UOp.const(dtypes.float,0))
    dst=out.index((UOp.const(dtypes.weakint,2*e)+half)*16+col)
    stores.append(dst.store(value.alu(Ops.MUL,recip).cast(dtypes.half)))
  return UOp.sink(*stores,arg=kernel_info)

def amd_gfx1100_q16_kv32_hd128_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info) -> UOp:
  """Exact B=H=1, Q=16, KV=32, Hd=128 online-softmax kernel graph."""
  owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q16-kv32-hd128 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(2048,4096,4096,2048):
    raise ValueError("q16-kv32-hd128 requires Q1/K2/V3/out0 sized 2048/4096/4096/2048")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q16-kv32-hd128 requires fp16 owners and positive finite scale")
  lane=UOp.special(32,"lidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); half=lane.alu(Ops.SHR,UOp.const(dtypes.weakint,4))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  sm=UOp.const(dtypes.float.vec(8),(-float("inf"),)*8); sl=UOp.const(dtypes.float.vec(8),(0.0,)*8)
  acc=[zero]*8
  for tile in range(2):
    qk=zero
    for hd_block in range(8):
      q_owner=q if tile == 0 else q.after(*acc); k_owner=k if tile == 0 else k.after(*acc)
      qf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(q_owner.index(col*128+hd_block*16+i).load() for i in range(16)),
        tag=("amd_gfx1100_fragment_load_hd128_v1","Q",tile,hd_block,q,lane,col))
      kbase=UOp.const(dtypes.weakint,tile*2048+hd_block*16)
      kf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(k_owner.index(kbase+col*128+i).load() for i in range(16)),
        tag=("amd_gfx1100_fragment_load_hd128_v1","K",tile,hd_block,k,lane,col))
      qk=UOp(Ops.WMMA,dtypes.float.vec(8),(qf,kf,qk),warg)
    if tile == 0:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_initial(qk,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="initial_state_v1"))
    else:
      p,sm,sl,alpha=amd_gfx1100_row_softmax_state(qk,sm,sl,
        spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="stateful_unnormalized_v1",kv_start=16,
          validity_mode="causal_v1",query_start=16,valid_kv=32))
    next_acc=[]
    for hd_block in range(8):
      corrected=zero if tile == 0 else acc[hd_block].alu(Ops.MUL,alpha)
      v_owner=v.after(corrected if tile else p); vbase=UOp.const(dtypes.weakint,tile*2048+hd_block*16)
      vf=UOp(Ops.STACK,dtypes.half.vec(16),tuple(v_owner.index(vbase+i*128+col).load() for i in range(16)),
        tag=("amd_gfx1100_fragment_load_hd128_v1","V",tile,hd_block,v,lane,col))
      next_acc.append(UOp(Ops.WMMA,dtypes.float.vec(8),(p,vf,corrected),warg))
    acc=next_acc
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec
  drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,sl,*acc),arg=AMDAttentionOutputDrainSpec())
  return UOp.sink(drain,arg=kernel_info)

def amd_gfx1100_q16_kv64_hd128_loop_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info,
                                               causal:bool=False, valid_kv:int=64, query_start:int|None=None) -> UOp:
  """Scheduler-only Q16/KV64/Hd128 recurrence with one runtime KV tile body.

  This deliberately has no AMD/HIP lowering yet.  The typed state and dynamic
  fragment carriers prevent it from silently falling back to the static KV32
  implementation while retaining the exact graph that a future backend must
  consume.
  """
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q16-kv64-hd128 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(2048,8192,8192,2048):
    raise ValueError("q16-kv64-hd128 requires Q1/K2/V3/out0 sized 2048/8192/8192/2048")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q16-kv64-hd128 requires fp16 owners and positive finite scale")
  if not isinstance(valid_kv,int) or isinstance(valid_kv,bool) or not 0 <= valid_kv <= 64: raise ValueError("valid_kv must be in [0,64]")
  if query_start is None: query_start=valid_kv-16
  if not isinstance(query_start,int) or isinstance(query_start,bool): raise ValueError("query_start must be integral")
  lane=UOp.special(32,"lidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  rng=UOp.range(4,9400,AxisType.REDUCE)
  mreg=UOp.placeholder((8,),dtypes.float,9401,addrspace=AddrSpace.REG)
  lreg=UOp.placeholder((8,),dtypes.float,9402,addrspace=AddrSpace.REG)
  creg=UOp.placeholder((64,),dtypes.float,9403,addrspace=AddrSpace.REG)
  state_owner=9404
  # NOTE: fragment() below is deliberately NOT centralized into
  # loop_state.packed_fragment_load -- this kernel has no grid/group source
  # (its AMD_PACKED_FRAGMENT_LOAD is a 4-tuple, not the shared 5-tuple), so
  # sharing it would change the emitted UOp's source arity.
  def state_write(reg, role, value, block=0, offset=0, access="write"):
    return loop_state_write(reg, value, role=role, owner=state_owner, offset=offset, block=block, access=access)
  m_init=UOp.group(*state_write(mreg,"m",UOp.const(dtypes.float.vec(8),(-float("inf"),)*8),access="init"))
  l_init=UOp.group(*state_write(lreg,"l",zero,access="init"))
  c_init=UOp.group(*(x for block in range(8) for x in state_write(creg,"acc",zero,block,block*8,access="init")))
  def state_read(reg, init, role, block=0, offset=0, final=False):
    return loop_state_read(reg, init, rng, role=role, owner=state_owner, block=block, final=final)
  def fragment(owner, role, block):
    return UOp(Ops.AMD_PACKED_FRAGMENT_LOAD,dtypes.half.vec(16),(owner,lane,col,rng),arg=AMDPackedFragmentLoopSpec(role=role,head_block=block))
  old_m,old_l=state_read(mreg,m_init,"m"),state_read(lreg,l_init,"l")
  qk=zero
  for block in range(8): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fragment(q,"Q",block),fragment(k,"K",block),qk),warg)
  p,new_m,new_l,alpha=amd_gfx1100_row_softmax_state(qk,old_m,old_l,
    spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="loop_state_v1",validity_mode="causal_v1" if causal else "tail_v1",
      query_start=query_start,kv_start=-1,valid_kv=valid_kv,dynamic_kv_v1=True),kv_tile=rng)
  writes=[*state_write(mreg,"m",new_m),*state_write(lreg,"l",new_l)]
  for block in range(8):
    old_c=state_read(creg,c_init,"acc",block,block*8)
    corrected=old_c.alu(Ops.MUL,alpha)
    pv=UOp(Ops.WMMA,dtypes.float.vec(8),(p,fragment(v,"V",block),corrected),warg)
    writes.extend(state_write(creg,"acc",pv,block,block*8))
  end=UOp.group(*writes).end(rng).replace(tag=("amd_gfx1100_attention_kv64_loop_end_v1",rng))
  final_l=state_read(lreg,end,"l",final=True)
  final_c=tuple(state_read(creg,end,"acc",block,block*8,final=True) for block in range(8))
  drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,final_l,*final_c),arg=AMDAttentionOutputDrainSpec())
  return UOp.sink(m_init,l_init,c_init,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_q16_kv64_hd128_loop_v1",))

def amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention(q:UOp, k:UOp, v:UOp, out:UOp, *, scale:float, kernel_info,
                                                          causal:bool=False, valid_kv:int=64, query_start:int|None=None) -> UOp:
  """Grid-native Q32/Hq4/Hkv2/G2 attention; one wave32 per Q-head tile."""
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec, AMDAttentionGridSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  grid=AMDAttentionGridSpec(); grid.validate(); owners=(q,k,v,out)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners): raise ValueError("q32-hq4-hkv2 requires PARAM owners")
  if tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=(16384,16384,16384,16384):
    raise ValueError("q32-hq4-hkv2 requires Q1/K2/V3/out0 sized 16384")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0:
    raise ValueError("q32-hq4-hkv2 requires fp16 owners and positive finite scale")
  if not isinstance(valid_kv,int) or isinstance(valid_kv,bool) or not 0 <= valid_kv <= 64: raise ValueError("valid_kv must be in [0,64]")
  if query_start is None: query_start=valid_kv-32
  if not isinstance(query_start,int) or isinstance(query_start,bool): raise ValueError("query_start must be integral")
  lane=UOp.special(32,"lidx0"); group=UOp.special(8,"gidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15))
  zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3)))
  warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  rng=UOp.range(4,9500,AxisType.REDUCE); mreg=UOp.placeholder((8,),dtypes.float,9501,addrspace=AddrSpace.REG)
  lreg=UOp.placeholder((8,),dtypes.float,9502,addrspace=AddrSpace.REG); creg=UOp.placeholder((64,),dtypes.float,9503,addrspace=AddrSpace.REG); state_owner=9504
  def state_write(reg,role,value,block=0,offset=0,access="write"):
    return loop_state_write(reg, value, role=role, owner=state_owner, offset=offset, block=block, access=access)
  m_init=UOp.group(*state_write(mreg,"m",UOp.const(dtypes.float.vec(8),(-float("inf"),)*8),access="init")); l_init=UOp.group(*state_write(lreg,"l",zero,access="init"))
  c_init=UOp.group(*(x for block in range(8) for x in state_write(creg,"acc",zero,block,block*8,access="init")))
  def state_read(reg,init,role,block=0,offset=0,final=False):
    return loop_state_read(reg, init, rng, role=role, owner=state_owner, block=block, final=final)
  def fragment(owner,role,block):
    return packed_fragment_load(owner, role=role, head_block=block, grid=grid, lane=lane, col=col, rng=rng, group=group)
  old_m,old_l=state_read(mreg,m_init,"m"),state_read(lreg,l_init,"l"); qk=zero
  for block in range(8): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fragment(q,"Q",block),fragment(k,"K",block),qk),warg)
  p,new_m,new_l,alpha=amd_gfx1100_row_softmax_state(qk,old_m,old_l,spec=AMDRowSoftmaxRepackSpec(score_scale=float(scale),mode="loop_state_v1",
    validity_mode="causal_v1" if causal else "tail_v1",query_start=query_start,kv_start=-1,valid_kv=valid_kv,dynamic_kv_v1=True,grid=grid),kv_tile=rng,grid_id=group)
  writes=[*state_write(mreg,"m",new_m),*state_write(lreg,"l",new_l)]
  for block in range(8):
    old_c=state_read(creg,c_init,"acc",block,block*8); pv=UOp(Ops.WMMA,dtypes.float.vec(8),(p,fragment(v,"V",block),old_c.alu(Ops.MUL,alpha)),warg)
    writes.extend(state_write(creg,"acc",pv,block,block*8))
  end=UOp.group(*writes).end(rng).replace(tag=("amd_gfx1100_attention_grid_kv64_loop_end_v1",rng)); final_l=state_read(lreg,end,"l",final=True)
  final_c=tuple(state_read(creg,end,"acc",block,block*8,final=True) for block in range(8))
  drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,group,final_l,*final_c),arg=AMDAttentionOutputDrainSpec(grid=grid))
  return UOp.sink(m_init,l_init,c_init,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_v1",))

def amd_gfx1100_q16_grid_hd128_loop_attention(q:UOp,k:UOp,v:UOp,out:UOp,*,q_tokens:int,q_heads:int,kv_heads:int,kv_tokens:int,scale:float,kernel_info,causal:bool=False,valid_kv:int|None=None,query_start:int|None=None,output_block_base:int=0,acc_blocks:int=8,phase_abi_v1:bool=False)->UOp:
  """Fixed 16-WMMA attention wave with compile-time model geometry."""
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec, AMDAttentionGridSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  grid=AMDAttentionGridSpec(q_tokens=q_tokens,q_heads=q_heads,kv_heads=kv_heads,group_ratio=q_heads//kv_heads,kv_tokens=kv_tokens); grid.validate()
  hd=grid.head_dim; hd_blocks=hd//16
  owners=(q,k,v,out); sizes=(q_heads*q_tokens*hd,kv_heads*kv_tokens*hd,kv_heads*kv_tokens*hd,q_heads*q_tokens*hd)
  if any(x.op is not Ops.PARAM or not isinstance(x.dtype,PtrDType) for x in owners) or tuple(x.arg.slot for x in owners)!=(1,2,3,0) or tuple(x.ptrdtype.size for x in owners)!=sizes: raise ValueError(f"grid loop requires Q1/K2/V3/out0 sized {sizes}")
  if tuple(x.ptrdtype.base for x in owners)!=(dtypes.half,)*4 or not isinstance(scale,float) or not math.isfinite(scale) or scale<=0: raise ValueError("grid loop requires fp16 and finite scale")
  valid_kv=kv_tokens if valid_kv is None else valid_kv
  if not isinstance(valid_kv,int) or isinstance(valid_kv,bool) or not 0<=valid_kv<=kv_tokens: raise ValueError("valid_kv is outside KV geometry")
  if (output_block_base,acc_blocks) != (0,hd_blocks) and (acc_blocks not in {1,2,4} or not 0 <= output_block_base <= hd_blocks-acc_blocks or output_block_base % acc_blocks): raise ValueError("grid loop requires a full or aligned accumulator slice")
  if query_start is None: query_start=valid_kv-q_tokens
  lane=UOp.special(32,"lidx0"); group=UOp.special(q_heads*grid.q_tiles,"gidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); zero=UOp.const(dtypes.float.vec(8),(0.0,)*8); axes=((),(),tuple((-120-i,2) for i in range(3))); warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  rng=UOp.range((kv_tokens+15)//16,9600,AxisType.REDUCE); creg=UOp.placeholder((acc_blocks*8,),dtypes.float,9603,addrspace=AddrSpace.REG)
  if phase_abi_v1:
    from tinygrad.uop.ops import StateRegionSpec, PhaseBoundarySpec, StateHandle
    phase_lds=UOp(Ops.DEFINE_LOCAL,dtypes.float.ptr(512,AddrSpace.LOCAL),arg=9610)
    ml=StateHandle(StateRegionSpec("online_ml",dtypes.float,16),PhaseBoundarySpec("loop_init","loop_final"),0,phase_lds,lane,16)
  else:
    mreg=UOp.placeholder((8,),dtypes.float,9601,addrspace=AddrSpace.REG); lreg=UOp.placeholder((8,),dtypes.float,9602,addrspace=AddrSpace.REG)
  def wr(reg,role,value,b=0,o=0,a="write"): return loop_state_write(reg, value, role=role, owner=9604, offset=o, block=b, access=a)
  if phase_abi_v1:
    mi=UOp.group(*(ml.loop_write(UOp.const(dtypes.float,-float("inf")),i) for i in range(8)))
    li=UOp.group(*(ml.loop_write(UOp.const(dtypes.float,0.0),8+i) for i in range(8)))
    init_token=UOp.group(mi,li)
  else:
    mi=UOp.group(*wr(mreg,"m",UOp.const(dtypes.float.vec(8),(-float("inf"),)*8),a="init")); li=UOp.group(*wr(lreg,"l",zero,a="init"))
  ci=UOp.group(*(x for b in range(acc_blocks) for x in wr(creg,"acc",zero,b,b*8,"init")))
  def rd(reg,init,role,b=0,o=0,final=False): return loop_state_read(reg, init, rng, role=role, owner=9604, block=b, final=final)
  def fr(owner,role,b): return packed_fragment_load(owner, role=role, head_block=b, grid=grid, lane=lane, col=col, rng=rng, group=group)
  if not phase_abi_v1: om,ol=rd(mreg,mi,"m"),rd(lreg,li,"l")
  qk=zero
  for b in range(hd_blocks): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fr(q,"Q",b),fr(k,"K",b),qk),warg,tag=("attention_wmma","QK",b))
  if phase_abi_v1:
    om=UOp(Ops.STACK,dtypes.float.vec(8),tuple(ml.loop_read(i,init_token) for i in range(8)))
    ol=UOp(Ops.STACK,dtypes.float.vec(8),tuple(ml.loop_read(8+i,init_token) for i in range(8)))
  p,nm,nl,alpha=amd_gfx1100_row_softmax_state(qk,om,ol,spec=AMDRowSoftmaxRepackSpec(score_scale=scale,mode="loop_state_v1",validity_mode="causal_v1" if causal else "tail_v1",query_start=query_start,kv_start=-1,valid_kv=valid_kv,dynamic_kv_v1=True,grid=grid),kv_tile=rng,grid_id=group)
  # Phase ABI keeps m/l in LDS. Commit the next recurrence state before the
  # PV body and make that body consume the commit token: p/alpha are already
  # formed from the old state, so this preserves the recurrence while giving
  # regalloc an earlier end to the next-m/l lease. The default register-state
  # route deliberately retains its existing ordering.
  ml_commit=UOp.group(*( [*(ml.loop_write(nm.gep(i),i,after=nm.gep(i)) for i in range(8)),
                            *(ml.loop_write(nl.gep(i),8+i,after=nl.gep(i)) for i in range(8))] )) if phase_abi_v1 else None
  writes=[ml_commit] if ml_commit is not None else [*wr(mreg,"m",nm),*wr(lreg,"l",nl)]
  # AMD_ROW_SOFTMAX_SLOT is a verifier ABI value and cannot be wrapped in AFTER.
  # Gate V's PARAM owner, which makes the PV fragment load wait for the commit
  # while keeping p/alpha as direct slot values.
  pv_v=v.after(ml_commit) if ml_commit is not None else v
  for b in range(acc_blocks):
    oc=rd(creg,ci,"acc",b,b*8); pv=UOp(Ops.WMMA,dtypes.float.vec(8),(p,fr(pv_v,"V",b+output_block_base),oc.alu(Ops.MUL,alpha)),warg,tag=("attention_wmma","PV",b)); writes.extend(wr(creg,"acc",pv,b,b*8))
  end=UOp.group(*writes).end(rng).replace(tag=("amd_gfx1100_attention_grid_loop_end_v1",rng)); final_token=end if phase_abi_v1 else None; fl=(UOp(Ops.STACK,dtypes.float.vec(8),tuple(ml.loop_read(8+i,final_token) for i in range(8))) if phase_abi_v1 else rd(lreg,end,"l",final=True)); fc=tuple(rd(creg,end,"acc",b,b*8,final=True) for b in range(acc_blocks)); drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,group,fl,*fc),arg=AMDAttentionOutputDrainSpec(native_abi="amd_gfx1100_attention_output_drain_v1" if acc_blocks==hd_blocks else "amd_gfx1100_attention_output_drain_acc_slice_v2",blocks=acc_blocks,grid=grid,output_block_base=output_block_base))
  return UOp.sink(mi,li,ci,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_q16_grid_hd128_loop_v1",))


def amd_gfx1100_q16_grid_qk_stats_stage(q:UOp,k:UOp,stats:UOp,*,q_tokens:int,q_heads:int,kv_heads:int,kv_tokens:int,scale:float,kernel_info,causal:bool=True,query_start:int|None=None)->UOp:
  """Direct diagnostic Stage A: QK online reduction to fp32 `[query, m/l]` state only."""
  from tinygrad.uop.ops import AMDAttentionGridSpec, AMDAttentionStatsDrainSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  grid=AMDAttentionGridSpec(q_tokens=q_tokens,q_heads=q_heads,kv_heads=kv_heads,group_ratio=q_heads//kv_heads,kv_tokens=kv_tokens); grid.validate()
  hd=grid.head_dim; hd_blocks=hd//16
  qn,kn=q_heads*q_tokens*hd,kv_heads*kv_tokens*hd
  if tuple(x.arg.slot for x in (q,k,stats)) != (1,2,0) or q.ptrdtype.size != qn or k.ptrdtype.size != kn or stats.ptrdtype.size != q_heads*q_tokens*2:
    raise ValueError("qk stats stage requires Q1/K2/stats0 exact geometry")
  if q.ptrdtype.base != dtypes.half or k.ptrdtype.base != dtypes.half or stats.ptrdtype.base != dtypes.float or not causal: raise ValueError("qk stats stage requires causal fp16 Q/K and fp32 stats")
  query_start=kv_tokens-q_tokens if query_start is None else query_start
  lane=UOp.special(32,"lidx0"); group=UOp.special(q_heads*grid.q_tiles,"gidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); zero=UOp.const(dtypes.float.vec(8),(0.0,)*8)
  axes=((),(),tuple((-120-i,2) for i in range(3))); warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,())
  rng=UOp.range((kv_tokens+15)//16,9700,AxisType.REDUCE); mreg=UOp.placeholder((8,),dtypes.float,9701,addrspace=AddrSpace.REG); lreg=UOp.placeholder((8,),dtypes.float,9702,addrspace=AddrSpace.REG)
  def wr(reg,role,value,a="write"): return loop_state_write(reg, value, role=role, owner=9704, access=a)
  def rd(reg,init,role,final=False): return loop_state_read(reg, init, rng, role=role, owner=9704, final=final)
  def fr(owner,role,b): return packed_fragment_load(owner, role=role, head_block=b, grid=grid, lane=lane, col=col, rng=rng, group=group)
  mi=UOp.group(*wr(mreg,"m",UOp.const(dtypes.float.vec(8),(-float("inf"),)*8),"init")); li=UOp.group(*wr(lreg,"l",zero,"init")); om,ol=rd(mreg,mi,"m"),rd(lreg,li,"l"); qk=zero
  for b in range(hd_blocks): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fr(q,"Q",b),fr(k,"K",b),qk),warg,tag=("attention_wmma","QK",b))
  _,nm,nl,_=amd_gfx1100_row_softmax_state(qk,om,ol,spec=AMDRowSoftmaxRepackSpec(score_scale=scale,mode="loop_state_v1",validity_mode="causal_v1",query_start=query_start,kv_start=-1,valid_kv=kv_tokens,dynamic_kv_v1=True,grid=grid),kv_tile=rng,grid_id=group)
  end=UOp.group(*wr(mreg,"m",nm),*wr(lreg,"l",nl)).end(rng); fm,fl=rd(mreg,end,"m",True),rd(lreg,end,"l",True)
  drain=UOp(Ops.AMD_ATTENTION_STATS_DRAIN,dtypes.void,(stats,group,fm,fl),arg=AMDAttentionStatsDrainSpec())
  return UOp.sink(mi,li,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_qk_stats_stage_v1",))

def amd_gfx1100_q16_grid_pv_slice_stage(q:UOp,k:UOp,v:UOp,stats:UOp,out:UOp,*,q_tokens:int,q_heads:int,kv_heads:int,kv_tokens:int,scale:float,kernel_info,output_block_base:int,v_input_block_base:int=0,acc_blocks:int=2,causal:bool=True,query_start:int|None=None)->UOp:
  """Direct diagnostic Stage B: reload final m/l, recompute QK/P, and own one aligned PV slice."""
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec, AMDAttentionGridSpec, AMDLoopStateSpec, AMDPackedFragmentLoopSpec, AxisType
  grid=AMDAttentionGridSpec(q_tokens=q_tokens,q_heads=q_heads,kv_heads=kv_heads,group_ratio=q_heads//kv_heads,kv_tokens=kv_tokens); grid.validate()
  hd=grid.head_dim; hd_blocks=hd//16
  qn,kn,outn=q_heads*q_tokens*hd,kv_heads*kv_tokens*hd,q_heads*q_tokens*hd
  if tuple(x.arg.slot for x in (q,k,v,stats,out)) != (1,2,3,4,0) or tuple(x.ptrdtype.size for x in (q,k,v,stats,out)) != (qn,kn,kn-v_input_block_base*16,q_heads*q_tokens*2,outn): raise ValueError("pv slice stage requires Q1/K2/prebiased-V3/stats4/out0 exact geometry")
  if (output_block_base,acc_blocks) not in {(0,2),(2,2),(4,2),(6,2)} or v_input_block_base != output_block_base or not causal: raise ValueError("pv slice stage requires one aligned prebiased causal two-block slice")
  query_start=kv_tokens-q_tokens if query_start is None else query_start; lane=UOp.special(32,"lidx0"); group=UOp.special(q_heads*grid.q_tiles,"gidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); zero=UOp.const(dtypes.float.vec(8),(0.0,)*8)
  axes=((),(),tuple((-120-i,2) for i in range(3))); warg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,axes,()); rng=UOp.range((kv_tokens+15)//16,9800,AxisType.REDUCE); creg=UOp.placeholder((16,),dtypes.float,9803,addrspace=AddrSpace.REG)
  def wr(value,b): return loop_state_write(creg, value, role="acc", owner=9804, offset=b*8, block=b, access="write")
  def rd(init,b,final=False): return loop_state_read(creg, init, rng, role="acc", owner=9804, block=b, final=final)
  def fr(owner,role,b): return packed_fragment_load(owner, role=role, head_block=b, grid=grid, lane=lane, col=col, rng=rng, group=group)
  def stat(which): return UOp(Ops.STACK,dtypes.float.vec(8),tuple(stats.index(group*UOp.const(dtypes.weakint,32)+(UOp.const(dtypes.weakint,2*e)+lane.alu(Ops.SHR,UOp.const(dtypes.weakint,4)))*UOp.const(dtypes.weakint,2)+UOp.const(dtypes.weakint,which)).load() for e in range(8)))
  fm,fl=stat(0),stat(1); ci=UOp.group(*(x for b in range(2) for x in wr(zero,b))); qk=zero
  for b in range(hd_blocks): qk=UOp(Ops.WMMA,dtypes.float.vec(8),(fr(q,"Q",b),fr(k,"K",b),qk),warg,tag=("attention_wmma","QK",b))
  p,_,_,alpha=amd_gfx1100_row_softmax_state(qk,fm,fl,spec=AMDRowSoftmaxRepackSpec(score_scale=scale,mode="loop_state_v1",validity_mode="causal_v1",query_start=query_start,kv_start=-1,valid_kv=kv_tokens,dynamic_kv_v1=True,grid=grid),kv_tile=rng,grid_id=group); writes=[]
  for b in range(2): writes.extend(wr(UOp(Ops.WMMA,dtypes.float.vec(8),(p,fr(v,"V",b),rd(ci,b).alu(Ops.MUL,alpha)),warg,tag=("attention_wmma","PV",b)),b))
  end=UOp.group(*writes).end(rng); fc=tuple(rd(end,b,True) for b in range(2)); drain=UOp(Ops.AMD_ATTENTION_OUTPUT_DRAIN,dtypes.void,(out,group,fl,*fc),arg=AMDAttentionOutputDrainSpec(native_abi="amd_gfx1100_attention_output_drain_acc_slice_v2",blocks=2,grid=grid,output_block_base=output_block_base))
  return UOp.sink(ci,end,drain,arg=kernel_info).replace(tag=("amd_gfx1100_pv_slice_stage_v1",))
