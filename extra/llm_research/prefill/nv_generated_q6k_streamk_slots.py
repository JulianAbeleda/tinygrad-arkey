"""Executable descriptor-driven Q6_K Stream-K partial producer."""
from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, RuntimeLocalAllocation, UOp
from extra.llm_research.prefill.nv_native_fragment_k16_gate import native_fragment_x2, packed_i8_sub

def q6_streamk_slot_kernel(partials, tile_ids, descriptors, blocks, b, dB,
                           total_k_blocks=48, slots=340, max_segment_blocks=37):
  """Produce one row-major 128x128 partial per ``(tile, begin, end)`` descriptor."""
  slot=UOp.special(slots,"gidx0"); lid=UOp.special(256,"lidx0")
  warp,lane=lid//32,lid%32; lr,lc=lane>>2,lane&3; band,phase=warp>>1,warp&1
  tile,lo,hi=descriptors[slot*3],descriptors[slot*3+1],descriptors[slot*3+2]
  valid=(tile>=0)&(hi>lo); safe_tile=valid.where(tile,UOp.const(dtypes.int32,0)); mt,nt=safe_tile//32,safe_tile%32
  blk=UOp.range(max_segment_blocks,0,axis_type=AxisType.REDUCE)
  active=valid & (blk < (hi-lo)); abs_blk=active.where(lo+blk,UOp.const(dtypes.int32,0))
  # llama.cpp mmq.cuh oracle: MMQ_ITER_K=256, consumed as two K=128 phases.
  q8_base=128*76
  arena=UOp.placeholder((q8_base+128*36,),dtypes.uint32,500,addrspace=AddrSpace.LOCAL).replace(
    tag=RuntimeLocalAllocation((q8_base+128*36)*dtypes.uint32.itemsize))
  sh,shq=arena,arena[q8_base:]
  sr=UOp.range(16,1,axis_type=AxisType.LOOP); srow=warp+8*sr; grow=nt*128+srow; hbase=(grow*total_k_blocks+abs_blk)*105; txi=lane
  ql=blocks[hbase+2*txi].cast(dtypes.uint32).bitwise_or(blocks[hbase+2*txi+1].cast(dtypes.uint32).lshift(16))
  qhi=(txi//16)*8+txi%8; qh=blocks[hbase+64+2*qhi].cast(dtypes.uint32).bitwise_or(blocks[hbase+64+2*qhi+1].cast(dtypes.uint32).lshift(16))
  qshift=txi.bitwise_and(8)>>2
  q0=ql.bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).lshift(4).bitwise_and(0x30303030))
  q1=ql.rshift(4).bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).bitwise_and(0x30303030)); kq0=2*txi-txi%16
  staged=UOp.group(sh[srow*76+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
    sh[srow*76+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020))),
    sh[srow*76+64].store(blocks[hbase+104].cast(dtypes.uint32),gate=(lane<1)),
    *(sh[srow*76+65+i].store(blocks[hbase+96+2*i].cast(dtypes.uint32).bitwise_or(
      blocks[hbase+97+2*i].cast(dtypes.uint32).lshift(16)),gate=((lane>=i)&(lane<i+1))) for i in range(4))).end(sr)
  ready_q6=UOp.barrier(UOp.group(staged))
  mcols=512; qcol=warp*16+(lane&15); warp_phase=phase
  acc=[UOp.placeholder((1,),dtypes.float32,700+i,addrspace=AddrSpace.REG) for i in range(64)]
  init=UOp.group(*(acc[i][0].store(0.0) for i in range(64))); acc=[x.after(init) for x in acc]; update=None
  for kphase in range(2):
   phase_base=abs_blk*256+kphase*128
   phase_gate=UOp.barrier(UOp.group(update)) if update is not None else None
   ydst=shq.after(phase_gate) if phase_gate is not None else shq
   qwords=tuple(ydst[qcol*36+4+i].store(
     b[(phase_base+i*4+0)*mcols+mt*128+qcol].cast(dtypes.uint32).bitwise_and(255).bitwise_or(
       b[(phase_base+i*4+1)*mcols+mt*128+qcol].cast(dtypes.uint32).bitwise_and(255).lshift(8)).bitwise_or(
       b[(phase_base+i*4+2)*mcols+mt*128+qcol].cast(dtypes.uint32).bitwise_and(255).lshift(16)).bitwise_or(
       b[(phase_base+i*4+3)*mcols+mt*128+qcol].cast(dtypes.uint32).bitwise_and(255).lshift(24)),gate=(lane<16)) for i in range(32))
   qscales=tuple(ydst[qcol*36+i].store(
     dB[(abs_blk*8+kphase*4+i)*mcols+mt*128+qcol].bitcast(dtypes.uint32),gate=(lane<16)) for i in range(4))
   ready_y=UOp.barrier(UOp.group(*qwords,*qscales))
   def mma(cg,n,g,dep):
    ordered_sh=sh.after(ready_q6).after(dep) if dep is not None else sh.after(ready_q6)
    ordered_y=shq.after(ready_y).after(dep) if dep is not None else shq.after(ready_y)
    row0=band*32+n*16
    av=native_fragment_x2(ordered_sh,(row0+(lane&15))*76+(kphase*8+g)*4).bitcast(dtypes.char.vec(8))
    lcol=cg*16+warp_phase*8+lr; qv=ordered_y[lcol*36+4+g*4+lc]
    bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(qv.rshift(8*q).bitwise_and(255).cast(dtypes.char) for q in range(4)))
    axes=(tuple((600+i,2) for i in range(3)),tuple((610+i,2) for i in range(2)),tuple((620+i,2) for i in range(2)))
    return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),
      ("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,()))
   for cg in range(8):
    for n in range(2):
     cs=[mma(cg,n,g,update) for g in range(8)]
     for r in range(4):
      ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+warp_phase*8+2*lc+(r&1)
      shared=sh.after(ready_q6).after(update) if update is not None else sh.after(ready_q6)
      shared_y=shq.after(ready_y).after(update) if update is not None else shq.after(ready_y)
      wd=shared[lrow*76+64].bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
      term=UOp.const(dtypes.float32,0.0)
      for p in range(4):
       sp=kphase*4+p; sw=shared[lrow*76+65+sp//2]
       s0=sw.rshift((2*sp%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.float32)
       s1=sw.rshift(((2*sp+1)%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.float32)
       dot=s0*cs[2*p].gep(r).cast(dtypes.float32)+s1*cs[2*p+1].gep(r).cast(dtypes.float32)
       term=term+wd*shared_y[lcol*36+p].bitcast(dtypes.float32)*dot
      term=active.cast(dtypes.float32)*term
      carrier=acc[ai].after(blk if update is None else update)
      update=carrier[0].store(acc[ai].after(blk)[0]+term)
  done=update.end(blk); stores=[]
  for cg in range(8):
   for n in range(2):
    for r in range(4):
     ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+phase*8+2*lc+(r&1)
     stores.append(partials[slot*16384+lcol*128+lrow].store(acc[ai].after(done)[0]))
  stores.append(tile_ids[slot].store(valid.where(tile,UOp.const(dtypes.int32,-1))))
  return UOp.sink(*stores,arg=KernelInfo(name="nv_generated_q6k_streamk_slots",opts_to_apply=()))

__all__=["q6_streamk_slot_kernel"]
