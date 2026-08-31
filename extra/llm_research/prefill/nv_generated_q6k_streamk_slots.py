"""Executable descriptor-driven Q6_K Stream-K partial producer."""
from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
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
  sh=UOp.placeholder((128*76,),dtypes.uint32,500,addrspace=AddrSpace.LOCAL)
  sr=UOp.range(16,1,axis_type=AxisType.LOOP); srow=warp+8*sr; grow=nt*128+srow; hbase=(grow*total_k_blocks+abs_blk)*105; txi=lane
  ql=blocks[hbase+2*txi].cast(dtypes.uint32).bitwise_or(blocks[hbase+2*txi+1].cast(dtypes.uint32).lshift(16))
  qhi=(txi//16)*8+txi%8; qh=blocks[hbase+64+2*qhi].cast(dtypes.uint32).bitwise_or(blocks[hbase+64+2*qhi+1].cast(dtypes.uint32).lshift(16))
  qshift=txi.bitwise_and(8)>>2
  q0=ql.bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).lshift(4).bitwise_and(0x30303030))
  q1=ql.rshift(4).bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).bitwise_and(0x30303030)); kq0=2*txi-txi%16
  staged=UOp.group(sh[srow*76+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
    sh[srow*76+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020)))).end(sr)
  ready=UOp.barrier(UOp.group(staged)); mcols=512
  def mma(cg,n,g):
    row0=band*32+n*16; av=native_fragment_x2(sh.after(ready),(row0+(lane&15))*76+g*4).bitcast(dtypes.char.vec(8))
    lcol=cg*16+phase*8+lr
    bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(b[(abs_blk*256+g*16+4*lc+q)*mcols+mt*128+lcol] for q in range(4)))
    axes=(tuple((600+i,2) for i in range(3)),tuple((610+i,2) for i in range(2)),tuple((620+i,2) for i in range(2)))
    return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),
      ("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,()))
  cs=[[[mma(cg,n,g) for g in range(16)] for n in range(2)] for cg in range(8)]
  acc=UOp.placeholder((64,),dtypes.float32,700,addrspace=AddrSpace.REG)
  init=UOp.group(*(acc[i].store(0.0) for i in range(64))); acc=acc.after(init); update=None
  def byte(off): return blocks[off//2].cast(dtypes.uint32).rshift((off%2)*8).bitwise_and(255)
  for cg in range(8):
   for n in range(2):
    for r in range(4):
     ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+phase*8+2*lc+(r&1)
     base=((nt*128+lrow)*total_k_blocks+abs_blk)*210; wd=blocks[(base+208)//2].bitcast(dtypes.half).cast(dtypes.float32); term=UOp.const(dtypes.float32,0.0)
     for p in range(8):
      s0=byte(base+192+2*p).cast(dtypes.char).cast(dtypes.float32); s1=byte(base+193+2*p).cast(dtypes.char).cast(dtypes.float32)
      dot=s0*cs[cg][n][2*p].gep(r).cast(dtypes.float32)+s1*cs[cg][n][2*p+1].gep(r).cast(dtypes.float32)
      term=term+(wd*dB[(abs_blk*8+p)*mcols+mt*128+lcol])*dot
     term=active.cast(dtypes.float32)*term
     update=acc.after(blk if update is None else update)[ai].store(acc.after(blk)[ai]+term)
  done=update.end(blk); stores=[]
  for cg in range(8):
   for n in range(2):
    for r in range(4):
     ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+phase*8+2*lc+(r&1)
     stores.append(partials[slot*16384+lcol*128+lrow].store(acc.after(done)[ai]))
  stores.append(tile_ids[slot].store(valid.where(tile,UOp.const(dtypes.int32,-1))))
  return UOp.sink(*stores,arg=KernelInfo(name="nv_generated_q6k_streamk_slots",opts_to_apply=()))

__all__=["q6_streamk_slot_kernel"]
