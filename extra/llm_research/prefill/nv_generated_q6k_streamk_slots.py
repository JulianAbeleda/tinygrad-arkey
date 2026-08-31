"""Executable descriptor-driven Q6_K Stream-K partial producer."""
from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, RuntimeLocalAllocation, UOp
from tinygrad.codegen.late.native_fragment import native_fragment_bitcast, native_fragment_materialized_x2
from extra.llm_research.prefill.nv_native_fragment_k16_gate import native_fragment_x2, packed_i8_sub

def q6_streamk_slot_kernel(partials, tile_ids, descriptors, blocks, b, dB,
                           total_k_blocks=48, slots=340, max_segment_blocks=37, tile_m=128):
  """Produce one row-major ``tile_m x 128`` partial per descriptor."""
  cg_count,slot_values=tile_m//16,tile_m*128
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
  mcols=512; qcol=warp*16+(lane&15); qcol_load=qcol%tile_m; warp_phase=phase
  acc=[UOp.placeholder((1,),dtypes.float32,700+i,addrspace=AddrSpace.REG) for i in range(cg_count*8)]
  init=UOp.group(*(acc[i][0].store(0.0) for i in range(cg_count*8))); acc=[x.after(init) for x in acc]; update=None
  for kphase in range(2):
   phase_base=abs_blk*256+kphase*128
   phase_gate=UOp.barrier(UOp.group(update)) if update is not None else None
   ydst=shq.after(phase_gate) if phase_gate is not None else shq
   qwords=tuple(ydst[qcol_load*36+4+i].store(
     b[(phase_base+i*4+0)*mcols+mt*tile_m+qcol_load].cast(dtypes.uint32).bitwise_and(255).bitwise_or(
       b[(phase_base+i*4+1)*mcols+mt*tile_m+qcol_load].cast(dtypes.uint32).bitwise_and(255).lshift(8)).bitwise_or(
       b[(phase_base+i*4+2)*mcols+mt*tile_m+qcol_load].cast(dtypes.uint32).bitwise_and(255).lshift(16)).bitwise_or(
       b[(phase_base+i*4+3)*mcols+mt*tile_m+qcol_load].cast(dtypes.uint32).bitwise_and(255).lshift(24)),gate=(lane<16)) for i in range(32))
   qscales=tuple(ydst[qcol_load*36+i].store(
     dB[(abs_blk*8+kphase*4+i)*mcols+mt*tile_m+qcol_load].bitcast(dtypes.uint32),gate=(lane<16)) for i in range(4))
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
   for cg in range(cg_count):
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
  for cg in range(cg_count):
   for n in range(2):
    for r in range(4):
     ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+phase*8+2*lc+(r&1)
     stores.append(partials[slot*slot_values+lcol*128+lrow].store(acc[ai].after(done)[0]))
  stores.append(tile_ids[slot].store(valid.where(tile,UOp.const(dtypes.int32,-1))))
  return UOp.sink(*stores,arg=KernelInfo(name="nv_generated_q6k_streamk_slots",opts_to_apply=()))

def q6_streamk_owner_kernel(partials,tile_ids,blocks,q8_record,total_k_blocks=48,owners=170,
                            kernel_name="nv_generated_q6k_streamk_owner"):
  """One-bank 170-owner Q6_K Stream-K main for M512,N4096,K12288."""
  owner=UOp.special(owners,"gidx0"); lid=UOp.special(256,"lidx0")
  warp,lane=lid//32,lid%32; lr,lc=lane>>2,lane&3; band,phase=warp>>1,warp&1
  total=UOp.const(dtypes.int32,128*total_k_blocks); lo=(owner*total)//owners; hi=((owner+1)*total)//owners
  blk=UOp.range(hi-lo,50,axis_type=AxisType.REDUCE); linear=lo+blk; tile=linear//total_k_blocks; abs_blk=linear%total_k_blocks
  previous=(linear-1)//total_k_blocks; transition=(blk>0)&(tile!=previous); mt,nt=tile//32,tile%32
  q8_base=128*76; arena=UOp.placeholder((q8_base+128*36,),dtypes.uint32,550,addrspace=AddrSpace.LOCAL).replace(
    tag=RuntimeLocalAllocation((q8_base+128*36)*dtypes.uint32.itemsize))
  sh,shq=arena,arena[q8_base:]
  sr=UOp.range(16,51,axis_type=AxisType.LOOP); srow=warp+8*sr; grow=nt*128+srow; hbase=(grow*total_k_blocks+abs_blk)*105; txi=lane
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
  ready_q6=UOp.barrier(UOp.group(staged)); qcol=warp*16+(lane&15); warp_phase=phase
  acc=[UOp.placeholder((1,),dtypes.float32,800+i,addrspace=AddrSpace.REG) for i in range(64)]
  init=UOp.group(*(acc[i][0].store(0.0) for i in range(64))); acc=[x.after(init) for x in acc]; update=None
  for kphase in range(2):
   boundary=transition if kphase==0 else UOp.const(dtypes.bool,False)
   phase_base=abs_blk*256+kphase*128; phase_gate=UOp.barrier(UOp.group(update)) if update is not None else None
   ydst=shq.after(phase_gate) if phase_gate is not None else shq
   # Canonical block_q8_1_mmq D4 input is already the exact shared layout.
   # Cooperatively copy its 128*36 words linearly: each warp reads contiguous
   # words instead of half-warps reading row records at a 144-byte stride.
   record_base=((abs_blk*2+kphase)*512+mt*128)*36
   qwords=tuple(ydst[lid+i*256].store(q8_record[record_base+lid+i*256]) for i in range(18))
   ready_y=UOp.barrier(UOp.group(*qwords))
   av_cache={}; bv_cache={}
   def mma(cg,n,g,dep):
    if (n,g) not in av_cache:
     # Fragment residency is established by the phase barrier. Do not attach
     # the per-consumer accumulator dependency: it creates a distinct UOp for
     # every WMMA consumer and defeats carrier CSE.
     sx=sh.after(ready_q6)
     av_cache[n,g]=native_fragment_bitcast(native_fragment_materialized_x2(sx,(band*32+n*16+(lane&15))*76+(kphase*8+g)*4), dtypes.char.vec(8))
    av=av_cache[n,g]
    if (cg,g) not in bv_cache:
     sy=shq.after(ready_y).after(dep) if dep is not None else shq.after(ready_y); lcol=cg*16+warp_phase*8+lr; qv=sy[lcol*36+4+g*4+lc]
     bv_cache[cg,g]=UOp(Ops.STACK,dtypes.char.vec(4),tuple(qv.rshift(8*q).bitwise_and(255).cast(dtypes.char) for q in range(4)))
    av=av_cache[n,g]; bv=bv_cache[cg,g]
    axes=(tuple((900+i,2) for i in range(3)),tuple((910+i,2) for i in range(2)),tuple((920+i,2) for i in range(2)))
    return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,()))
   for cg in range(8):
    for n in range(2):
     # Consume one Q6 scale pair at a time.  Holding all eight int4 WMMA
     # results live while updating four FP32 outputs costs 32 integer lanes;
     # pairwise folding needs eight integer lanes plus four running FP32 terms.
     # The p order is unchanged, so each output keeps the exact FP32 recurrence.
     terms=[UOp.const(dtypes.float32,0.0) for _ in range(4)]
     for p in range(4):
      cs0,cs1=mma(cg,n,2*p,update),mma(cg,n,2*p+1,update)
      for r in range(4):
       lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+warp_phase*8+2*lc+(r&1)
       sx=sh.after(ready_q6).after(update) if update is not None else sh.after(ready_q6)
       sy=shq.after(ready_y).after(update) if update is not None else shq.after(ready_y)
       sp=kphase*4+p; sw=sx[lrow*76+65+sp//2]
       s0=sw.rshift((2*sp%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.float32)
       s1=sw.rshift(((2*sp+1)%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.float32)
       yscale=sy[lcol*36+p].bitcast(dtypes.float32)
       terms[r]=terms[r]+yscale*(s0*cs0.gep(r).cast(dtypes.float32)+s1*cs1.gep(r).cast(dtypes.float32))
     for r in range(4):
      ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+warp_phase*8+2*lc+(r&1)
      sx=sh.after(ready_q6).after(update) if update is not None else sh.after(ready_q6)
      wd=sx[lrow*76+64].bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
      term=wd*terms[r]
      carrier=acc[ai].after(blk if update is None else update); outidx=owner*2*16384+lrow*128+lcol
      flush=partials[outidx].store(carrier[0],gate=boundary)
      update=carrier.after(flush)[0].store(boundary.where(term,carrier[0]+term))
  done=update.end(blk); crossed=(lo//total_k_blocks)!=((hi-1)//total_k_blocks); stores=[]
  for cg in range(8):
   for n in range(2):
    for r in range(4):
     ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+phase*8+2*lc+(r&1)
     stores.append(partials[(owner*2+crossed.cast(dtypes.int32))*16384+lrow*128+lcol].store(acc[ai].after(done)[0]))
  stores.extend((tile_ids[owner*2].store(lo//total_k_blocks),tile_ids[owner*2+1].store(crossed.where((hi-1)//total_k_blocks,-1))))
  return UOp.sink(*stores,arg=KernelInfo(name=kernel_name,opts_to_apply=()))

def q6_streamk_owner_segmented_kernel(partials,tile_ids,blocks,q8_record,total_k_blocks=48,owners=170,
                                      kernel_name="nv_generated_q6k_streamk_owner_segmented",tile_m=128,max_segments=2):
  """Sequential owner segments sharing one accumulator bank.

  This mirrors llama's control boundary: writeback/reset occurs outside each K
  loop, so the hot body contains no transition predicate or partial stores.
  """
  if tile_m < 16 or 512%tile_m or tile_m%16: raise ValueError("tile_m must divide 512 in 16-column groups")
  if max_segments < 1: raise ValueError("max_segments must be positive")
  owner=UOp.special(owners,"gidx0"); lid=UOp.special(256,"lidx0")
  warp,lane=lid//32,lid%32; lr,lc=lane>>2,lane&3; band,phase=warp>>1,warp&1
  cg_count,tiles,slot_values=tile_m//16,(512//tile_m)*32,tile_m*128
  total=UOp.const(dtypes.int32,tiles*total_k_blocks); lo=(owner*total)//owners; hi=((owner+1)*total)//owners
  tile0=lo//total_k_blocks
  q8_base=128*76; arena=UOp.placeholder((q8_base+128*36,),dtypes.uint32,1050,addrspace=AddrSpace.LOCAL).replace(
    tag=RuntimeLocalAllocation((q8_base+128*36)*dtypes.uint32.itemsize))
  sh,shq=arena,arena[q8_base:]
  acc=[UOp.placeholder((1,),dtypes.float32,1060+i,addrspace=AddrSpace.REG) for i in range(cg_count*8)]
  init=UOp.group(*(acc[i][0].store(0.0) for i in range(cg_count*8)))

  def run_segment(seg_lo,seg_hi,tile,slot,axis,dep):
   blk=UOp.range(seg_hi-seg_lo,axis,axis_type=AxisType.REDUCE); abs_blk=(seg_lo+blk)%total_k_blocks; mt,nt=tile//32,tile%32
   segment_sh=sh.after(dep)
   sr=UOp.range(16,axis+10,axis_type=AxisType.LOOP); srow=warp+8*sr; grow=nt*128+srow; hbase=(grow*total_k_blocks+abs_blk)*105; txi=lane
   ql=blocks[hbase+2*txi].cast(dtypes.uint32).bitwise_or(blocks[hbase+2*txi+1].cast(dtypes.uint32).lshift(16))
   qhi=(txi//16)*8+txi%8; qh=blocks[hbase+64+2*qhi].cast(dtypes.uint32).bitwise_or(blocks[hbase+64+2*qhi+1].cast(dtypes.uint32).lshift(16))
   qshift=txi.bitwise_and(8)>>2
   q0=ql.bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).lshift(4).bitwise_and(0x30303030))
   q1=ql.rshift(4).bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).bitwise_and(0x30303030)); kq0=2*txi-txi%16
   staged=UOp.group(segment_sh[srow*76+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
     segment_sh[srow*76+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020))),
     segment_sh[srow*76+64].store(blocks[hbase+104].cast(dtypes.uint32),gate=(lane<1)),
     *(segment_sh[srow*76+65+i].store(blocks[hbase+96+2*i].cast(dtypes.uint32).bitwise_or(
       blocks[hbase+97+2*i].cast(dtypes.uint32).lshift(16)),gate=((lane>=i)&(lane<i+1))) for i in range(4))).end(sr)
   ready_q6=UOp.barrier(UOp.group(staged)); qcol=warp*16+(lane&15); warp_phase=phase
   update=None
   for kphase in range(2):
    phase_gate=UOp.barrier(UOp.group(update)) if update is not None else None
    ydst=shq.after(phase_gate) if phase_gate is not None else shq.after(dep)
    record_base=((abs_blk*2+kphase)*512+mt*tile_m)*36
    qword_count=tile_m*36
    qwords=[]
    for i in range((qword_count+255)//256):
     qidx=lid+i*256; qvalid=qidx<qword_count; safe_qidx=qvalid.where(qidx,lid*0)
     qwords.append(ydst[qidx].store(q8_record[record_base+safe_qidx],gate=qvalid))
    qwords=tuple(qwords)
    ready_y=UOp.barrier(UOp.group(*qwords)); av_cache={}; bv_cache={}
    def mma(cg,n,g,mdep):
     if (n,g) not in av_cache:
      av_cache[n,g]=native_fragment_bitcast(native_fragment_materialized_x2(sh.after(ready_q6),
        (band*32+n*16+(lane&15))*76+(kphase*8+g)*4),dtypes.char.vec(8))
     if (cg,g) not in bv_cache:
      sy=shq.after(ready_y).after(mdep) if mdep is not None else shq.after(ready_y)
      lcol=cg*16+warp_phase*8+lr; qv=sy[lcol*36+4+g*4+lc]
      bv_cache[cg,g]=UOp(Ops.STACK,dtypes.char.vec(4),tuple(qv.rshift(8*q).bitwise_and(255).cast(dtypes.char) for q in range(4)))
     axes=(tuple((1200+i,2) for i in range(3)),tuple((1210+i,2) for i in range(2)),tuple((1220+i,2) for i in range(2)))
     return UOp(Ops.WMMA,dtypes.int.vec(4),(av_cache[n,g],bv_cache[cg,g],UOp.const(dtypes.int.vec(4),0)),
       ("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,()))
    for cg in range(cg_count):
     for n in range(2):
      terms=[UOp.const(dtypes.float32,0.0) for _ in range(4)]
      for p in range(4):
       cs0,cs1=mma(cg,n,2*p,update),mma(cg,n,2*p+1,update)
       for r in range(4):
        lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+warp_phase*8+2*lc+(r&1)
        sx=sh.after(ready_q6).after(update) if update is not None else sh.after(ready_q6)
        sy=shq.after(ready_y).after(update) if update is not None else shq.after(ready_y)
        sp=kphase*4+p; sw=sx[lrow*76+65+sp//2]
        s0=sw.rshift((2*sp%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
        s1=sw.rshift(((2*sp+1)%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
        yscale=sy[lcol*36+p].bitcast(dtypes.float32)
        scaled_dot=(s0*cs0.gep(r)+s1*cs1.gep(r)).cast(dtypes.float32)
        terms[r]=terms[r]+yscale*scaled_dot
      for r in range(4):
       ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1)
       sx=sh.after(ready_q6).after(update) if update is not None else sh.after(ready_q6)
       wd=sx[lrow*76+64].bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
       carrier=acc[ai].after(blk,dep if update is None else update)
       update=carrier[0].store(carrier[0]+wd*terms[r])
   done=update.end(blk); stores=[]
   for cg in range(cg_count):
    for n in range(2):
     for r in range(4):
      ai=cg*8+n*4+r; lrow=band*32+n*16+lr+8*(r>>1); lcol=cg*16+phase*8+2*lc+(r&1)
      stores.append(partials[slot*slot_values+lrow*tile_m+lcol].store(acc[ai].after(done)[0]))
   return UOp.group(*stores)

  segments=[]; ids=[]; dep=init
  for segment in range(max_segments):
   tile=tile0+segment; raw_lo=lo if segment == 0 else tile*total_k_blocks
   seg_lo=(raw_lo<hi).where(raw_lo,hi); boundary=(tile+1)*total_k_blocks
   seg_hi=(hi<boundary).where(hi,boundary); seg_hi=(seg_lo<seg_hi).where(seg_hi,seg_lo); valid=seg_lo<seg_hi
   part=run_segment(seg_lo,seg_hi,tile,owner*max_segments+segment,61+segment,dep)
   segments.append(part); ids.append(tile_ids[owner*max_segments+segment].store(valid.where(tile,UOp.const(dtypes.int32,-1))))
   if segment+1 < max_segments:
    dep=UOp.group(*(acc[i].after(part)[0].store(0.0) for i in range(cg_count*8)))
  return UOp.sink(*segments,*ids,arg=KernelInfo(name=kernel_name,opts_to_apply=()))

__all__=["q6_streamk_slot_kernel","q6_streamk_owner_kernel","q6_streamk_owner_segmented_kernel"]
