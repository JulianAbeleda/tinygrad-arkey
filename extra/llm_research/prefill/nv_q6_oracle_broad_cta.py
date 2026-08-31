"""Broad 128x128xK256 generated Q6_K CTA matching the pinned llama schedule."""
from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, Ops, RuntimeLocalAllocation, UOp
from tinygrad.codegen.late.native_fragment import native_fragment_bitcast, native_fragment_materialized_x2
from extra.llm_research.prefill.nv_native_fragment_k16_gate import packed_i8_sub

ROWS, COLS, K = 128, 128, 256
Q6_STRIDE, Q8_STRIDE = 76, 36
Q6_WORDS, Q8_WORDS = ROWS*Q6_STRIDE, COLS*Q8_STRIDE
SHARED_BYTES = (Q6_WORDS+Q8_WORDS)*4


def q6_oracle_broad_cta_kernel(out, blocks, q8_record, *, replicas:int=1, prefetch_second_panel:bool=True,
                               combined_initial_publish:bool=False, factor_dA:bool=False,
                               oracle_publisher:bool=False, depth:int=1, streamk_owners:int|None=None,
                               streamk_segment:int=0):
  """One exact llama-normalized 128x128xK256 work unit per CTA.

  ``blocks`` is 128 canonical Q6_K rows. ``q8_record`` is two canonical
  128x36-word Q8_1 panels. The second panel can either be register-prefetched
  across the first consumer or loaded after it for a causal A/B.
  """
  if replicas < 1: raise ValueError("replicas must be positive")
  if depth < 1: raise ValueError("depth must be positive")
  if streamk_owners is not None and (not 1 <= streamk_owners <= 256 or streamk_segment not in (0,1)):
    raise ValueError("streamk requires 1..256 owners and segment zero or one")
  lid=UOp.special(256,"lidx0"); warp,lane=lid//32,lid%32
  lr,lc=lane>>2,lane&3; band,warp_phase=warp>>1,warp&1
  grid=streamk_owners if streamk_owners is not None else replicas
  bid=UOp.special(grid,"gidx0") if grid > 1 else UOp.const(dtypes.int32,0)
  arena=UOp.placeholder((Q6_WORDS+Q8_WORDS,),dtypes.uint32,1500,addrspace=AddrSpace.LOCAL).replace(
    tag=RuntimeLocalAllocation(SHARED_BYTES))
  sh,shq=arena,arena[Q6_WORDS:]
  if streamk_owners is None:
    epoch=UOp.range(depth,1499); block_row_stride=105
    block_epoch=epoch*(ROWS*105); q8_epoch=epoch*(2*Q8_WORDS); output_slot=bid
    active=UOp.const(dtypes.bool,True)
  else:
    work_units,tiles_m,k_blocks=128*48,4,48
    owner_start=bid*work_units//streamk_owners; owner_stop=(bid+1)*work_units//streamk_owners
    tile0=owner_start//k_blocks; boundary=(tile0+1)*k_blocks; first_stop=owner_stop.minimum(boundary)
    if streamk_segment == 0:
      tile=tile0; epoch_start=owner_start-tile0*k_blocks; segment_depth=first_stop-owner_start
      active=UOp.const(dtypes.bool,True)
    else:
      tile=tile0+1; epoch_start=UOp.const(dtypes.int32,0); segment_depth=owner_stop-first_stop
      active=owner_stop>boundary
    epoch=UOp.range(segment_depth,1499); tile_m=tile%tiles_m; tile_n=tile//tiles_m
    block_row_stride=k_blocks*105
    block_epoch=tile_n*ROWS*k_blocks*105+(epoch_start+epoch)*105
    q8_epoch=(tile_m*k_blocks+epoch_start+epoch)*(2*Q8_WORDS)
    output_slot=streamk_segment*streamk_owners+bid

  # Expand all 128 canonical Q6 rows into llama's 76-word shared layout.
  txi=lane
  if oracle_publisher:
    quant_stores=[]
    for sri in range(16):
      srow=warp+8*sri; hbase=block_epoch+srow*block_row_stride
      ql=blocks[hbase+2*txi].cast(dtypes.uint32).bitwise_or(blocks[hbase+2*txi+1].cast(dtypes.uint32).lshift(16))
      qhi=(txi//16)*8+txi%8
      qh=blocks[hbase+64+2*qhi].cast(dtypes.uint32).bitwise_or(blocks[hbase+64+2*qhi+1].cast(dtypes.uint32).lshift(16))
      qshift=txi.bitwise_and(8)>>2
      q0=ql.bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).lshift(4).bitwise_and(0x30303030))
      q1=ql.rshift(4).bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).bitwise_and(0x30303030))
      kq0=2*txi-txi%16
      quant_stores.extend((
        sh[srow*Q6_STRIDE+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
        sh[srow*Q6_STRIDE+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020)))))
    drow=(warp*32+lane)%128; dhbase=block_epoch+drow*block_row_stride
    d_store=sh[drow*Q6_STRIDE+64].store(blocks[dhbase+104].cast(dtypes.uint32))
    scale_stores=[]
    for i0 in (0,64):
      srow=i0+warp*8+lane//4; hbase=block_epoch+srow*block_row_stride; si=lane%4
      scale=blocks[hbase+96+2*si].cast(dtypes.uint32).bitwise_or(blocks[hbase+97+2*si].cast(dtypes.uint32).lshift(16))
      scale_stores.append(sh[srow*Q6_STRIDE+65+si].store(scale))
    published_q6=UOp.group(*quant_stores,d_store,*scale_stores)
  else:
    sr=UOp.range(16,1501); srow=warp+8*sr; hbase=block_epoch+srow*block_row_stride
    ql=blocks[hbase+2*txi].cast(dtypes.uint32).bitwise_or(blocks[hbase+2*txi+1].cast(dtypes.uint32).lshift(16))
    qhi=(txi//16)*8+txi%8
    qh=blocks[hbase+64+2*qhi].cast(dtypes.uint32).bitwise_or(blocks[hbase+64+2*qhi+1].cast(dtypes.uint32).lshift(16))
    qshift=txi.bitwise_and(8)>>2
    q0=ql.bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).lshift(4).bitwise_and(0x30303030))
    q1=ql.rshift(4).bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).bitwise_and(0x30303030))
    kq0=2*txi-txi%16
    staged=UOp.group(
      sh[srow*Q6_STRIDE+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
      sh[srow*Q6_STRIDE+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020))),
      sh[srow*Q6_STRIDE+64].store(blocks[hbase+104].cast(dtypes.uint32),gate=(lane<1)),
      *(sh[srow*Q6_STRIDE+65+i].store(blocks[hbase+96+2*i].cast(dtypes.uint32).bitwise_or(
        blocks[hbase+97+2*i].cast(dtypes.uint32).lshift(16)),gate=((lane>=i)&(lane<i+1))) for i in range(4))).end(sr)
    published_q6=UOp.group(staged)
  ready_q6=None if combined_initial_publish else UOp.barrier(published_q6)

  # The first Q8 panel is published exactly once. Each lane owns 18 words.
  panel0=tuple(q8_record[q8_epoch+lid+i*256] for i in range(18))
  y0_target=shq.after(published_q6) if combined_initial_publish else shq
  published_y0=UOp.group(*(y0_target[lid+i*256].store(panel0[i]) for i in range(18)))
  if combined_initial_publish:
    ready_q6=ready_y0=UOp.barrier(published_y0)
  else:
    ready_y0=UOp.barrier(published_y0)
  if prefetch_second_panel:
    panel1_raw=tuple(q8_record[q8_epoch+Q8_WORDS+lid+i*256].load() for i in range(18))
    panel1_reg=tuple(UOp.placeholder((1,),dtypes.uint32,1510+i,addrspace=AddrSpace.REG) for i in range(18))
    preload=UOp.group(*(panel1_reg[i][0].store(panel1_raw[i]) for i in range(18)))
    panel1=tuple(panel1_reg[i].after(preload)[0] for i in range(18))
  else:
    panel1=()
    preload=None

  acc=[UOp.placeholder((1,),dtypes.float32,1520+i,addrspace=AddrSpace.REG) for i in range(64)]
  init=UOp.group(*(x[0].store(0.0) for x in acc)); acc=[x.after(init) for x in acc]

  def consume(kphase:int, ready_y, dep):
    # One warp retains the 16 Q6 fragments for this K128 half. The rolling
    # band schedule folds two IMMA results immediately into the 64 FP32 bank.
    sx=sh.after(ready_q6)
    av={(n,g):native_fragment_bitcast(native_fragment_materialized_x2(sx,
      (band*32+n*16+(lane&15))*Q6_STRIDE+(kphase*8+g)*4),dtypes.char.vec(8)) for n in range(2) for g in range(8)}
    meta={}
    for n in range(2):
      for r in range(4):
        row=band*32+n*16+lr+8*(r>>1)
        meta[n,r]=(sx[row*Q6_STRIDE+64],sx[row*Q6_STRIDE+65+kphase*2],sx[row*Q6_STRIDE+66+kphase*2])
    if factor_dA:
      update=dep
      # Four p-ordered partials are sufficient for one cg/n group. Folding
      # them immediately keeps the 64 persistent output accumulators live
      # without also retaining a second 64-register phase bank.
      for cg in range(8):
        for n in range(2):
          tmp=[UOp.placeholder((1,),dtypes.float32,1700+kphase*4+r,addrspace=AddrSpace.REG) for r in range(4)]
          tmp_init=UOp.group(*(x.after(update if update is not None else ready_y)[0].store(0.0) for x in tmp))
          tmp=[x.after(tmp_init) for x in tmp]; update=tmp_init
          for p in range(4):
            g0,g1=2*p,2*p+1; lfrag_col=cg*16+warp_phase*8+lr
            sy=shq.after(ready_y).after(preload) if preload is not None and kphase == 0 else shq.after(ready_y)
            qv0,qv1=sy[lfrag_col*Q8_STRIDE+4+g0*4+lc],sy[lfrag_col*Q8_STRIDE+4+g1*4+lc]
            def bv(qv): return UOp(Ops.STACK,dtypes.char.vec(4),tuple(qv.rshift(8*q).bitwise_and(255).cast(dtypes.char) for q in range(4)))
            axes=(tuple((1600+i,2) for i in range(3)),tuple((1610+i,2) for i in range(2)),tuple((1620+i,2) for i in range(2)))
            arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
            c0=UOp(Ops.WMMA,dtypes.int.vec(4),(av[n,g0],bv(qv0),UOp.const(dtypes.int.vec(4),0)),arg)
            c1=UOp(Ops.WMMA,dtypes.int.vec(4),(av[n,g1],bv(qv1),UOp.const(dtypes.int.vec(4),0)),arg)
            for r in range(4):
              col=cg*16+warp_phase*8+2*lc+(r&1); _,sw0,sw1=meta[n,r]; sw=(sw0,sw1)[p//2]; sp=kphase*4+p
              s0=sw.rshift((2*sp%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
              s1=sw.rshift(((2*sp+1)%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
              yscale=sy[col*Q8_STRIDE+p].bitcast(dtypes.float32)
              dot=(s0*c0.gep(r)+s1*c1.gep(r)).cast(dtypes.float32)
              carrier=tmp[r].after(update); update=carrier[0].store(carrier[0]+yscale*dot)
          for r in range(4):
            ai=cg*8+n*4+r; dw=meta[n,r][0]
            wd=dw.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
            carrier=acc[ai].after(update); update=carrier[0].store(carrier[0]+tmp[r].after(update)[0]*wd)
      return update

    update=dep
    for cg in range(8):
      for n in range(2):
        for p in range(4):
          g0,g1=2*p,2*p+1; lfrag_col=cg*16+warp_phase*8+lr
          sy=shq.after(ready_y).after(preload) if preload is not None and kphase == 0 else shq.after(ready_y)
          qv0,qv1=sy[lfrag_col*Q8_STRIDE+4+g0*4+lc],sy[lfrag_col*Q8_STRIDE+4+g1*4+lc]
          def bv(qv): return UOp(Ops.STACK,dtypes.char.vec(4),tuple(qv.rshift(8*q).bitwise_and(255).cast(dtypes.char) for q in range(4)))
          axes=(tuple((1600+i,2) for i in range(3)),tuple((1610+i,2) for i in range(2)),tuple((1620+i,2) for i in range(2)))
          arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
          c0=UOp(Ops.WMMA,dtypes.int.vec(4),(av[n,g0],bv(qv0),UOp.const(dtypes.int.vec(4),0)),arg)
          c1=UOp(Ops.WMMA,dtypes.int.vec(4),(av[n,g1],bv(qv1),UOp.const(dtypes.int.vec(4),0)),arg)
          for r in range(4):
            ai=cg*8+n*4+r; col=cg*16+warp_phase*8+2*lc+(r&1)
            dw,sw0,sw1=meta[n,r]; sw=(sw0,sw1)[p//2]; sp=kphase*4+p
            s0=sw.rshift((2*sp%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
            s1=sw.rshift(((2*sp+1)%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
            wd=dw.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
            yscale=sy[col*Q8_STRIDE+p].bitcast(dtypes.float32)
            dot=(s0*c0.gep(r)+s1*c1.gep(r)).cast(dtypes.float32)
            carrier=acc[ai].after(update) if update is not None else acc[ai]
            update=carrier[0].store(carrier[0]+wd*yscale*dot)
    return update

  phase0=consume(0,ready_y0,ready_y0)
  before_overwrite=UOp.barrier(UOp.group(phase0))
  if not prefetch_second_panel:
    ordered_record=q8_record.after(before_overwrite)
    panel1=tuple(ordered_record[q8_epoch+Q8_WORDS+lid+i*256].load() for i in range(18))
  ready_y1=UOp.barrier(UOp.group(*(shq.after(before_overwrite)[lid+i*256].store(panel1[i]) for i in range(18))))
  phase1=consume(1,ready_y1,phase0)
  # The final barrier protects shared Q6/Q8 from the next K256 epoch. Closing
  # the RANGE on it makes the accumulator register bank loop-carried.
  lifecycle_end=UOp.barrier(UOp.group(phase1))
  loop_end=UOp.group(lifecycle_end).end(epoch)

  stores=[]
  for cg in range(8):
    for n in range(2):
      for r in range(4):
        ai=cg*8+n*4+r; row=band*32+n*16+lr+8*(r>>1); col=cg*16+warp_phase*8+2*lc+(r&1)
        stores.append(out[output_slot*ROWS*COLS+row*COLS+col].store(acc[ai].after(loop_end)[0],gate=active))
  suffix="prefetch" if prefetch_second_panel else "serial"
  if combined_initial_publish: suffix += "_combined_publish"
  if factor_dA: suffix += "_factor_da"
  if oracle_publisher: suffix += "_oracle_publisher"
  suffix += f"_streamk_s{streamk_segment}" if streamk_owners is not None else f"_d{depth}"
  return UOp.sink(*stores,arg=KernelInfo(name=f"nv_q6_oracle_broad_cta_{suffix}",opts_to_apply=()))


__all__=["ROWS","COLS","K","Q6_STRIDE","Q8_STRIDE","SHARED_BYTES","q6_oracle_broad_cta_kernel"]
