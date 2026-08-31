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
                               combined_initial_publish:bool=False):
  """One exact llama-normalized 128x128xK256 work unit per CTA.

  ``blocks`` is 128 canonical Q6_K rows. ``q8_record`` is two canonical
  128x36-word Q8_1 panels. The second panel can either be register-prefetched
  across the first consumer or loaded after it for a causal A/B.
  """
  if replicas < 1: raise ValueError("replicas must be positive")
  lid=UOp.special(256,"lidx0"); warp,lane=lid//32,lid%32
  lr,lc=lane>>2,lane&3; band,warp_phase=warp>>1,warp&1
  bid=UOp.special(replicas,"gidx0") if replicas > 1 else UOp.const(dtypes.int32,0)
  arena=UOp.placeholder((Q6_WORDS+Q8_WORDS,),dtypes.uint32,1500,addrspace=AddrSpace.LOCAL).replace(
    tag=RuntimeLocalAllocation(SHARED_BYTES))
  sh,shq=arena,arena[Q6_WORDS:]

  # Expand all 128 canonical Q6 rows into llama's 76-word shared layout.
  sr=UOp.range(16,1501); srow=warp+8*sr; hbase=srow*105; txi=lane
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
  panel0=tuple(q8_record[lid+i*256] for i in range(18))
  y0_target=shq.after(published_q6) if combined_initial_publish else shq
  published_y0=UOp.group(*(y0_target[lid+i*256].store(panel0[i]) for i in range(18)))
  if combined_initial_publish:
    ready_q6=ready_y0=UOp.barrier(published_y0)
  else:
    ready_y0=UOp.barrier(published_y0)
  if prefetch_second_panel:
    panel1_raw=tuple(q8_record[Q8_WORDS+lid+i*256].load() for i in range(18))
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
            term=wd*yscale*(s0*c0.gep(r)+s1*c1.gep(r)).cast(dtypes.float32)
            carrier=acc[ai].after(update) if update is not None else acc[ai]
            update=carrier[0].store(carrier[0]+term)
    return update

  phase0=consume(0,ready_y0,None)
  before_overwrite=UOp.barrier(UOp.group(phase0))
  if not prefetch_second_panel:
    ordered_record=q8_record.after(before_overwrite)
    panel1=tuple(ordered_record[Q8_WORDS+lid+i*256].load() for i in range(18))
  ready_y1=UOp.barrier(UOp.group(*(shq.after(before_overwrite)[lid+i*256].store(panel1[i]) for i in range(18))))
  phase1=consume(1,ready_y1,phase0)
  lifecycle_end=UOp.barrier(UOp.group(phase1)) if combined_initial_publish else phase1

  stores=[]
  for cg in range(8):
    for n in range(2):
      for r in range(4):
        ai=cg*8+n*4+r; row=band*32+n*16+lr+8*(r>>1); col=cg*16+warp_phase*8+2*lc+(r&1)
        stores.append(out[bid*ROWS*COLS+row*COLS+col].store(acc[ai].after(lifecycle_end)[0]))
  suffix="prefetch" if prefetch_second_panel else "serial"
  if combined_initial_publish: suffix += "_combined_publish"
  return UOp.sink(*stores,arg=KernelInfo(name=f"nv_q6_oracle_broad_cta_{suffix}",opts_to_apply=()))


__all__=["ROWS","COLS","K","Q6_STRIDE","Q8_STRIDE","SHARED_BYTES","q6_oracle_broad_cta_kernel"]
