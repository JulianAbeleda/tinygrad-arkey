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
                               oracle_publisher:bool=False, depth:int=1, tile_grid:tuple[int,int]|None=None,
                               fp32_scale_grouping:str="legacy", fp32_p_tree:str="legacy",
                               fp32_contraction:str="implicit",
                               weight_scale_contract:str="legacy", trace=None, trace_config:tuple[int,int]|None=None,
                               streamk_owners:int|None=None, streamk_segment:int=0):
  """One exact llama-normalized 128x128xK256 work unit per CTA.

  ``blocks`` is 128 canonical Q6_K rows. ``q8_record`` is two canonical
  128x36-word Q8_1 panels. The second panel can either be register-prefetched
  across the first consumer or loaded after it for a causal A/B.
  """
  if replicas < 1: raise ValueError("replicas must be positive")
  if depth < 1: raise ValueError("depth must be positive")
  if streamk_owners is not None and (not 1 <= streamk_owners <= 256 or streamk_segment not in (0,1)):
    raise ValueError("streamk requires 1..256 owners and segment zero or one")
  if tile_grid is not None:
    if streamk_owners is not None: raise ValueError("tile_grid and streamk are mutually exclusive")
    if len(tile_grid) != 2: raise ValueError("tile_grid requires tiles_m and tiles_n")
    tiles_m,tiles_n=tile_grid
    if tiles_m < 1 or tiles_n < 1 or replicas != tiles_m*tiles_n:
      raise ValueError("replicas must equal the positive tile_grid product")
  if factor_dA:
    if fp32_scale_grouping != "legacy": raise ValueError("factored dA has one scale grouping")
    if fp32_p_tree == "legacy":
      if fp32_contraction != "implicit": raise ValueError("legacy factored dA requires implicit contraction")
    elif fp32_p_tree not in ("left","inner_left","inner_right","right","balanced") or \
         fp32_contraction not in ("none","tmp_only","final_only","both"):
      raise ValueError("illegal factored FP32 tree or contraction")
  else:
    if fp32_p_tree != "legacy": raise ValueError("direct dA has no local p reduction tree")
    if fp32_scale_grouping == "legacy":
      if fp32_contraction != "implicit": raise ValueError("legacy direct dA requires implicit contraction")
    elif fp32_scale_grouping not in ("wd_yscale_then_dot","wd_then_yscale_dot") or fp32_contraction not in ("none","final"):
      raise ValueError("illegal direct FP32 grouping or contraction")
  fp_mul=lambda a,b: UOp(Ops.CUSTOMI,dtypes.float32,(a,b),arg="__fmul_rn({0},{1})")
  fp_add=lambda a,b: UOp(Ops.CUSTOMI,dtypes.float32,(a,b),arg="__fadd_rn({0},{1})")
  fp_fma=lambda a,b,c: UOp(Ops.CUSTOMI,dtypes.float32,(a,b,c),arg="__fmaf_rn({0},{1},{2})")
  if weight_scale_contract not in ("legacy","trusted_fp16","trusted_fp16_packed"): raise ValueError("unknown weight scale contract")
  if weight_scale_contract != "legacy" and (factor_dA or fp32_contraction != "implicit"):
    raise ValueError("trusted weight scale contract requires legacy direct accumulation")
  packed_weight_scales=weight_scale_contract == "trusted_fp16_packed"
  if (trace is None) != (trace_config is None): raise ValueError("trace and trace_config must be supplied together")
  if trace_config is not None:
    trace_row,trace_col=trace_config
    if not (0 <= trace_row < ROWS and 0 <= trace_col < COLS): raise ValueError("trace coordinate outside tile")
    trace_band=trace_row//32; trace_n=(trace_row%32)//16; trace_lr=trace_row%8
    trace_r=2*((trace_row%16)//8)+(trace_col&1); trace_cg=trace_col//16
    trace_warp_phase=(trace_col%16)//8; trace_lc=(trace_col%8)//2
    trace_warp=2*trace_band+trace_warp_phase; trace_lane=4*trace_lr+trace_lc
    trace_thread=32*trace_warp+trace_lane; trace_ai=8*trace_cg+4*trace_n+trace_r
  lid=UOp.special(256,"lidx0"); warp,lane=lid//32,lid%32
  lane_gate=lambda i:(lid>=i)&(lid<i+1)
  lr,lc=lane>>2,lane&3; band,warp_phase=warp>>1,warp&1
  grid=streamk_owners if streamk_owners is not None else replicas
  bid=UOp.special(grid,"gidx0") if grid > 1 else UOp.const(dtypes.int32,0)
  arena=UOp.placeholder((Q6_WORDS+Q8_WORDS,),dtypes.uint32,1500,addrspace=AddrSpace.LOCAL).replace(
    tag=RuntimeLocalAllocation(SHARED_BYTES))
  sh,shq=arena,arena[Q6_WORDS:]
  if streamk_owners is None:
    epoch=UOp.range(depth,1499); output_slot=bid
    if tile_grid is None:
      block_row_stride=105
      block_epoch=epoch*(ROWS*105); q8_epoch=epoch*(2*Q8_WORDS)
    else:
      tile_m=bid%tiles_m; tile_n=bid//tiles_m; block_row_stride=depth*105
      block_epoch=tile_n*ROWS*depth*105+epoch*105
      q8_epoch=(tile_m*depth+epoch)*(2*Q8_WORDS)
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
  def packed_weight_pair(hbase, si):
    scale_pair=blocks[hbase+96+si]
    d=blocks[hbase+104].bitcast(dtypes.half).cast(dtypes.float32)
    s0=scale_pair.bitwise_and(255).cast(dtypes.char).cast(dtypes.float32)
    s1=scale_pair.rshift(8).bitwise_and(255).cast(dtypes.char).cast(dtypes.float32)
    h0=fp_mul(d,s0).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
    h1=fp_mul(d,s1).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
    return h0.bitwise_or(h1.lshift(16))
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
    dword=(blocks[dhbase+104].bitcast(dtypes.half).cast(dtypes.float32).bitcast(dtypes.uint32)
           if weight_scale_contract != "legacy" else blocks[dhbase+104].cast(dtypes.uint32))
    d_store=sh[drow*Q6_STRIDE+64].store(dword)
    scale_stores=[]
    if packed_weight_scales:
      for i0 in (0,32,64,96):
        srow=i0+warp*4+lane//8; hbase=block_epoch+srow*block_row_stride; si=lane%8
        scale_stores.append(sh[srow*Q6_STRIDE+65+si].store(packed_weight_pair(hbase,si)))
    else:
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
    if packed_weight_scales:
      si=lane.bitwise_and(7)
      scale_stores=(sh[srow*Q6_STRIDE+65+si].store(packed_weight_pair(hbase,si),gate=(lane<8)),)
    else:
      scale_stores=tuple(sh[srow*Q6_STRIDE+65+i].store(blocks[hbase+96+2*i].cast(dtypes.uint32).bitwise_or(
        blocks[hbase+97+2*i].cast(dtypes.uint32).lshift(16)),gate=((lane>=i)&(lane<i+1))) for i in range(4))
    staged=UOp.group(
      sh[srow*Q6_STRIDE+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
      sh[srow*Q6_STRIDE+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020))),
      sh[srow*Q6_STRIDE+64].store((blocks[hbase+104].bitcast(dtypes.half).cast(dtypes.float32).bitcast(dtypes.uint32)
        if weight_scale_contract != "legacy" else blocks[hbase+104].cast(dtypes.uint32)),gate=(lane<1)),*scale_stores).end(sr)
    published_q6=staged
  ready_q6=None if combined_initial_publish else UOp.barrier(published_q6)

  # The first Q8 panel is published exactly once. Each lane owns 18 words.
  panel0=tuple(q8_record[q8_epoch+lid+i*256] for i in range(18))
  y0_target=shq.after(published_q6) if combined_initial_publish else shq
  published_y0=UOp.group(*(y0_target[lid+i*256].store(panel0[i]) for i in range(18)))
  if combined_initial_publish:
    ready_q6=ready_y0=UOp.barrier(published_y0)
  else:
    ready_y0=UOp.barrier(published_y0)
  trace_y0=None
  if trace is not None:
    trace_epoch=16+epoch*248
    pad_start=73 if packed_weight_scales else 69
    marked=UOp.group(*(sh.after(ready_q6)[trace_row*Q6_STRIDE+i].store(UOp.const(dtypes.uint32,0xdeadbeef),gate=lane_gate(i-pad_start))
      for i in range(pad_start,76)))
    marked_ready=UOp.barrier(marked)
    q6_dump=UOp.group(*(trace[trace_epoch+i].store(sh.after(marked_ready)[trace_row*Q6_STRIDE+i],gate=lane_gate(i)) for i in range(76)))
    ready_q6=UOp.barrier(q6_dump)
    dword=sh.after(ready_q6)[trace_row*Q6_STRIDE+64]
    dfp=(dword.bitcast(dtypes.float32) if weight_scale_contract != "legacy" else
         dword.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32))
    trace_d=trace[trace_epoch+148].store(dfp.bitcast(dtypes.uint32),gate=lane_gate(trace_thread))
    trace_y0=UOp.group(trace_d,*(trace[trace_epoch+76+i].store(shq.after(ready_y0)[trace_col*Q8_STRIDE+i],gate=lane_gate(i)) for i in range(36)))
  if prefetch_second_panel:
    panel1_raw=tuple(q8_record[q8_epoch+Q8_WORDS+lid+i*256].load() for i in range(18))
    panel1_reg=tuple(UOp.placeholder((1,),dtypes.uint32,1510+i,addrspace=AddrSpace.REG) for i in range(18))
    preload=UOp.group(*(panel1_reg[i][0].store(panel1_raw[i]) for i in range(18)))
    panel1=tuple(panel1_reg[i].after(preload)[0] for i in range(18))
  else:
    panel1=()
    preload=None

  acc=[UOp.placeholder((1,),dtypes.float32,1520+i,addrspace=AddrSpace.REG) for i in range(64)]
  header=() if trace is None else tuple(trace[i].store(UOp.const(dtypes.uint32,v),gate=lane_gate(trace_thread)) for i,v in enumerate(
    (0x51365452,1,trace_row,trace_col,trace_thread,trace_warp,trace_lane,trace_band,trace_warp_phase,trace_lr,trace_lc,
     trace_cg,trace_n,trace_r,trace_ai,248)))
  init=UOp.group(*(x[0].store(0.0) for x in acc),*header); acc=[x.after(init) for x in acc]

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
        meta[n,r]=(tuple(sx[row*Q6_STRIDE+65+kphase*4+p] for p in range(4)) if packed_weight_scales else
          (sx[row*Q6_STRIDE+64],sx[row*Q6_STRIDE+65+kphase*2],sx[row*Q6_STRIDE+66+kphase*2]))
    if weight_scale_contract in ("trusted_fp16","trusted_fp16_packed"):
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
              ai=cg*8+n*4+r; col=cg*16+warp_phase*8+2*lc+(r&1); yscale=sy[col*Q8_STRIDE+p].bitcast(dtypes.float32)
              z0,z1=c0.gep(r),c1.gep(r)
              if packed_weight_scales:
                sw=meta[n,r][p]
                ws0=sw.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
                ws1=sw.rshift(16).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
                dot0=fp_mul(ws0,z0.cast(dtypes.float32)); dot1=fp_mul(ws1,z1.cast(dtypes.float32)); weighted=fp_add(dot0,dot1)
              else:
                dw,sw0,sw1=meta[n,r]; sw=(sw0,sw1)[p//2]; sp=kphase*4+p
                s0=sw.rshift((2*sp%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
                s1=sw.rshift(((2*sp+1)%4)*8).bitwise_and(255).cast(dtypes.char).cast(dtypes.int32)
                wd=dw.bitcast(dtypes.float32); dot0=s0*z0; dot1=s1*z1; dot=dot0+dot1
                ws0=(wd*s0.cast(dtypes.float32)).cast(dtypes.half).cast(dtypes.float32)
                ws1=(wd*s1.cast(dtypes.float32)).cast(dtypes.half).cast(dtypes.float32)
                weighted=fp_add(fp_mul(ws0,z0.cast(dtypes.float32)),fp_mul(ws1,z1.cast(dtypes.float32)))
              carrier=acc[ai].after(update) if update is not None else acc[ai]
              next_value=fp_add(carrier[0],fp_mul(yscale,weighted))
              update=carrier[0].store(next_value)
              if trace is not None and cg==trace_cg and n==trace_n and r==trace_r:
                tb=trace_epoch+149+kphase*48+p*12
                vals=((qv0,qv1,yscale.bitcast(dtypes.uint32),ws0.bitcast(dtypes.uint32),ws1.bitcast(dtypes.uint32),
                  z0.cast(dtypes.uint32),z1.cast(dtypes.uint32),dot0.bitcast(dtypes.uint32),dot1.bitcast(dtypes.uint32),weighted.bitcast(dtypes.uint32),
                  carrier[0].bitcast(dtypes.uint32),next_value.bitcast(dtypes.uint32)) if packed_weight_scales else
                  (qv0,qv1,yscale.bitcast(dtypes.uint32),s0.cast(dtypes.uint32),s1.cast(dtypes.uint32),
                  z0.cast(dtypes.uint32),z1.cast(dtypes.uint32),dot0.cast(dtypes.uint32),dot1.cast(dtypes.uint32),dot.cast(dtypes.uint32),
                  carrier[0].bitcast(dtypes.uint32),next_value.bitcast(dtypes.uint32)))
                update=UOp.group(update,*(trace[tb+i].store(v,gate=lane_gate(trace_thread)) for i,v in enumerate(vals)))
      return update
    if factor_dA and fp32_p_tree != "legacy":
      update=dep
      trees={"left":(((0,1),2),3),"inner_left":((0,(1,2)),3),"inner_right":(0,((1,2),3)),
             "right":(0,(1,(2,3))),"balanced":((0,1),(2,3))}
      for cg in range(8):
        for n in range(2):
          tmp=[UOp.placeholder((1,),dtypes.float32,1700+kphase*4+r,addrspace=AddrSpace.REG) for r in range(4)]
          terms=[[] for _ in range(4)]
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
              terms[r].append((yscale,dot))
          for r in range(4):
            def eval_tree(node):
              if isinstance(node,int): return fp_mul(*terms[r][node])
              left,right=node
              if fp32_contraction in ("tmp_only","both"):
                if isinstance(left,int): return fp_fma(*terms[r][left],eval_tree(right))
                if isinstance(right,int): return fp_fma(*terms[r][right],eval_tree(left))
              return fp_add(eval_tree(left),eval_tree(right))
            carrier=tmp[r].after(update if update is not None else ready_y)
            update=carrier[0].store(eval_tree(trees[fp32_p_tree]))
          for r in range(4):
            ai=cg*8+n*4+r; dw=meta[n,r][0]
            wd=dw.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
            value=tmp[r].after(update)[0]; carrier=acc[ai].after(update)
            next_value=(fp_fma(value,wd,carrier[0]) if fp32_contraction in ("final_only","both") else
                        fp_add(carrier[0],fp_mul(value,wd)))
            update=carrier[0].store(next_value)
      return update

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
            z0,z1=c0.gep(r),c1.gep(r); dot0=s0*z0; dot1=s1*z1; dot=(dot0+dot1).cast(dtypes.float32)
            carrier=acc[ai].after(update) if update is not None else acc[ai]
            if fp32_scale_grouping == "legacy": next_value=carrier[0]+wd*yscale*dot
            else:
              if fp32_scale_grouping == "wd_yscale_then_dot": lhs,rhs=fp_mul(wd,yscale),dot
              else: lhs,rhs=wd,fp_mul(yscale,dot)
              next_value=(fp_fma(lhs,rhs,carrier[0]) if fp32_contraction == "final" else
                          fp_add(carrier[0],fp_mul(lhs,rhs)))
            update=carrier[0].store(next_value)
            if trace is not None and cg==trace_cg and n==trace_n and r==trace_r:
              tb=trace_epoch+149+kphase*48+p*12
              vals=(qv0,qv1,yscale.bitcast(dtypes.uint32),s0.cast(dtypes.uint32),s1.cast(dtypes.uint32),
                z0.cast(dtypes.uint32),z1.cast(dtypes.uint32),dot0.cast(dtypes.uint32),dot1.cast(dtypes.uint32),
                (dot0+dot1).cast(dtypes.uint32),
                carrier[0].bitcast(dtypes.uint32),next_value.bitcast(dtypes.uint32))
              update=UOp.group(update,*(trace[tb+i].store(v,gate=lane_gate(trace_thread)) for i,v in enumerate(vals)))
    return update

  phase0=consume(0,ready_y0,trace_y0 if trace_y0 is not None else ready_y0)
  trace_p0=None if trace is None else trace.after(phase0)[trace_epoch+245].store(
    acc[trace_ai].after(phase0)[0].bitcast(dtypes.uint32),gate=lane_gate(trace_thread))
  before_overwrite=UOp.barrier(trace_p0 if trace_p0 is not None else phase0)
  if not prefetch_second_panel:
    ordered_record=q8_record.after(before_overwrite)
    panel1=tuple(ordered_record[q8_epoch+Q8_WORDS+lid+i*256].load() for i in range(18))
  ready_y1=UOp.barrier(UOp.group(*(shq.after(before_overwrite)[lid+i*256].store(panel1[i]) for i in range(18))))
  trace_y1=None if trace is None else UOp.group(*(trace[trace_epoch+112+i].store(shq.after(ready_y1)[trace_col*Q8_STRIDE+i],gate=lane_gate(i)) for i in range(36)))
  phase1=consume(1,ready_y1,trace_y1 if trace_y1 is not None else phase0)
  # The final barrier protects shared Q6/Q8 from the next K256 epoch. Closing
  # the RANGE on it makes the accumulator register bank loop-carried.
  trace_end=() if trace is None else (
    trace[trace_epoch+246].store(acc[trace_ai].after(phase1)[0].bitcast(dtypes.uint32),gate=lane_gate(trace_thread)),
    trace[trace_epoch+247].store(acc[trace_ai].after(phase1)[0].bitcast(dtypes.uint32),gate=lane_gate(trace_thread)))
  lifecycle_end=UOp.barrier(UOp.group(*trace_end) if trace_end else phase1)
  loop_end=UOp.group(lifecycle_end).end(epoch)

  stores=[]
  for cg in range(8):
    for n in range(2):
      for r in range(4):
        ai=cg*8+n*4+r; row=band*32+n*16+lr+8*(r>>1); col=cg*16+warp_phase*8+2*lc+(r&1)
        out_index=(output_slot*ROWS*COLS+row*COLS+col if tile_grid is None else
          (tile_m*COLS+col)*(tiles_n*ROWS)+tile_n*ROWS+row)
        stores.append(out[out_index].store(acc[ai].after(loop_end)[0],gate=active))
  suffix="prefetch" if prefetch_second_panel else "serial"
  if combined_initial_publish: suffix += "_combined_publish"
  if factor_dA: suffix += "_factor_da"
  if oracle_publisher: suffix += "_oracle_publisher"
  if fp32_contraction != "implicit": suffix += f"_fp32_{fp32_scale_grouping}_{fp32_p_tree}_{fp32_contraction}"
  if weight_scale_contract == "trusted_fp16": suffix += "_trusted_fp16_ws"
  if weight_scale_contract == "trusted_fp16_packed": suffix += "_trusted_fp16_packed_ws"
  if trace is not None: suffix += "_trace"
  if tile_grid is not None: suffix += f"_tiles{tiles_m}x{tiles_n}"
  suffix += f"_streamk_s{streamk_segment}" if streamk_owners is not None else f"_d{depth}"
  return UOp.sink(*stores,arg=KernelInfo(name=f"nv_q6_oracle_broad_cta_{suffix}",opts_to_apply=()))


__all__=["ROWS","COLS","K","Q6_STRIDE","Q8_STRIDE","SHARED_BYTES","q6_oracle_broad_cta_kernel"]
