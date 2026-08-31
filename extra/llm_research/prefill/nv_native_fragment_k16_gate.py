"""Generated-UOp K16 native fragment gate."""
import numpy as np
import struct
from tinygrad import Tensor,dtypes,Device
from tinygrad.codegen.late.native_fragment import native_fragment_x2,packed_i8_sub
from tinygrad.uop.ops import AxisType,KernelInfo,Ops,UOp

def _fragment_x2_i8(buffer, native_index, scalar_index, style:str):
 """Return the exact eight-byte MMA-A carrier through native or scalar LDS."""
 if style == "native": return native_fragment_x2(buffer,native_index).bitcast(dtypes.char.vec(8))
 if style == "scalar":
  # ldmatrix.x2 takes row addresses from lanes 0..15, then redistributes two
  # words to every lane.  The destination ownership is (row,word)=(lane>>2,
  # lane&3) in matrix 0 and the same word in row+8 in matrix 1.
  words=(buffer[scalar_index],buffer[scalar_index+32])
  return UOp(Ops.STACK,dtypes.char.vec(8),tuple(words[q//4].rshift((q%4)*8).bitwise_and(255).cast(dtypes.char) for q in range(8)))
 raise ValueError(f"unknown fragment load style {style!r}")
def kernel(out, a, b):
 lane=UOp.special(32,"lidx0"); sh=UOp.placeholder((64,),dtypes.uint32,20,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.LOCAL)
 ready=UOp.barrier(UOp.group(*(sh[lane+32*i].store(a[lane+32*i]) for i in range(2))))
 av=native_fragment_x2(sh.after(ready),(lane&15)*4+(lane>>4)*2).bitcast(dtypes.char.vec(8)); lr,lc=lane>>2,lane&3
 bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(b[(4*lc+q)*8+lr] for q in range(4)))
 axes=(tuple((100+i,2) for i in range(3)),tuple((110+i,2) for i in range(2)),tuple((120+i,2) for i in range(2)))
 arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
 c=UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),arg)
 return UOp.sink(*(out[(lr+8*(r>>1))*8+2*lc+(r&1)].store(c.gep(r)) for r in range(4)),arg=KernelInfo(name='nv_native_fragment_k16',opts_to_apply=()))

def q6_two_k16_kernel(out, dot0, dot1, a0, b0, s0, d, a1, b1, s1, dB):
 """Bounded Q6 semantic gate: two independent signed K=16 IMMA nodes.

 Inputs A0/A1 are row-major int8 [16,16], B0/B1 row-major int8 [16,8].
 The two int32 tiles are materialized in ``dot0`` and ``dot1`` before the
 scalar FP32 Q6 epilogue is stored in ``out``.
 """
 lane=UOp.special(32,"lidx0")
 def one(a, b, slot):
  sh=UOp.placeholder((64,),dtypes.uint32,slot,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.LOCAL)
  # Four adjacent signed bytes form each word consumed by ldmatrix.
  words=tuple(UOp(Ops.STACK,dtypes.char.vec(4),tuple(a[4*(lane+32*i)+q] for q in range(4))).bitcast(dtypes.uint32)
              for i in range(2))
  ready=UOp.barrier(UOp.group(*(sh[lane+32*i].store(words[i]) for i in range(2))))
  av=native_fragment_x2(sh.after(ready),(lane&15)*4+(lane>>4)*2).bitcast(dtypes.char.vec(8))
  lr,lc=lane>>2,lane&3
  bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(b[(4*lc+q)*8+lr] for q in range(4)))
  axes=(tuple((100+i,2) for i in range(3)),tuple((110+i,2) for i in range(2)),tuple((120+i,2) for i in range(2)))
  arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
  return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),arg), lr, lc
 c0,lr,lc=one(a0,b0,0); c1,_,_=one(a1,b1,1)
 scale0=s0[0].cast(dtypes.float32); scale1=s1[0].cast(dtypes.float32)
 gain=d[0].cast(dtypes.float32)*dB[0].cast(dtypes.float32)
 writes=[]
 for r in range(4):
  idx=(lr+8*(r>>1))*8+2*lc+(r&1)
  z0,z1=c0.gep(r),c1.gep(r)
  writes += [dot0[idx].store(z0), dot1[idx].store(z1), out[idx].store(gain*(scale0*z0.cast(dtypes.float32)+scale1*z1.cast(dtypes.float32)))]
 return UOp.sink(*writes,arg=KernelInfo(name='nv_native_fragment_q6_two_k16',opts_to_apply=()))

def q6_first_two_k16_numpy(block):
 """Decode the first two canonical Q6_K K16 groups (the tile contract oracle)."""
 if len(block) not in (210,16*210): raise ValueError("expected one or sixteen canonical Q6_K blocks")
 raw=np.frombuffer(block,dtype=np.uint8).reshape(-1,210)
 ql,qh=raw[:,:128],raw[:,128:192]
 q0=(ql[:,:16]&15) | ((qh[:,:16]&3)<<4)
 q1=(ql[:,16:32]&15) | ((qh[:,16:32]&3)<<4)
 q=np.stack((q0,q1),axis=1).astype(np.int8)-32
 scales=raw[:,192:194].view(np.int8)
 d=np.frombuffer(raw[:,208:210].tobytes(),dtype='<f2').astype(np.float32)
 return (q[0],scales[0],float(d[0])) if len(raw)==1 else (q,scales,d)

def q6_packed_two_k16_kernel(out, dot0, dot1, blocks, b0, b1, dB):
 """Direct packed Q6_K gate: sixteen canonical blocks, no expanded A input."""
 lane=UOp.special(32,"lidx0")
 def byte(off): return blocks[off//2].cast(dtypes.uint32).rshift((off%2)*8).bitwise_and(255)
 def one(b, slot):
  sh=UOp.placeholder((64,),dtypes.uint32,30+slot,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.LOCAL)
  stores=[]
  for i in range(2):
   z=lane+32*i; row=z>>2; word=z&3; base=row*210; vals=[]
   ql_off=(slot//8)*64+(slot%4)*16; qh_off=(slot//8)*32+(slot%2)*16; qh_shift=(slot%8//2)*2
   for q in range(4):
    lo=byte(base+ql_off+4*word+q); hi=byte(base+128+qh_off+4*word+q)
    lbits=lo.rshift(4 if slot%8>=4 else 0).bitwise_and(15); hbits=hi.rshift(qh_shift).bitwise_and(3).lshift(4)
    v=lbits.bitwise_or(hbits).alu(
      Ops.SUB,UOp.const(dtypes.uint32,32)).cast(dtypes.char)
    vals.append(v)
   stores.append(sh[z].store(UOp(Ops.STACK,dtypes.char.vec(4),tuple(vals)).bitcast(dtypes.uint32)))
  ready=UOp.barrier(UOp.group(*stores))
  av=native_fragment_x2(sh.after(ready),(lane&15)*4+(lane>>4)*2).bitcast(dtypes.char.vec(8))
  lr,lc=lane>>2,lane&3
  bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(b[(4*lc+q)*8+lr] for q in range(4)))
  axes=(tuple((200+i,2) for i in range(3)),tuple((210+i,2) for i in range(2)),tuple((220+i,2) for i in range(2)))
  arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
  return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),arg)
 c0=one(b0,0); c1=one(b1,1)
 lr,lc=lane>>2,lane&3; writes=[]
 for r in range(4):
  idx=(lr+8*(r>>1))*8+2*lc+(r&1)
  row=lr+8*(r>>1); base=row*210; z0,z1=c0.gep(r),c1.gep(r)
  scale0=byte(base+192).cast(dtypes.char).cast(dtypes.float32); scale1=byte(base+193).cast(dtypes.char).cast(dtypes.float32)
  wd=blocks[(base+208)//2].bitcast(dtypes.half).cast(dtypes.float32)
  col=2*lc+(r&1)
  value=(wd*dB[col].cast(dtypes.float32))*(scale0*z0.cast(dtypes.float32)+scale1*z1.cast(dtypes.float32))
  writes += [dot0[idx].store(z0),dot1[idx].store(z1),out[idx].store(value)]
 return UOp.sink(*writes,arg=KernelInfo(name='nv_native_fragment_q6_packed_two_k16',opts_to_apply=()))

def q6_packed_k256_kernel(out, dot, blocks, b, dB, *, fragment_load:str="native", replicas:int=1):
 """Full one-block Q6_K K=256 gate, unrolled as sixteen K16 MMAs.

 ``scalar`` is the apples-to-apples control for the native x2 fragment load:
 it reads the same two adjacent shared words into the same eight-byte WMMA-A
 carrier.  ``replicas`` only gives the timing gate enough independent CTAs to
 rise above launch latency; every CTA executes the identical K256 body.
 """
 if fragment_load not in ("native","scalar"): raise ValueError("fragment_load must be native or scalar")
 if replicas < 1: raise ValueError("replicas must be positive")
 lane=UOp.special(32,"lidx0"); bid=UOp.special(replicas,"gidx0") if replicas > 1 else UOp.const(dtypes.int32,0); lr,lc=lane>>2,lane&3
 def byte(off): return blocks[off//2].cast(dtypes.uint32).rshift((off%2)*8).bitwise_and(255)
 def group(g):
  sh=UOp.placeholder((64,),dtypes.uint32,100+g,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.LOCAL)
  stores=[]; ql_off=(g//8)*64+(g%4)*16; qh_off=(g//8)*32+(g%2)*16; qh_shift=((g%8)//2)*2
  for i in range(2):
   z=lane+32*i; row=z>>2; word=z&3; base=row*210; vals=[]
   for q in range(4):
    lo=byte(base+ql_off+4*word+q); hi=byte(base+128+qh_off+4*word+q)
    vals.append(lo.rshift(4 if g%8>=4 else 0).bitwise_and(15).bitwise_or(hi.rshift(qh_shift).bitwise_and(3).lshift(4)).alu(
      Ops.SUB,UOp.const(dtypes.uint32,32)).cast(dtypes.char))
   stores.append(sh[z].store(UOp(Ops.STACK,dtypes.char.vec(4),tuple(vals)).bitcast(dtypes.uint32)))
  ready=UOp.barrier(UOp.group(*stores)); frag_index=(lane&15)*4+(lane>>4)*2
  av=_fragment_x2_i8(sh.after(ready),frag_index,lane,fragment_load)
  bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(b[(g*16+4*lc+q)*8+lr] for q in range(4)))
  axes=(tuple((300+i,2) for i in range(3)),tuple((310+i,2) for i in range(2)),tuple((320+i,2) for i in range(2)))
  arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
  return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),arg)
 cs=[group(g) for g in range(16)]; writes=[]
 for r in range(4):
  idx=(lr+8*(r>>1))*8+2*lc+(r&1); row=lr+8*(r>>1); base=row*210; col=2*lc+(r&1)
  acc=None
  for p in range(8):
   z=cs[2*p].gep(r); z1=cs[2*p+1].gep(r)
   s0=byte(base+192+2*p).cast(dtypes.char).cast(dtypes.float32); s1=byte(base+193+2*p).cast(dtypes.char).cast(dtypes.float32)
   term=(s0*z.cast(dtypes.float32)+s1*z1.cast(dtypes.float32))*(blocks[(base+208)//2].bitcast(dtypes.half).cast(dtypes.float32)*dB[p*8+col])
   acc=term if acc is None else acc+term
  writes.append(dot[bid*128+idx].store(acc))
  writes.append(out[bid*128+idx].store(acc))
 name='nv_native_fragment_q6_packed_k256' if fragment_load == "native" and replicas == 1 else f'nv_{fragment_load}_fragment_q6_packed_k256_ab'
 return UOp.sink(*writes,arg=KernelInfo(name=name,opts_to_apply=()))

def q6_packed_kblocks_kernel(k_blocks:int):
 """Looped 16x8 Q6_K x Q8 body for an integral number of canonical K=256 blocks."""
 if k_blocks < 1: raise ValueError("k_blocks must be positive")
 def kernel(out, blocks, b, dB):
  lane=UOp.special(32,"lidx0"); lr,lc=lane>>2,lane&3; blk=UOp.range(k_blocks,0,axis_type=AxisType.REDUCE)
  def byte(off): return blocks[off//2].cast(dtypes.uint32).rshift((off%2)*8).bitwise_and(255)
  sh=UOp.placeholder((16*76,),dtypes.uint32,200,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.LOCAL)
  st=UOp.range(16,1,axis_type=AxisType.LOOP); hbase=(st*k_blocks+blk)*105; txi=lane
  ql=blocks[hbase+2*txi].cast(dtypes.uint32).bitwise_or(blocks[hbase+2*txi+1].cast(dtypes.uint32).lshift(16))
  qhi=(txi//16)*8+txi%8
  qh=blocks[hbase+64+2*qhi].cast(dtypes.uint32).bitwise_or(blocks[hbase+64+2*qhi+1].cast(dtypes.uint32).lshift(16))
  qshift=(txi.bitwise_and(8))>>2
  q0=ql.bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).lshift(4).bitwise_and(0x30303030))
  q1=ql.rshift(4).bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).bitwise_and(0x30303030))
  kq0=2*txi-txi%16
  staged=UOp.group(sh[st*76+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
    sh[st*76+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020)))).end(st)
  ready=UOp.barrier(UOp.group(staged))
  def group(g):
   av=native_fragment_x2(sh.after(ready),(lane&15)*76+g*4).bitcast(dtypes.char.vec(8))
   bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(b[(blk*256+g*16+4*lc+q)*8+lr] for q in range(4)))
   axes=(tuple((400+i,2) for i in range(3)),tuple((410+i,2) for i in range(2)),tuple((420+i,2) for i in range(2)))
   arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
   return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),arg)
  cs=[group(g) for g in range(16)]; acc=UOp.placeholder((4,),dtypes.float32,300,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.REG)
  init=UOp.group(*(acc[r].store(0.0) for r in range(4))); acc=acc.after(init); update=None
  for r in range(4):
   row=lr+8*(r>>1); col=2*lc+(r&1); base=(row*k_blocks+blk)*210; term=UOp.const(dtypes.float32,0.0)
   wd=blocks[(base+208)//2].bitcast(dtypes.half).cast(dtypes.float32)
   for p in range(8):
    s0=byte(base+192+2*p).cast(dtypes.char).cast(dtypes.float32); s1=byte(base+193+2*p).cast(dtypes.char).cast(dtypes.float32)
    term=term+(wd*dB[(blk*8+p)*8+col])*(s0*cs[2*p].gep(r).cast(dtypes.float32)+s1*cs[2*p+1].gep(r).cast(dtypes.float32))
   update=acc.after(blk if update is None else update)[r].store(acc.after(blk)[r]+term)
  done=update.end(blk)
  return UOp.sink(*(out[(lr+8*(r>>1))*8+2*lc+(r&1)].store(acc.after(done)[r]) for r in range(4)),
    arg=KernelInfo(name=f'nv_native_fragment_q6_packed_k{k_blocks*256}',opts_to_apply=()))
 return kernel

# Research-only CTA contract.  This is deliberately separate from the
# single-warp gate above: it records the pinned llama geometry without making
# it selectable by any production route.
Q6_CTA_K_BLOCKS = 2
Q6_CTA_ROWS, Q6_CTA_COLS = 128, 16
Q6_CTA_WARPS, Q6_CTA_LANES = 8, 32
Q6_CTA_WEIGHT_STRIDE = 76
Q6_CTA_SHARED_BYTES = Q6_CTA_ROWS * Q6_CTA_WEIGHT_STRIDE * 4

def q6_cta_ownership(warp:int, lane:int):
 """Return the pinned (row, column) ownership for one CTA lane."""
 if not (0 <= warp < Q6_CTA_WARPS and 0 <= lane < Q6_CTA_LANES):
  raise ValueError("warp/lane outside the 8x32 CTA")
 band, phase = warp >> 1, warp & 1
 lr, lc = lane >> 2, lane & 3
 return (band * 32 + lr, phase * 8 + 2 * lc)

def q6_cta_geometry():
 """Machine-readable research gate geometry; never used for route selection."""
 return {"block": (32, 8), "rows": Q6_CTA_ROWS, "cols": Q6_CTA_COLS,
         "k": Q6_CTA_K_BLOCKS * 256, "weight_stride": Q6_CTA_WEIGHT_STRIDE,
         "warps": Q6_CTA_WARPS, "lanes": Q6_CTA_LANES,
         "warp_rows": 16, "warp_band_rows": 32, "phase_cols": 8,
         "barriers": 1, "shared_bytes": Q6_CTA_SHARED_BYTES,
         "promotion": "research_only"}

def q6_packed_cta_kernel(out, blocks, b, dB, k_blocks:int, col_groups:int=1,
                         block_start:int=0, segment_blocks:int=None, total_k_blocks:int=None,
                         activation_stride:int=None, activation_offset:int|UOp=0, allocation_base:int=0,
                         register_base:int|None=None, axis_base:int=0):
 """Generated 128x(16*col_groups) eight-warp CTA gate matching llama warp ownership."""
 if k_blocks < 1 or col_groups < 1: raise ValueError("k_blocks and col_groups must be positive")
 if segment_blocks is None: segment_blocks=k_blocks
 if total_k_blocks is None: total_k_blocks=k_blocks
 # Stream-K owner scheduling supplies runtime scalar bounds. Retain the
 # static validation when they are compile-time integers, but let RANGE carry
 # a symbolic uniform endpoint for the generated owner loop.
 if isinstance(block_start,int) and isinstance(segment_blocks,int):
  if block_start < 0 or segment_blocks < 1 or block_start+segment_blocks > total_k_blocks:
   raise ValueError("invalid K-block segment")
 cols=16*col_groups
 if activation_stride is None: activation_stride=cols
 if register_base is None: register_base=allocation_base
 def access(buf, idx): return buf.index(idx) if buf.ndim == 0 else buf[idx]
 lid=UOp.special(256,"lidx0"); warp,lane=lid//32,lid%32; lr,lc=lane>>2,lane&3; band,phase=warp>>1,warp&1
 blk=UOp.range(segment_blocks,axis_base,axis_type=AxisType.REDUCE)
 abs_blk=(UOp.const(dtypes.int32,block_start) if isinstance(block_start,int) else block_start.cast(dtypes.int32))+blk
 sh=UOp.placeholder((Q6_CTA_ROWS*Q6_CTA_WEIGHT_STRIDE,),dtypes.uint32,500+allocation_base,
   addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.LOCAL)
 sr=UOp.range(16,axis_base+1,axis_type=AxisType.LOOP); srow=warp+8*sr; hbase=(srow*total_k_blocks+abs_blk)*105; txi=lane
 ql=access(blocks,hbase+2*txi).cast(dtypes.uint32).bitwise_or(access(blocks,hbase+2*txi+1).cast(dtypes.uint32).lshift(16))
 qhi=(txi//16)*8+txi%8; qh=access(blocks,hbase+64+2*qhi).cast(dtypes.uint32).bitwise_or(access(blocks,hbase+64+2*qhi+1).cast(dtypes.uint32).lshift(16))
 qshift=txi.bitwise_and(8)>>2
 q0=ql.bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).lshift(4).bitwise_and(0x30303030))
 q1=ql.rshift(4).bitwise_and(0x0f0f0f0f).bitwise_or(qh.rshift(qshift).bitwise_and(0x30303030)); kq0=2*txi-txi%16
 staged=UOp.group(sh[srow*76+kq0].store(packed_i8_sub(q0,UOp.const(dtypes.uint32,0x20202020))),
   sh[srow*76+kq0+16].store(packed_i8_sub(q1,UOp.const(dtypes.uint32,0x20202020)))).end(sr)
 ready=UOp.barrier(UOp.group(staged))
 def mma(cg,n,g):
  row0=band*32+n*16
  av=native_fragment_x2(sh.after(ready),(row0+(lane&15))*76+g*4).bitcast(dtypes.char.vec(8))
  bv=UOp(Ops.STACK,dtypes.char.vec(4),tuple(access(b,(abs_blk*256+g*16+4*lc+q)*activation_stride+activation_offset+cg*16+phase*8+lr) for q in range(4)))
  axes=(tuple((600+i,2) for i in range(3)),tuple((610+i,2) for i in range(2)),tuple((620+i,2) for i in range(2)))
  arg=("WMMA_8_16_16_signed_char_int",(8,16,16),dtypes.char,dtypes.int,"NV",32,axes,())
  return UOp(Ops.WMMA,dtypes.int.vec(4),(av,bv,UOp.const(dtypes.int.vec(4),0)),arg)
 cs=[[[mma(cg,n,g) for g in range(16)] for n in range(2)] for cg in range(col_groups)]
 acc=UOp.placeholder((8*col_groups,),dtypes.float32,700+register_base,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.REG)
 init=UOp.group(*(acc[i].store(0.0) for i in range(8*col_groups))); acc=acc.after(init); update=None
 for cg in range(col_groups):
  for n in range(2):
   for r in range(4):
    ai=cg*8+n*4+r; row=band*32+n*16+lr+8*(r>>1); col=cg*16+phase*8+2*lc+(r&1); base=(row*total_k_blocks+abs_blk)*210; term=UOp.const(dtypes.float32,0.0)
    wd=access(blocks,(base+208)//2).bitcast(dtypes.half).cast(dtypes.float32)
    for p in range(8):
     def byte(off): return access(blocks,off//2).cast(dtypes.uint32).rshift((off%2)*8).bitwise_and(255)
     s0=byte(base+192+2*p).cast(dtypes.char).cast(dtypes.float32); s1=byte(base+193+2*p).cast(dtypes.char).cast(dtypes.float32)
     term=term+(wd*access(dB,(abs_blk*8+p)*activation_stride+activation_offset+col))*(s0*cs[cg][n][2*p].gep(r).cast(dtypes.float32)+s1*cs[cg][n][2*p+1].gep(r).cast(dtypes.float32))
    update=acc.after(blk if update is None else update)[ai].store(acc.after(blk)[ai]+term)
 done=update.end(blk)
 return UOp.sink(*(access(out,(band*32+n*16+lr+8*(r>>1))*cols+cg*16+phase*8+2*lc+(r&1)).store(acc.after(done)[cg*8+n*4+r])
   for cg in range(col_groups) for n in range(2) for r in range(4)),
   arg=KernelInfo(name=f'nv_native_fragment_q6_cta_128x{cols}x{k_blocks*256}',opts_to_apply=()))

def q6_packed_cta_k512_kernel(out, blocks, b, dB): return q6_packed_cta_kernel(out,blocks,b,dB,Q6_CTA_K_BLOCKS)

__all__=['kernel','q6_two_k16_kernel','q6_packed_two_k16_kernel','q6_packed_k256_kernel','q6_packed_kblocks_kernel','q6_first_two_k16_numpy',
         'Q6_CTA_K_BLOCKS','Q6_CTA_ROWS','Q6_CTA_COLS','Q6_CTA_WARPS','Q6_CTA_LANES','Q6_CTA_WEIGHT_STRIDE','Q6_CTA_SHARED_BYTES',
         'q6_cta_ownership','q6_cta_geometry','q6_packed_cta_kernel','q6_packed_cta_k512_kernel']
