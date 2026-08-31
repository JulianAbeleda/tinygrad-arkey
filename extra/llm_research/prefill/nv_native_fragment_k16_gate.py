"""Generated-UOp K16 native fragment gate."""
import numpy as np
import struct
from tinygrad import Tensor,dtypes,Device
from tinygrad.codegen.late.native_fragment import native_fragment_x2
from tinygrad.uop.ops import AxisType,KernelInfo,Ops,UOp
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

def q6_packed_k256_kernel(out, dot, blocks, b, dB):
 """Full one-block Q6_K K=256 gate, unrolled as sixteen K16 MMAs."""
 lane=UOp.special(32,"lidx0"); lr,lc=lane>>2,lane&3
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
  ready=UOp.barrier(UOp.group(*stores)); av=native_fragment_x2(sh.after(ready),(lane&15)*4+(lane>>4)*2).bitcast(dtypes.char.vec(8))
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
  writes.append(dot[idx].store(acc))
  writes.append(out[idx].store(acc))
 return UOp.sink(*writes,arg=KernelInfo(name='nv_native_fragment_q6_packed_k256',opts_to_apply=()))

def q6_packed_kblocks_kernel(k_blocks:int):
 """Looped 16x8 Q6_K x Q8 body for an integral number of canonical K=256 blocks."""
 if k_blocks < 1: raise ValueError("k_blocks must be positive")
 def kernel(out, blocks, b, dB):
  lane=UOp.special(32,"lidx0"); lr,lc=lane>>2,lane&3; blk=UOp.range(k_blocks,0,axis_type=AxisType.REDUCE)
  def byte(off): return blocks[off//2].cast(dtypes.uint32).rshift((off%2)*8).bitwise_and(255)
  sh=UOp.placeholder((16*76,),dtypes.uint32,200,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.LOCAL)
  st=UOp.range(32,1,axis_type=AxisType.LOOP); z=lane+32*st; srow=z//64; word=z%64; g=word//4; win=word%4
  pgrp=g%8; base=(srow*k_blocks+blk)*210; ql_off=(g//8)*64+(g%4)*16; qh_off=(g//8)*32+(g%2)*16
  vals=[]
  for q in range(4):
   lo=byte(base+ql_off+4*win+q); hi=byte(base+128+qh_off+4*win+q)
   vals.append(lo.rshift((pgrp//4)*4).bitwise_and(15).bitwise_or(hi.rshift((pgrp//2)*2).bitwise_and(3).lshift(4)).alu(
     Ops.SUB,UOp.const(dtypes.uint32,32)).cast(dtypes.char))
  staged=sh[srow*76+word].store(UOp(Ops.STACK,dtypes.char.vec(4),tuple(vals)).bitcast(dtypes.uint32)).end(st)
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

__all__=['kernel','q6_two_k16_kernel','q6_packed_two_k16_kernel','q6_packed_k256_kernel','q6_packed_kblocks_kernel','q6_first_two_k16_numpy']
