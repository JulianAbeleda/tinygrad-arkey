"""Generated-UOp K16 native fragment gate."""
import numpy as np
from tinygrad import Tensor,dtypes,Device
from tinygrad.codegen.late.native_fragment import native_fragment_x2
from tinygrad.uop.ops import KernelInfo,Ops,UOp
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

__all__=['kernel','q6_two_k16_kernel']
