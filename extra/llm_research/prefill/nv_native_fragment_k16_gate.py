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
