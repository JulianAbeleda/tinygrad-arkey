"""Research gate for the neutral four-register native fragment load."""
from tinygrad import dtypes
from tinygrad.codegen.late.native_fragment import PackedFragmentSpec, native_fragment_x2, native_fragment_x4, packed_fragment_load
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo,Ops,UOp

def emit_native_fragment_readback():
  def kernel(out:UOp, source:UOp) -> UOp:
    lane=UOp.special(32,"lidx0")
    shared=UOp.placeholder((128,),dtypes.uint32,20,addrspace=AddrSpace.LOCAL)
    stores=UOp.group(*(shared[lane+32*i].store(source[lane+32*i]) for i in range(4)))
    ready=UOp.barrier(stores)
    fragment=native_fragment_x4(shared.after(ready),(lane&15)*8+(lane>>4)*4)
    writes=tuple(out[lane*4+i].store(fragment.gep(i)) for i in range(4))
    return UOp.sink(*writes,arg=KernelInfo(name="nv_native_fragment_x4_readback",opts_to_apply=()))
  return kernel

def emit_native_fragment_imma():
  def kernel(out:UOp, source_a:UOp, source_b:UOp) -> UOp:
    lane=UOp.special(32,"lidx0")
    shared=UOp.placeholder((128,),dtypes.uint32,30,addrspace=AddrSpace.LOCAL)
    ready=UOp.barrier(UOp.group(*(shared[lane+32*i].store(source_a[lane+32*i]) for i in range(4))))
    a=native_fragment_x4(shared.after(ready),(lane&15)*8+(lane>>4)*4).bitcast(dtypes.char.vec(16))
    lr,lc=lane>>2,lane&3
    b=UOp(Ops.STACK,dtypes.char.vec(8),tuple(source_b[(4*(lc+4*r)+q)*8+lr] for r in range(2) for q in range(4)))
    axes=(tuple((100+i,2) for i in range(4)),tuple((110+i,2) for i in range(3)),tuple((120+i,2) for i in range(2)))
    arg=("WMMA_8_16_32_signed_char_int",(8,16,32),dtypes.char,dtypes.int,"NV",32,axes,())
    c=UOp(Ops.WMMA,dtypes.int.vec(4),(a,b,UOp.const(dtypes.int.vec(4),0)),arg)
    writes=tuple(out[(lr+8*(r>>1))*8+2*lc+(r&1)].store(c.gep(r)) for r in range(4))
    return UOp.sink(*writes,arg=KernelInfo(name="nv_native_fragment_x4_imma",opts_to_apply=()))
  return kernel

def emit_q6k_k64_fragment_readback():
  """Causal Q6 K64 packed-B load gate; arithmetic is deliberately identity."""
  def kernel(out:UOp, source:UOp) -> UOp:
    lane=UOp.special(32,"lidx0")
    shared=UOp.placeholder((128,),dtypes.uint32,20,addrspace=AddrSpace.LOCAL)
    ready=UOp.barrier(UOp.group(*(shared[lane+32*i].store(source[lane+32*i]) for i in range(4))))
    frag=packed_fragment_load(shared.after(ready),(lane&15)*8+(lane>>4)*4,PackedFragmentSpec.q6k_k64())
    writes=tuple(out[lane*2+i].store(frag.gep(i)) for i in range(2))
    return UOp.sink(*writes,arg=KernelInfo(name="nv_q6k_k64_fragment_readback",opts_to_apply=()))
  return kernel

__all__=["emit_native_fragment_imma","emit_native_fragment_readback","emit_q6k_k64_fragment_readback"]
