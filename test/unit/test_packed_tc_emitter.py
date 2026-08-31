from tinygrad.codegen.opt.postrange import _build_q4_q6_tc_uops
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

class _TC:
  dtype_out=dtypes.int
  elements_per_thread=(1,1,8)

def test_full_range_emitter_preserves_zero_seeded_wmma():
  src=(UOp.const(dtypes.int.vec(8),1),UOp.const(dtypes.int.vec(8),2))
  paired,out=_build_q4_q6_tc_uops(tc=_TC(),wmma_srcs=src,tc_upcast_axes=((),(),()),wmma_arg=("gate",),bounded_k_carriers=None)
  assert paired is None and out.op is Ops.UNROLL
  wmma=out.src[0]
  assert wmma.op is Ops.WMMA and wmma.src[:2] == src and wmma.src[2] == UOp.const(dtypes.int.vec(8),0)

def test_bounded_q6_carriers_pass_through_without_rebuild():
  low,high=UOp.const(dtypes.int,3),UOp.const(dtypes.int,4)
  paired,out=_build_q4_q6_tc_uops(tc=_TC(),wmma_srcs=(low,high),tc_upcast_axes=((),(),()),wmma_arg=(),bounded_k_carriers=(low,high))
  assert paired == (low,high) and out is low
