from tinygrad import dtypes
from tinygrad.codegen.late.devectorizer import (_devec_distinct_reg_store, _manual_reduce_lanes,
                                                pm_reduce_acc_upcast_fix)
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite


def test_manual_reduce_lane_projection_is_output_major():
  inp = UOp.const(dtypes.float.vec(8), tuple(float(x) for x in range(8)))
  lanes = _manual_reduce_lanes(inp, Ops.ADD, 2)
  assert len(lanes) == 2 and all(x.dtype == dtypes.float for x in lanes)
  assert [x.simplify().arg for x in lanes] == [6.0, 22.0]


def test_distinct_vector_reg_store_reconstructs_scalar_lane_stores():
  reg = UOp.placeholder((4,), dtypes.float, 9700, addrspace=AddrSpace.REG)
  ptrs = tuple(reg.index(UOp.const(dtypes.weakint, i)) for i in range(4))
  tgt = UOp(Ops.STACK, ptrs[0].dtype.vec(4), ptrs)
  out = _devec_distinct_reg_store(tgt, UOp.const(dtypes.float.vec(4), (1., 2., 3., 4.)))
  assert out is not None and out.op is Ops.GROUP and len(out.src) == 4
  assert [x.src[0].src[1].arg for x in out.src] == list(range(4))
  assert all(x.src[1].dtype == dtypes.float for x in out.src)


def test_distinct_scalar_reg_store_is_unchanged_shape():
  reg = UOp.placeholder((1,), dtypes.float, 9701, addrspace=AddrSpace.REG)
  ptr = reg.index(UOp.const(dtypes.weakint, 0))
  tgt = UOp(Ops.STACK, ptr.dtype.vec(1), (ptr,))
  out = _devec_distinct_reg_store(tgt, UOp.const(dtypes.float, 3.0))
  assert out is not None and out.op is Ops.STORE and out.src[0] is ptr and out.src[1].dtype == dtypes.float


def test_manual_accumulator_widens_and_reduces_vector_lanes():
  reg = UOp.placeholder((1,), dtypes.float, 9702, addrspace=AddrSpace.REG)
  zero = UOp.const(dtypes.weakint, 0)
  slot = reg.index(zero)
  init = slot.broadcast(2).store(UOp.const(dtypes.float.vec(2), (0., 0.)))
  rr = UOp.range(4, 9703, AxisType.REDUCE)
  read = reg.after(rr).index(zero)
  update = slot.broadcast(4).store(read.broadcast(4) + UOp.const(dtypes.float.vec(4), (1., 2., 3., 4.)))
  end = update.end(rr)
  output = reg.after(end).index(zero).broadcast(2)
  rewritten = graph_rewrite(UOp.sink(init, end, output), pm_reduce_acc_upcast_fix)
  regs = [x for x in rewritten.toposort() if x.op is Ops.DEFINE_REG]
  stores = [x for x in rewritten.toposort() if x.op is Ops.STORE and regs[0] in x.src[0].backward_slice]
  assert len(regs) == 1 and regs[0].ptrdtype.size == 2
  assert stores and all(x.src[1].dtype == dtypes.float for x in stores)
  assert any(x.op is Ops.STACK and x.dtype == dtypes.float.vec(2) for x in rewritten.toposort())
