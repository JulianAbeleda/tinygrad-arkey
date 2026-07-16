from tinygrad import dtypes
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.uop.ops import Ops, UOp


def test_vector_uop_metadata_has_deterministic_priority_fallback():
  left_arg, right_arg = UOp.const(dtypes.int.vec(8), 0), UOp.const(dtypes.int.vec(8), 1)
  left, right = UOp(Ops.NOOP, dtypes.void, arg=left_arg), UOp(Ops.NOOP, dtypes.void, arg=right_arg)
  sink = UOp.sink(left, right)

  first = linearize(sink)
  assert first == linearize(sink)
  assert set(first) == {left, right, sink}
