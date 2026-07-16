import os

import numpy as np
import pytest

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Context
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.isa.amd import isel_store
from tinygrad.uop.ops import Ops, ScheduleHints, UOp


def test_nested_integer_and_declared_fp32_accumulator_initializers_keep_loop_order():
  order, pin, store = UOp(Ops.NOOP), UOp.const(dtypes.int32, 8).rtag(), UOp(Ops.STORE)
  ctx = IselContext(UOp.sink())
  integer = UOp(Ops.NOOP, dtypes.int32, src=(order, pin), arg=("wmma_acc", 1, 0, 0, 8))
  fixed = UOp(Ops.NOOP, dtypes.float32, src=(order, pin), arg=("fixed_acc", "add"))
  plain = UOp(Ops.NOOP, dtypes.float32, src=(order, pin), arg=("wmma_acc", 1, 0, 0, 8))
  assert isel_store(ctx, integer, UOp.const(dtypes.int32, 0), store).src[1] is order
  assert isel_store(ctx, fixed, UOp.const(dtypes.float32, 0), store).src[1] is order
  assert len(isel_store(ctx, plain, UOp.const(dtypes.float32, 0), store).src) == 1


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_signed_int8_wmma_operand_and_result_lane_mapping_on_hardware():
  rng = np.random.default_rng(0)
  a = rng.integers(-128, 128, size=(16, 16), dtype=np.int8)
  b = rng.integers(-128, 128, size=(16, 16), dtype=np.int8)
  reference = a.astype(np.int32) @ b.astype(np.int32)
  assert np.unique(reference).size == reference.size  # every wrong result-lane permutation is observable
  assert not np.array_equal(reference, b.astype(np.int32) @ a.astype(np.int32))
  assert not np.array_equal(reference, a.astype(np.uint8).astype(np.int32) @ b.astype(np.uint8).astype(np.int32))

  hint = ScheduleHints(name="signed_int8_wmma_hardware_canary",
                       opts_to_apply=(Opt(OptOps.TC, 0, (-1, 2, 1)),))
  def matmul(lhs, rhs): return lhs.matmul(rhs, dtype=dtypes.int32).contiguous(arg=hint)

  with Context(DEV="AMD:ISA"):
    probe = matmul(Tensor.empty(16, 16, dtype=dtypes.int8, device="AMD"),
                   Tensor.empty(16, 16, dtype=dtypes.int8, device="AMD"))
    sink = next(u for u in probe.schedule_linear().toposort() if u.op is Ops.SINK)
    program = to_program(sink, Device["AMD"].renderer)
    isa = [str(u.arg) for linear in program.src if linear.op is Ops.LINEAR for u in linear.src if not isinstance(u.arg, tuple)]
    wmma = [line for line in isa if line.startswith("v_wmma_i32_16x16x16_iu8")]
    assert len(wmma) == 1 and wmma[0].endswith(", 3)")
    got = matmul(Tensor(a, device="AMD"), Tensor(b, device="AMD")).realize().numpy()
  np.testing.assert_array_equal(got, reference)
