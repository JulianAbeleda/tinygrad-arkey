import unittest

from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import AccumulatorSlot, CompositeReduce, KernelInfo, Ops, AxisType, UOp


class TestCompositeAxisConstraints(unittest.TestCase):
  def test_composite_state_ranges_exclude_reduction_axis(self):
    out = UOp.range(16, 100, AxisType.LOOP)
    red_axis = UOp.range(16, 101, AxisType.REDUCE)
    inp = UOp.const(dtypes.float32, 1.0)
    composite = CompositeReduce((AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "state"),), "online_softmax")
    red = UOp(Ops.REDUCE, dtypes.float32, (inp, out, red_axis), (composite, ()))
    scheduler = Scheduler(UOp.sink(red).replace(arg=KernelInfo()), None)
    self.assertEqual(scheduler.composite_state_ranges, frozenset({out}))
    self.assertNotIn(red_axis, scheduler.composite_state_ranges)
    with self.assertRaises(KernelOptError): scheduler.apply_opt(Opt(OptOps.UPCAST, 0, 4))

  def test_ordinary_reduce_has_no_state_constraint(self):
    out = UOp.range(8, 110, AxisType.LOOP)
    red_axis = UOp.range(8, 111, AxisType.REDUCE)
    inp = UOp.const(dtypes.float32, 1.0)
    red = UOp(Ops.REDUCE, dtypes.float32, (inp, out, red_axis), (Ops.ADD, ()))
    scheduler = Scheduler(UOp.sink(red).replace(arg=KernelInfo()), None)
    self.assertEqual(scheduler.composite_state_ranges, frozenset())


if __name__ == "__main__": unittest.main()
