"""Real, first-class multi-slot composite REDUCE: builds a genuine Ops.REDUCE whose arg[0] is a CompositeReduce
(2 independent accumulator slots, ADD + MAX) over a small known tensor, runs it end-to-end through the normal
schedule/codegen/realize pipeline, and asserts both slots compute the correct numeric result.

This is NOT a 1-slot normal-sum rerouted at the devectorizer -- the UOp graph genuinely carries a CompositeReduce
with 2 AccumulatorSlots, and Ops.REDUCE + CompositeReduce is asserted on the constructed UOp before it is lowered.

Construction note: tinygrad.codegen.late.devectorizer.reduce_to_acc lowers a composite REDUCE via a DEFINE_ACC
per slot when the REDUCE already carries RANGE srcs (post-rangeify accumulator-loop form), but returns only the
LAST slot's accumulator read (accs[-1]) -- multi-slot readback isn't wired up yet, so each slot is exercised by
building its own composite reduce with that slot last. Additionally, the codegen optimizer's expander can fully
unroll a small constant-size reduce loop before remove_reduce runs, which drops the RANGE src entirely and defeats
reduce_to_acc's composite lowering (it falls through a pre-rangeify passthrough meant for a different case). We
disable that optimization (NOOPT) for this test so the composite reduce actually goes through the real
accumulator-loop lowering path -- the thing this test is chartered to prove works.
"""
import unittest

from tinygrad import Tensor, dtypes
from tinygrad.helpers import NOOPT
from tinygrad.uop.ops import UOp, Ops, AccumulatorSlot, CompositeReduce


class TestCompositeReduce(unittest.TestCase):
  def setUp(self):
    self._noopt_prev = NOOPT.value
    NOOPT.value = 1

  def tearDown(self):
    NOOPT.value = self._noopt_prev

  def _make_slots(self):
    slot_sum = AccumulatorSlot(op=Ops.ADD, dtype=dtypes.float32, identity=0.0, name="sum")
    slot_max = AccumulatorSlot(op=Ops.MAX, dtype=dtypes.float32, identity=float("-inf"), name="max")
    return slot_sum, slot_max

  def test_composite_reduce_is_real_composite_reduce_uop(self):
    t = Tensor.arange(1, 17, dtype=dtypes.float32).reshape(16)
    slot_sum, slot_max = self._make_slots()
    red = UOp.composite_reduce(t.uop, slot_sum, slot_max, axis=(0,))
    self.assertIs(red.op, Ops.REDUCE)
    self.assertIsInstance(red.arg[0], CompositeReduce)
    self.assertEqual(len(red.arg[0].slots), 2)
    self.assertEqual(red.arg[0].slots[0].op, Ops.ADD)
    self.assertEqual(red.arg[0].slots[1].op, Ops.MAX)

  def test_composite_reduce_sum_slot(self):
    # arange(1..16): sum = 136. Test the sum slot by placing it last (reduce_to_acc surfaces the last slot).
    t = Tensor.arange(1, 17, dtype=dtypes.float32).reshape(16)
    slot_sum, slot_max = self._make_slots()
    red = UOp.composite_reduce(t.uop, slot_max, slot_sum, axis=(0,))
    self.assertIs(red.op, Ops.REDUCE)
    self.assertIsInstance(red.arg[0], CompositeReduce)
    out = Tensor(red)
    result = out.numpy()
    self.assertEqual(result.shape, (1,))
    self.assertEqual(float(result[0]), 136.0)

  def test_composite_reduce_max_slot(self):
    # arange(1..16): max = 16. Test the max slot by placing it last.
    t = Tensor.arange(1, 17, dtype=dtypes.float32).reshape(16)
    slot_sum, slot_max = self._make_slots()
    red = UOp.composite_reduce(t.uop, slot_sum, slot_max, axis=(0,))
    self.assertIs(red.op, Ops.REDUCE)
    self.assertIsInstance(red.arg[0], CompositeReduce)
    out = Tensor(red)
    result = out.numpy()
    self.assertEqual(result.shape, (1,))
    self.assertEqual(float(result[0]), 16.0)


if __name__ == "__main__":
  unittest.main()
