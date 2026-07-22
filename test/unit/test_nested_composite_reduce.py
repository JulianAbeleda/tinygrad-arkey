"""Generic nested-reduction boundary: an inner contraction feeds one stateful outer reduction."""
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import AccumulatorSlot, UOp


def test_nested_reduction_with_logical_element_input_stays_in_one_schedule():
  # This is intentionally not attention-specific. `score` is an inner
  # contraction over D; the composite reduction consumes score and a second
  # KV-preserving logical element in one outer KV reduction.
  lhs = Tensor.empty(1, 1, 4, 8, dtype=dtypes.float16)
  rhs = Tensor.empty(1, 1, 6, 8, dtype=dtypes.float16)
  value = Tensor.empty(1, 1, 6, 1, dtype=dtypes.float16)
  score = lhs @ rhs.transpose(-2, -1)
  slot_m = AccumulatorSlot(Ops.MAX, dtypes.float32, float("-inf"), "m")
  slot_l = AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "l")
  slot_acc = AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc")
  red = score.cast(dtypes.float32).uop.composite_reduce(
    slot_m, slot_l, slot_acc, axis=(3,), inputs=(value.cast(dtypes.float32).uop,), combine_fn="online_softmax")
  acc = Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 2))
  calls = acc.schedule_linear().src
  assert len(calls) == 1
  assert sum(u.op is Ops.REDUCE for u in calls[0].src[0].toposort()) >= 2
