import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.codegen.late.flash_attn import merge_online_softmax_tile, normalize_online_softmax_state
from tinygrad.codegen.late.composite_combines import COMBINE_REGISTRY, _combine_step_online_softmax_state
from tinygrad.uop.ops import AccumulatorSlot, Ops, UOp, dtypes

class TestOnlineSoftmaxStateSplit(unittest.TestCase):
  def test_state_combine_is_registered_separately(self):
    self.assertIn("online_softmax_state", COMBINE_REGISTRY)
    self.assertIsNot(COMBINE_REGISTRY["online_softmax_state"], COMBINE_REGISTRY["online_softmax"])

  def test_lane_state_step_has_scalar_scalar_vector_abi(self):
    m, l, score = (UOp.const(dtypes.float32, x) for x in (0.5, 2.0, 1.0))
    acc = UOp.const(dtypes.float32.vec(2), (3.0, 4.0))
    value = UOp.const(dtypes.float32.vec(2), (5.0, 6.0))
    new_m, new_l, new_acc = _combine_step_online_softmax_state(m, l, acc, score, value)
    self.assertEqual((new_m.dtype.count, new_l.dtype.count, new_acc.dtype.count), (1, 1, 2))
    self.assertTrue(any(u.op is Ops.STACK for u in new_acc.toposort()))

  def test_normalization_of_raw_state_matches_attention(self):
    rng = np.random.default_rng(31)
    scores = rng.standard_normal((1, 1, 16), dtype=np.float32)
    values = rng.standard_normal((1, 16, 16), dtype=np.float32)
    m, l, acc = Tensor.full((1, 1, 1), -float("inf")), Tensor.zeros(1, 1, 1), Tensor.zeros(1, 1, 16)
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16)):
      m, l, acc = merge_online_softmax_tile(m, l, acc, Tensor(scores[..., lo:hi]), Tensor(values[:, lo:hi]))
    got = normalize_online_softmax_state(acc, l).numpy()
    p = np.exp(scores - scores.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, p @ values, rtol=1e-5, atol=1e-5)

  def test_kv3_hd2_state_abi_is_heterogeneous_and_numeric(self):
    scores = Tensor([[[0.5, -1.0, 2.0]]])
    values = Tensor([[[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]])
    slots = (AccumulatorSlot(Ops.MAX, scores.dtype, -float("inf"), "m"),
             AccumulatorSlot(Ops.ADD, scores.dtype, 0.0, "l"),
             AccumulatorSlot(Ops.ADD, scores.dtype, 0.0, "acc"))
    red = scores.uop.composite_reduce(*slots, axis=(2,), inputs=(values.uop,), combine_fn="online_softmax_state",
      slot_shapes=((1, 1), (1, 1), (1, 1, 2)), lane_shapes=((), (), (2,)))
    self.assertEqual(red.arg[0].lane_shapes, ((), (), (2,)))
    m, l, acc = Tensor.full((1, 1, 1), -float("inf")), Tensor.zeros(1, 1, 1), Tensor.zeros(1, 1, 2)
    m, l, acc = merge_online_softmax_tile(m, l, acc, scores, values)
    got = normalize_online_softmax_state(acc, l).numpy()
    score_np, value_np = scores.numpy(), values.numpy()
    weights = np.exp(score_np-score_np.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, weights @ value_np, rtol=1e-6, atol=1e-6)

  def test_q16_hd16_state_abi_and_cpu_numeric_gate(self):
    rng = np.random.default_rng(32)
    scores = Tensor(rng.standard_normal((1, 16, 16), dtype=np.float32))
    values = Tensor(rng.standard_normal((1, 16, 16), dtype=np.float32))
    m, l, acc = Tensor.full((1, 16, 1), -float("inf")), Tensor.zeros(1, 16, 1), Tensor.zeros(1, 16, 16)
    m, l, acc = merge_online_softmax_tile(m, l, acc, scores, values)
    got = normalize_online_softmax_state(acc, l).numpy()
    score_np, value_np = scores.numpy(), values.numpy()
    weights = np.exp(score_np-score_np.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, weights @ value_np, rtol=1e-5, atol=1e-5)

if __name__ == "__main__": unittest.main()
