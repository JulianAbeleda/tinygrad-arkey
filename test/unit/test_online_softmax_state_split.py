import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.codegen.late.flash_attn import merge_online_softmax_tile, normalize_online_softmax_state
from tinygrad.codegen.late.composite_combines import (COMBINE_REGISTRY, _combine_step_online_softmax_state,
  _handle_no_range_generic, _pack_online_softmax_v_lanes)
from tinygrad.codegen.late.devectorizer import physical_composite_slot_dtype
from tinygrad.uop.ops import AccumulatorSlot, CompositeInputSpec, Ops, UOp, dtypes

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

  def test_physical_register_cardinality_is_1_1_hd(self):
    slots = (AccumulatorSlot(Ops.MAX, dtypes.float32, -float("inf"), "m"),
             AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "l"),
             AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc"))
    red = UOp.placeholder((16,), dtypes.float32, 0).composite_reduce(*slots, axis=(0,),
      combine_fn="online_softmax_state", lane_shapes=((), (), (16,)))
    self.assertEqual(tuple(physical_composite_slot_dtype(red.arg[0], i).count for i in range(3)), (1, 1, 16))

  def test_kv3_hd2_packing_keeps_distinct_lanes(self):
    flat = UOp.vectorize(*(UOp.const(dtypes.float32, x) for x in (1.0, 2.0, 3.0, -1.0, 0.5, 4.0)))
    packed = _pack_online_softmax_v_lanes(flat, 3, 2, dtypes.float32)
    self.assertEqual([[lane.simplify().arg for lane in vec.src] for vec in packed], [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])

  def test_kv3_hd2_direct_composite_numeric(self):
    score_vals, value_vals = (0.5, -1.0, 2.0), (1.0, 2.0, 3.0, -1.0, 0.5, 4.0)
    scores = UOp.vectorize(*(UOp.const(dtypes.float32, x) for x in score_vals))
    values = UOp.vectorize(*(UOp.const(dtypes.float32, x) for x in value_vals))
    slots = (AccumulatorSlot(Ops.MAX, dtypes.float32, -float("inf"), "m"),
             AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "l"),
             AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc"))
    red = scores.composite_reduce(*slots, axis=(), inputs=(values,), combine_fn="online_softmax_state",
      input_specs=(CompositeInputSpec("logical", lane_group=2),), lane_shapes=((), (), (2,)))
    _, den, acc = _handle_no_range_generic(scores, red.arg[0], red, (values,))
    got = np.array([acc.gep(i).alu(Ops.MUL, den.alu(Ops.RECIPROCAL)).simplify().arg for i in range(2)])
    weights = np.exp(np.array(score_vals)-max(score_vals)); weights /= weights.sum()
    np.testing.assert_allclose(got, weights @ np.array(value_vals).reshape(3, 2), rtol=1e-6, atol=1e-6)

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
    score_np = rng.standard_normal((1, 16, 16), dtype=np.float32)
    value_np = rng.standard_normal((1, 16, 16), dtype=np.float32)
    slots = (AccumulatorSlot(Ops.MAX, dtypes.float32, -float("inf"), "m"),
             AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "l"),
             AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc"))
    values = UOp.vectorize(*(UOp.const(dtypes.float32, float(x)) for x in value_np.reshape(-1)))
    rows = []
    for row in score_np.reshape(16, 16):
      scores = UOp.vectorize(*(UOp.const(dtypes.float32, float(x)) for x in row))
      red = scores.composite_reduce(*slots, axis=(), inputs=(values,), combine_fn="online_softmax_state",
        input_specs=(CompositeInputSpec("logical", lane_group=16),), lane_shapes=((), (), (16,)))
      _, den, acc = _handle_no_range_generic(scores, red.arg[0], red, (values,))
      rows.append([acc.gep(i).alu(Ops.MUL, den.alu(Ops.RECIPROCAL)).simplify().arg for i in range(16)])
    got = np.array(rows, dtype=np.float32).reshape(1, 16, 16)
    weights = np.exp(score_np-score_np.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, weights @ value_np, rtol=1e-5, atol=1e-5)

if __name__ == "__main__": unittest.main()
