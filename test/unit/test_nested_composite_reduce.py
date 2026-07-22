"""Generic scoped nested-reduction boundary, intentionally not attention-specific."""
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import AccumulatorSlot, UOp, ScopedReduceSpec, CompositeInputSpec
from tinygrad.codegen.late.devectorizer import _partition_composite_sources
from tinygrad.codegen.late.composite_combines import resolve_reduce_slot_tensor
from tinygrad.schedule.rangeify import cleanup_dead_axes

def test_lane_aware_composite_input_is_explicit_and_scalar_safe():
  """The grouped-load carrier is metadata; scalar source ownership is unchanged."""
  spec = CompositeInputSpec("logical", (0, 1, None, 3, 4), lane_axis=4, lane_group=16)
  assert spec.validate_lane_abi() is spec
  assert spec.axis_map[-1] == 4 and spec.lane_axis == 4 and spec.lane_group == 16
  value = Tensor.empty(1, 1, 16, 64, dtype=dtypes.float16)
  score = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
  red = score.uop.composite_reduce(
    AccumulatorSlot(Ops.MAX, dtypes.float32, float("-inf"), "m"),
    AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "l"),
    AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc"), axis=(3,),
    inputs=(value.uop,), input_specs=(spec,), combine_fn="online_softmax")
  assert red.arg[0].input_specs[0] == spec

def test_composite_state_slots_preserve_heterogeneous_logical_shapes():
  """State metadata keeps scalar m/l distinct from the logical Hd accumulator."""
  score = Tensor.empty(1, 1, 2, 3, 4, dtype=dtypes.float32)
  value = Tensor.empty(1, 1, 2, 3, 4, dtype=dtypes.float32)
  red = score.uop.composite_reduce(
    AccumulatorSlot(Ops.MAX, dtypes.float32, float("-inf"), "m"),
    AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "l"),
    AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc"), axis=(3,),
    inputs=(value.uop,), combine_fn="online_softmax",
    slot_shapes=((1, 1, 2), (1, 1, 2), (1, 1, 2, 4)))
  assert Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 0)).shape == (1, 1, 2)
  assert Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 1)).shape == (1, 1, 2)
  assert Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 2)).shape == (1, 1, 2, 4)


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

def test_scoped_reduce_keeps_producer_inputs_and_ssa_result_explicit():
  # Non-attention proof: inner dot product is consumed by an outer weighted
  # reduction. The fallback is ordinary tinygrad, while scoped IR records the
  # exact producer/input correspondence a fused backend must own.
  lhs = Tensor.empty(1, 1, 4, 8, dtype=dtypes.float32)
  rhs = Tensor.empty(1, 1, 6, 8, dtype=dtypes.float32)
  weights = Tensor.empty(1, 1, 4, 6, dtype=dtypes.float32)
  inner = lhs @ rhs.transpose(-2, -1)
  fallback = (inner * weights).sum(axis=3)
  scoped = fallback.uop.scoped_reduce(
    inner.uop, weights.uop, axis=(3,),
    axis_maps=((0, 1, 2, 3), (0, 1, 2, 3)), scope_owner=3,
    result_dtypes=(dtypes.float32, dtypes.float32))
  result = scoped.scoped_value(0)

  assert result.op is Ops.SCOPED_VALUE
  assert result.src == (scoped,)
  assert isinstance(scoped.arg, ScopedReduceSpec)
  assert scoped.src[1] is inner.uop
  assert scoped.src[2] is weights.uop
  assert scoped.arg.source_axis_maps == ((0, 1, 2, 3), (0, 1, 2, 3))
  assert scoped.arg.scope_owner == 3

  # The generic fallback reaches rangeify once; the semantic boundary cannot
  # recursively re-enter rangeify because it lowers directly to fallback.
  calls = Tensor(result).schedule_linear().src
  assert len(calls) == 1


def test_auxiliary_value_survives_one_owned_composite_reduction():
  """Decisive feasibility gate: inner dot + mapped value + outer state reduce."""
  rng = np.random.default_rng(0)
  lhs_np = rng.standard_normal((1, 1, 3, 4), dtype=np.float32)
  rhs_np = rng.standard_normal((1, 1, 5, 4), dtype=np.float32)
  value_np = rng.standard_normal((1, 1, 5, 1), dtype=np.float32)

  lhs, rhs, value = Tensor(lhs_np, device="CPU"), Tensor(rhs_np, device="CPU"), Tensor(value_np, device="CPU")
  score = lhs @ rhs.transpose(-2, -1)
  slots = (AccumulatorSlot(Ops.MAX, dtypes.float32, float("-inf"), "m"),
           AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "l"),
           AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc"))
  mapped_value = value.transpose(-2, -1).expand(*score.shape).uop.scoped_value((0, 1, None, 3))
  red = score.uop.composite_reduce(*slots, axis=(3,), inputs=(mapped_value,), combine_fn="online_softmax")
  l_val = Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 1))
  acc_val = Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 2))
  result = acc_val / l_val

  score_np = lhs_np @ np.swapaxes(rhs_np, -1, -2)
  weights = np.exp(score_np - score_np.max(axis=-1, keepdims=True))
  expected = (weights * value_np.swapaxes(-2, -1)).sum(axis=-1, keepdims=True) / weights.sum(axis=-1, keepdims=True)
  np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)

def test_composite_source_partition_excludes_range_carriers():
  """Rangeify-owned RANGE UOps never count as logical auxiliary inputs."""
  lhs = UOp.placeholder((2,), dtypes.float32, 0)
  value = UOp.placeholder((2,), dtypes.float32, 1)
  r0 = UOp.range(UOp.const(dtypes.weakint, 2), 7)
  red = lhs.composite_reduce(
    AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "acc"), axis=(0,),
    inputs=(value,), combine_fn="online_softmax")
  ranges, aux = _partition_composite_sources((r0, value), red.arg[0])
  assert ranges == (r0,)
  assert aux == (value,)

def test_cleanup_dead_axes_preserves_unranged_logical_lane():
  """Slot-specific logical lanes are not scheduler ranges and must survive cleanup."""
  src = UOp.placeholder((2, 4), dtypes.float32, 0)
  rng = UOp.range(UOp.const(dtypes.weakint, 2), 0)
  stage = src.bufferize(rng)
  cleaned = cleanup_dead_axes(stage)
  assert cleaned is not None
  assert cleaned.shape[-1] == 4
  assert cleaned.shape == stage.shape

def test_slot_projection_uses_carried_composite_metadata_for_vector_shape():
  """REDUCE_SLOT can recover a lane-shaped slot after REDUCE lowering."""
  slot = AccumulatorSlot(Ops.ADD, dtypes.float32.vec(4), 0.0, "acc")
  red = UOp.placeholder((1,), dtypes.float32, 0).composite_reduce(slot, axis=(0,), slot_shapes=((4,),))
  lowered = UOp(Ops.TUPLE, dtypes.void, (UOp.const(dtypes.float32, 1.0),)).replace(tag=("composite_reduce", red.arg[0]))
  projected = resolve_reduce_slot_tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32.vec(4), (lowered,), 0))
  assert projected.dtype == dtypes.float32.vec(4) and projected.shape == (4,)
