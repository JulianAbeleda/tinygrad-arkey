"""Generic scoped nested-reduction boundary, intentionally not attention-specific."""
import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import AccumulatorSlot, UOp, ScopedReduceSpec, CompositeInputSpec
from tinygrad.codegen.late.devectorizer import _partition_composite_sources
from tinygrad.codegen.late.devectorizer import lower_composite_accumulator, composite_reduce_state_adapter
from tinygrad.codegen.late.composite_combines import resolve_reduce_slot_tensor, resolve_composite_reduce_slot_prebufferize
from tinygrad.schedule.rangeify import cleanup_dead_axes
from tinygrad.uop.spec import spec_tensor, type_verify

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

def test_composite_slot_resolves_before_bufferize_only_with_tagged_provenance():
  src = Tensor.empty(1, 2, 3, dtype=dtypes.float32)
  red = src.uop.composite_reduce(AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "sum"), axis=(2,), slot_shapes=((1, 2),))
  lowered = UOp(Ops.TUPLE, dtypes.void, (UOp.placeholder((1, 2), dtypes.float32, 91),)).replace(tag=("composite_reduce", red.arg[0]))
  slot = UOp(Ops.REDUCE_SLOT, dtypes.float32, (lowered,), 0)
  resolved = resolve_composite_reduce_slot_prebufferize(slot)
  assert resolved is not None and resolved.shape == (1, 2)

def test_prebufferize_slot_resolver_leaves_ordinary_tuple_untouched():
  plain = UOp(Ops.TUPLE, dtypes.void, (UOp.const(dtypes.float32, 1.0),))
  assert resolve_composite_reduce_slot_prebufferize(UOp(Ops.REDUCE_SLOT, dtypes.float32, (plain,), 0)) is None

def test_prebufferize_rejects_untagged_index_view():
  base = UOp(Ops.TUPLE, dtypes.void, (UOp.placeholder((2,), dtypes.float32, 92),))
  idx = UOp(Ops.INDEX, dtypes.float32, (base, UOp.const(dtypes.weakint, 0)), None)
  assert resolve_composite_reduce_slot_prebufferize(UOp(Ops.REDUCE_SLOT, dtypes.float32, (idx,), 0)) is None

def test_spec_accepts_only_tagged_composite_index_slot_view():
  src = Tensor.empty(1, 2, 3, dtype=dtypes.float32)
  red = src.uop.composite_reduce(AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "sum"), axis=(2,), slot_shapes=((1, 2),))
  tup = UOp(Ops.TUPLE, dtypes.void, (UOp.placeholder((1, 2), dtypes.float32, 93),)).replace(tag=("composite_reduce", red.arg[0]))
  idx = UOp(Ops.INDEX, dtypes.float32, (tup, UOp.const(dtypes.weakint, 0)), None)
  type_verify(UOp(Ops.REDUCE_SLOT, dtypes.float32, (idx,), 0), spec_tensor)
  plain = UOp(Ops.TUPLE, dtypes.void, (UOp.const(dtypes.float32, 1.0),))
  untagged = UOp(Ops.INDEX, dtypes.float32, (plain, UOp.const(dtypes.weakint, 0)), None)
  with pytest.raises(RuntimeError, match="UOp verification failed"):
    type_verify(UOp(Ops.REDUCE_SLOT, dtypes.float32, (untagged,), 0), spec_tensor)

def test_composite_reduce_slot_constructor_carries_validated_provenance():
  src = Tensor.empty(1, 2, 3, dtype=dtypes.float32)
  red = src.uop.composite_reduce(AccumulatorSlot(Ops.ADD, dtypes.float32, 0.0, "sum"), axis=(2,), slot_shapes=((1, 2),))
  slot = red.composite_reduce_slot(0)
  assert slot.arg == 0 and slot.tag[0] == "composite_slot" and slot.tag[1] is red.arg[0]
  type_verify(slot, spec_tensor)


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

def test_scoped_reduce_rejects_generic_scoped_value_as_producer():
  """A logical carrier cannot silently become a scheduler-owned producer."""
  fallback = Tensor.empty(1, 1, 2, 3, dtype=dtypes.float32)
  logical = fallback.uop.scoped_value((0, 1, 2, 3))
  with pytest.raises(ValueError, match="explicit computation"):
    fallback.uop.scoped_reduce(logical, axis=(3,), axis_maps=((0, 1, 2, 3),), scope_owner=3)

def test_scoped_reduce_accepts_explicit_inner_matmul_producer():
  """An ordinary computation is a valid producer when ownership is explicit."""
  lhs = Tensor.empty(1, 1, 2, 4, dtype=dtypes.float32)
  rhs = Tensor.empty(1, 1, 3, 4, dtype=dtypes.float32)
  inner = lhs @ rhs.transpose(-2, -1)
  fallback = inner.sum(axis=3)
  scoped = fallback.uop.scoped_reduce(inner.uop, axis=(3,), axis_maps=((0, 1, 2, 3),), scope_owner=3)
  assert scoped.op is Ops.SCOPED_REDUCE and scoped.src[1] is inner.uop


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

def test_composite_accumulator_carrier_preserves_scalar_and_vector_slots():
  """The first-class carrier keeps m/l scalar and acc lanes explicit."""
  state = UOp(Ops.COMPOSITE_ACCUMULATOR, dtypes.float32.vec(2),
              (UOp.const(dtypes.float32, 1.0), UOp.const(dtypes.float32, 2.0)),
              ((1,), (1,), (2,)))
  assert state.shape == ((1,), (1,), (2,))
  assert state.op is Ops.COMPOSITE_ACCUMULATOR

def test_composite_accumulator_lowering_keeps_heterogeneous_slots():
  state = UOp(Ops.COMPOSITE_ACCUMULATOR, dtypes.float32.vec(2),
              (UOp.const(dtypes.float32, 1.0), UOp.const(dtypes.float32, 2.0),
               UOp.const(dtypes.float32.vec(2), (3.0, 4.0))),
              ((), (), (2,)))
  lowered = lower_composite_accumulator(state)
  assert lowered.op is Ops.TUPLE and lowered.dtype is dtypes.void
  assert tuple(x.dtype for x in lowered.src) == (dtypes.float32, dtypes.float32, dtypes.float32.vec(2))

def test_composite_reduce_state_adapter_is_opt_in_and_numeric_carrier_safe():
  vals = (UOp.const(dtypes.float32, 1.5), UOp.const(dtypes.float32, 2.0),
          UOp.const(dtypes.float32.vec(2), (3.0, 4.0)))
  state = composite_reduce_state_adapter(vals, ((), (), (2,)))
  lowered = lower_composite_accumulator(state)
  assert tuple(x.dtype for x in lowered.src) == (dtypes.float32, dtypes.float32, dtypes.float32.vec(2))

def test_composite_reduce_state_adapter_bounded_attention_graph():
  """The opt-in carrier can wrap a real bounded attention state without changing it."""
  from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
  q = Tensor([[[(1.0, 0.0), (0.0, 1.0)]]], dtype=dtypes.float32)
  k = Tensor([[[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]]], dtype=dtypes.float32)
  v = Tensor([[[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]]], dtype=dtypes.float32)
  out = shared_prefill_attention(q, k, v)
  # Keep the reference computation separate: the adapter is only a carrier
  # boundary and must never replace the proven scalar attention path.
  scores = q @ k.transpose(-1, -2) / (2 ** 0.5)
  m = scores.max(axis=-1, keepdim=True)
  weights = (scores - m).exp()
  l = weights.sum(axis=-1, keepdim=True)
  acc = weights @ v
  state = composite_reduce_state_adapter((m, l, acc), (m.shape, l.shape, acc.shape))
  lowered = lower_composite_accumulator(state)
  assert lowered.op is Ops.TUPLE
  assert tuple(x.shape for x in lowered.src) == (m.shape, l.shape, acc.shape)
  np.testing.assert_allclose(out.numpy(), (acc / l).numpy(), rtol=1e-5, atol=1e-5)

def test_composite_reduce_state_adapter_q16_hd64_fp16_numeric_gate():
  """The opt-in heterogeneous carrier scales to a representative prefill tile."""
  rng = np.random.default_rng(1)
  q_np = rng.standard_normal((1, 1, 16, 64)).astype(np.float16)
  k_np = rng.standard_normal((1, 1, 16, 64)).astype(np.float16)
  v_np = rng.standard_normal((1, 1, 16, 64)).astype(np.float16)
  q, k, v = (Tensor(x, device="CPU") for x in (q_np, k_np, v_np))
  out = q.scaled_dot_product_attention(k, v)
  scores = (q.cast(dtypes.float32) @ k.cast(dtypes.float32).transpose(-1, -2)) / (64 ** 0.5)
  m = scores.max(axis=-1, keepdim=True)
  weights = (scores - m).exp()
  l = weights.sum(axis=-1, keepdim=True)
  acc = weights @ v.cast(dtypes.float32)
  state = composite_reduce_state_adapter((m, l, acc), (m.shape, l.shape, acc.shape))
  lowered = lower_composite_accumulator(state)
  assert lowered.op is Ops.TUPLE
  assert tuple(x.shape for x in lowered.src) == (m.shape, l.shape, acc.shape)
  np.testing.assert_allclose(out.numpy(), (acc / l).numpy(), rtol=3e-2, atol=3e-2)
