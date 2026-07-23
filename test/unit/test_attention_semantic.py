import unittest
import os
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import AttentionSpec
from tinygrad.uop.ops import AxisType, CompositeReduce, UOp
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
from tinygrad.codegen.late.flash_attn import merge_online_softmax_tile
from tinygrad.schedule.rangeify import lower_attention_semantic

class TestAttentionSemantic(unittest.TestCase):
  @staticmethod
  def _combine_name():
    return "online_softmax_state" if os.getenv("TINYGRAD_ONLINE_SOFTMAX_STATE", "0") not in ("", "0") else "online_softmax"

  def test_opt_in_state_mode_small_hd16_numeric(self):
    if self._combine_name() != "online_softmax_state": self.skipTest("state combine is opt-in")
    rng = np.random.default_rng(7)
    q, k, v = (Tensor(rng.standard_normal(shape).astype(np.float16), dtype=dtypes.float16, device="CPU")
               for shape in ((1, 1, 16, 16), (1, 1, 16, 16), (1, 1, 16, 16)))
    out = shared_prefill_attention(q, k, v).numpy()
    ref = q.cast(dtypes.float32).scaled_dot_product_attention(k.cast(dtypes.float32), v.cast(dtypes.float32)).numpy()
    np.testing.assert_allclose(out, ref, rtol=3e-2, atol=3e-2)

  def test_opt_in_state_mode_uses_structured_composite_reduce(self):
    if self._combine_name() != "online_softmax_state": self.skipTest("state combine is opt-in")
    q = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    k = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    v = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    primitive = lower_attention_semantic(shared_prefill_attention(q, k, v).uop)
    state_reduces = [u for u in primitive.toposort() if u.op is Ops.REDUCE and
                     isinstance(u.arg[0], CompositeReduce) and u.arg[0].combine_fn == "online_softmax_state"]
    self.assertEqual(len(state_reduces), 1)
    carrier = state_reduces[0].arg[0].tile_carrier
    self.assertEqual(carrier.typed_fragment_abi, "online_softmax_qk_pv_v1")
    self.assertEqual(carrier.fragment_abi()["lane_group"], 16)
    fragments = state_reduces[0].arg[0].tile_fragments
    self.assertEqual(len(fragments), 3)
    self.assertTrue(any(u.op is Ops.REDUCE for u in fragments[0].src[0].toposort()), "score fragment must own the real QK graph")
    self.assertTrue(any(u.op is Ops.BUFFER for u in fragments[1].src[0].toposort()), "value fragment must own the real V graph")
    self.assertFalse(any(u.op is Ops.COMPOSITE_ACCUMULATOR for u in primitive.toposort()))

  def test_opt_in_state_producer_is_dependency_only(self):
    if self._combine_name() != "online_softmax_state": self.skipTest("state combine is opt-in")
    q = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    k = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    v = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    calls = shared_prefill_attention(q, k, v).schedule_linear().src
    owners = [u for call in calls for u in call.src[0].toposort() if u.op is Ops.DEFERRED_REDUCE_OWNER]
    projections = [u for call in calls for u in call.src[0].toposort() if u.op is Ops.DEFERRED_REDUCE_SLOT]
    self.assertEqual((len(calls), len(owners), len(projections)), (1, 0, 1))
    self.assertIs(projections[0].src[0].op, Ops.REDUCE)
    self.assertFalse(any(u.op is Ops.REDUCE_SLOT for call in calls for u in call.src[0].toposort()))
    self.assertEqual(sum(u.op is Ops.STORE for u in calls[0].src[0].toposort()), 1)

  def test_opt_in_state_final_ir_consumes_shared_carrier(self):
    if self._combine_name() != "online_softmax_state": self.skipTest("state combine is opt-in")
    from tinygrad.codegen import full_rewrite_to_sink
    from tinygrad.device import Device
    rng = np.random.default_rng(11)
    q, k, v = (Tensor(rng.standard_normal((1, 1, 16, 16)).astype(np.float16), dtype=dtypes.float16, device="CPU") for _ in range(3))
    calls = shared_prefill_attention(q, k, v).schedule_linear().src
    compute = [call for call in calls if any(u.op is Ops.DEFERRED_REDUCE_SLOT for u in call.src[0].toposort())]
    self.assertEqual(len(compute), 1)
    ast = compute[0].src[0]
    final = full_rewrite_to_sink(ast, Device[compute[0].device].renderer, optimize=ast.tag is None)
    self.assertFalse(any(u.op in (Ops.DEFERRED_REDUCE_OWNER, Ops.DEFERRED_REDUCE_SLOT, Ops.TUPLE) for u in final.toposort()))
    lane_stores = [u for u in final.toposort() if u.op is Ops.STORE and u.src[0].op is Ops.STACK and u.src[-1].op is Ops.STACK]
    self.assertEqual(len(lane_stores), 1)
    accesses = lane_stores[0].src[0].src
    self.assertTrue(all(x.op is Ops.INDEX for x in accesses), "output STORE lanes must remain addresses, not loaded values")
    self.assertEqual(len({idx.src[-1].render() for idx in accesses}), 16)

  def test_opt_in_state_amd_output_stores_have_scalar_param_addresses(self):
    if self._combine_name() != "online_softmax_state": self.skipTest("state combine is opt-in")
    from tinygrad.codegen import full_rewrite_to_sink
    from tinygrad.device import Device
    rng = np.random.default_rng(13)
    q, k, v = (Tensor(rng.standard_normal((1, 1, 16, 16)).astype(np.float16), dtype=dtypes.float16, device="AMD") for _ in range(3))
    compute = [call for call in shared_prefill_attention(q, k, v).schedule_linear().src
               if any(u.op is Ops.DEFERRED_REDUCE_SLOT for u in call.src[0].toposort())]
    self.assertEqual(len(compute), 1)
    final = full_rewrite_to_sink(compute[0].src[0], Device["AMD"].renderer, optimize=compute[0].src[0].tag is None)
    output_stores = [u for u in final.toposort() if u.op is Ops.STORE and any(
      x.op is Ops.PARAM and getattr(x.arg, "slot", None) == 0 for x in u.src[0].backward_slice)]
    self.assertEqual(len(output_stores), 16)
    self.assertTrue(all(u.src[0].op is Ops.INDEX and len(u.src[0].src) == 2 for u in output_stores))

  def test_state_composite_carries_authoritative_kv_range_owner(self):
    if self._combine_name() != "online_softmax_state": self.skipTest("state combine is opt-in")
    from tinygrad.schedule.rangeify import get_kernel_graph
    q = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    k = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    v = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    graph = get_kernel_graph(UOp.sink(shared_prefill_attention(q, k, v).uop))
    reds = [u for u in graph.toposort() if u.op is Ops.REDUCE and isinstance(u.arg[0], CompositeReduce) and
            u.arg[0].combine_fn == "online_softmax_state"]
    self.assertEqual(len(reds), 1)
    red, owners = reds[0], reds[0].arg[0].reduce_range_axes
    self.assertEqual(len(owners), 1)
    ranges = [u for u in red.src[1:] if u.op is Ops.RANGE]
    kv = [u for u in ranges if u.arg[0] in owners]
    # Output/Hd ownership lives in the primary expression rather than the
    # REDUCE context sources.  It must remain a distinct LOOP even though it
    # has the same extent as KV and the nested QK contraction.
    output = [u for u in red.src[0].toposort() if u.op is Ops.RANGE and
              u.arg[0] not in owners and u.arg[1] is AxisType.LOOP and u.vmax == 15]
    self.assertEqual(len(kv), 1)
    self.assertEqual(len(output), 1)

  def test_ordinary_reduce_has_no_composite_range_owner(self):
    graph = Tensor.ones(16, 16).sum(axis=1).schedule_linear()
    reds = [u for u in graph.toposort() if u.op is Ops.REDUCE]
    self.assertTrue(all(not isinstance(u.arg[0], CompositeReduce) for u in reds))
  def test_shared_attention_keeps_all_tensor_dependencies(self):
    q = Tensor.empty(1, 2, 4, 8, dtype=dtypes.float16)
    k = Tensor.empty(1, 2, 4, 8, dtype=dtypes.float16)
    v = Tensor.empty(1, 2, 4, 8, dtype=dtypes.float16)
    mask = Tensor.empty(1, 1, 4, 4, dtype=dtypes.float16)
    out = shared_prefill_attention(q, k, v, mask=mask)
    self.assertIs(out.uop.op, Ops.ATTENTION)
    self.assertIsInstance(out.uop.arg, AttentionSpec)
    self.assertEqual(out.uop.src[2:], (q.uop, k.uop, v.uop, mask.uop))
    self.assertEqual(out.uop.arg.kv_block, 64)

  def test_unrelated_reduction_is_not_marked_attention(self):
    red = (Tensor.ones(4, 8, dtype=dtypes.float32) * 3).sum(axis=-1)
    self.assertNotEqual(red.uop.op, Ops.ATTENTION)

  def test_fp16_primitive_exposes_qk_and_pv_contractions(self):
    q = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    k = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    v = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    out = shared_prefill_attention(q, k, v)
    primitive = out.uop.src[1]
    def contraction_body(red):
      body = red.src[0]
      while body.op is Ops.CAST: body = body.src[0]
      return body
    contractions = [u for u in primitive.toposort()
                    if u.op is Ops.REDUCE and u.arg[0] is Ops.ADD and contraction_body(u).op is Ops.MUL]
    self.assertGreaterEqual(len(contractions), 2)
    # Both operands of each contraction are fp16, so the centralized AMD TC
    # matcher can select its standard fp16->fp32 WMMA descriptor.
    for contraction in contractions:
      self.assertEqual(tuple(x.dtype.scalar() for x in contraction_body(contraction).src), (dtypes.float16, dtypes.float16))

  def test_semantic_attention_scheduler_keeps_qk_and_pv_in_one_fused_call(self):
    """The admitted semantic shape is represented by one composite call.

    This is distinct from generic SDPA scheduling: the semantic attention
    boundary deliberately owns QK, online-softmax state, and PV together so
    score/probability buffers are not materialized.
    """
    q = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    k = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    v = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
    calls = shared_prefill_attention(q, k, v).schedule_linear().src
    def is_fp16_contraction(red):
      body = red.src[0]
      while body.op is Ops.CAST: body = body.src[0]
      return red.arg[0] is Ops.ADD and body.op is Ops.MUL and tuple(x.dtype.scalar() for x in body.src) == (dtypes.float16, dtypes.float16)
    contraction_calls = [i for i,call in enumerate(calls) if any(is_fp16_contraction(red)
      for red in call.src[0].toposort() if red.op is Ops.REDUCE)]
    self.assertEqual(len(contraction_calls), 1)

  def test_bounded_online_primitive_matches_attention(self):
    rng = np.random.default_rng(0)
    q = Tensor(rng.standard_normal((1, 2, 3, 4), dtype=np.float32))
    k = Tensor(rng.standard_normal((1, 2, 5, 4), dtype=np.float32))
    v = Tensor(rng.standard_normal((1, 2, 5, 4), dtype=np.float32))
    attention = shared_prefill_attention(q, k, v)
    # Exercise the bounded online primitive itself, not the semantic marker's
    # fail-closed ordinary-SDPA source.
    got = Tensor(attention.uop.src[1]).numpy()
    scores = q.numpy() @ np.swapaxes(k.numpy(), -1, -2) / np.sqrt(4)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = probs / probs.sum(axis=-1, keepdims=True) @ v.numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_plain_sdpa_matches_reference_before_semantic_lowering(self):
    rng = np.random.default_rng(11)
    qv = rng.standard_normal((1, 2, 3, 4), dtype=np.float32)
    kv = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
    vv = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
    got = Tensor(qv).scaled_dot_product_attention(Tensor(kv), Tensor(vv)).numpy()
    scores = qv @ np.swapaxes(kv, -1, -2) / np.sqrt(4)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = probs / probs.sum(axis=-1, keepdims=True) @ vv
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_bounded_primitive_merges_multiple_kv_blocks(self):
    """The bounded primitive remains correct when KV spans multiple blocks."""
    rng = np.random.default_rng(12)
    qv = rng.standard_normal((1, 1, 2, 4), dtype=np.float32)
    kv = rng.standard_normal((1, 1, 65, 4), dtype=np.float32)
    vv = rng.standard_normal((1, 1, 65, 4), dtype=np.float32)
    attention = shared_prefill_attention(Tensor(qv), Tensor(kv), Tensor(vv))
    got = Tensor(attention.uop.src[1]).numpy()
    scores = qv @ np.swapaxes(kv, -1, -2) / np.sqrt(4)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = probs / probs.sum(axis=-1, keepdims=True) @ vv
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_tile_state_merge_matches_single_pass_reference(self):
    rng = np.random.default_rng(123)
    scores = rng.standard_normal((1, 2, 7), dtype=np.float32)
    values = rng.standard_normal((1, 7, 4), dtype=np.float32)
    m, l = Tensor.full((1, 2, 1), -float("inf")), Tensor.zeros(1, 2, 1)
    acc = Tensor.zeros(1, 2, 4)
    for lo, hi in ((0, 3), (3, 5), (5, 7)):
      m, l, acc = merge_online_softmax_tile(m, l, acc, Tensor(scores[..., lo:hi]), Tensor(values[:, lo:hi]))
    got = (acc / l).numpy()
    p = np.exp(scores - scores.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True)
    expected = p @ values
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_semantic_marker_fail_closes_to_ordinary_sdpa(self):
    attention = shared_prefill_attention(
      Tensor.empty(1, 1, 2, 4, dtype=dtypes.float32),
      Tensor.empty(1, 1, 65, 4, dtype=dtypes.float32),
      Tensor.empty(1, 1, 65, 4, dtype=dtypes.float32))
    linear = attention.schedule_linear()
    self.assertFalse(any(u.op is Ops.ATTENTION for u in linear.toposort()))

  def test_bounded_semantic_admission_uses_scoped_composite_before_score_bufferization(self):
    q = Tensor.empty(1, 1, 2, 1, dtype=dtypes.float32)
    k = Tensor.empty(1, 1, 3, 1, dtype=dtypes.float32)
    v = Tensor.empty(1, 1, 3, 1, dtype=dtypes.float32)
    calls = shared_prefill_attention(q, k, v).schedule_linear().src
    reductions = [u for call in calls for u in call.src[0].toposort() if u.op is Ops.REDUCE]
    composite = [u for u in reductions if hasattr(u.arg[0], "combine_fn") and u.arg[0].combine_fn == self._combine_name()]
    self.assertEqual(len(composite), 1)
    self.assertEqual(len(composite[0].arg[0].input_specs), 1)
    self.assertEqual(composite[0].arg[0].input_specs[0].role, "logical")

  def test_bounded_semantic_admission_keeps_logical_hd_axis_and_fp16_inputs(self):
    for hd, dtype in ((64, dtypes.float32), (128, dtypes.float16)):
      q = Tensor.empty(1, 1, 2, hd, dtype=dtype)
      k = Tensor.empty(1, 1, 3, hd, dtype=dtype)
      v = Tensor.empty(1, 1, 3, hd, dtype=dtype)
      calls = shared_prefill_attention(q, k, v).schedule_linear().src
      composite = [u for call in calls for u in call.src[0].toposort()
                   if u.op is Ops.REDUCE and hasattr(u.arg[0], "combine_fn") and u.arg[0].combine_fn == self._combine_name()]
      self.assertEqual(len(composite), 1)
      self.assertEqual(composite[0].arg[0].input_specs[0].axis_map, (0, 1, None, 3, 4))

  def test_bounded_semantic_admission_inlines_qk_under_composite_call(self):
    q = Tensor.empty(1, 1, 2, 64, dtype=dtypes.float16)
    k = Tensor.empty(1, 1, 3, 64, dtype=dtypes.float16)
    v = Tensor.empty(1, 1, 3, 64, dtype=dtypes.float16)
    calls = shared_prefill_attention(q, k, v).schedule_linear().src
    self.assertEqual(len(calls), 1)
    self.assertFalse(calls[0].ranges)
    reductions = [u for u in calls[0].src[0].toposort() if u.op is Ops.REDUCE]
    self.assertTrue(any(r.arg[0] is Ops.ADD for r in reductions))
    self.assertTrue(any(hasattr(r.arg[0], "combine_fn") and r.arg[0].combine_fn == self._combine_name() for r in reductions))

  def test_wmma_shape_semantic_admission_keeps_qk_and_composite_in_one_call(self):
    q = Tensor.empty(1, 1, 16, 64, dtype=dtypes.float16)
    k = Tensor.empty(1, 1, 16, 64, dtype=dtypes.float16)
    v = Tensor.empty(1, 1, 16, 64, dtype=dtypes.float16)
    calls = shared_prefill_attention(q, k, v).schedule_linear().src
    self.assertEqual(len(calls), 1)
    self.assertFalse(calls[0].ranges)
    reductions = [u for u in calls[0].src[0].toposort() if u.op is Ops.REDUCE]
    self.assertTrue(any(r.arg[0] is Ops.ADD for r in reductions))
    self.assertTrue(any(hasattr(r.arg[0], "combine_fn") and r.arg[0].combine_fn == self._combine_name() for r in reductions))

  def test_bounded_semantic_admission_handles_gqa_and_additive_mask(self):
    rng = np.random.default_rng(21)
    qv = rng.standard_normal((1, 2, 2, 64), dtype=np.float32)
    kv = rng.standard_normal((1, 1, 3, 64), dtype=np.float32)
    vv = rng.standard_normal((1, 1, 3, 64), dtype=np.float32)
    mask = rng.standard_normal((1, 1, 2, 3), dtype=np.float32)
    got = shared_prefill_attention(Tensor(qv), Tensor(kv), Tensor(vv), mask=Tensor(mask)).numpy()
    kk, vv = np.repeat(kv, 2, axis=1), np.repeat(vv, 2, axis=1)
    scores = qv @ kk.swapaxes(-2, -1) / np.sqrt(64) + mask
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True)); expected = probs / probs.sum(axis=-1, keepdims=True) @ vv
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_bounded_semantic_admission_handles_causal_mask_source(self):
    rng = np.random.default_rng(22)
    qv = rng.standard_normal((1, 1, 2, 64), dtype=np.float32)
    kv = rng.standard_normal((1, 1, 3, 64), dtype=np.float32)
    vv = rng.standard_normal((1, 1, 3, 64), dtype=np.float32)
    got = shared_prefill_attention(Tensor(qv), Tensor(kv), Tensor(vv), causal=True).numpy()
    expected = Tensor(qv).scaled_dot_product_attention(Tensor(kv), Tensor(vv), is_causal=True).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_prefill_shape_semantic_lowering_does_not_promote_block_materialization(self):
    q = Tensor.empty(1, 32, 512, 128, dtype=dtypes.float16)
    k = Tensor.empty(1, 32, 512, 128, dtype=dtypes.float16)
    v = Tensor.empty(1, 32, 512, 128, dtype=dtypes.float16)
    fallback = shared_prefill_attention(q, k, v)
    selected = shared_prefill_attention(
      Tensor.empty(1, 32, 512, 128, dtype=dtypes.float16),
      Tensor.empty(1, 32, 512, 128, dtype=dtypes.float16),
      Tensor.empty(1, 32, 512, 128, dtype=dtypes.float16))
    fallback_calls = len(Tensor(fallback.uop.src[0]).schedule_linear().src)
    selected_calls = len(selected.schedule_linear().src)
    # The candidate primitive currently expands to 84 calls at this shape;
    # retain SDPA until generic tiled lowering removes that materialization.
    self.assertEqual(selected_calls, fallback_calls)

  def test_shared_primitive_matches_grouped_query_attention(self):
    rng = np.random.default_rng(1)
    qv = rng.standard_normal((1, 4, 3, 4), dtype=np.float32)
    kv = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
    vv = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
    got = shared_prefill_attention(Tensor(qv), Tensor(kv), Tensor(vv)).numpy()
    k_expanded, v_expanded = np.repeat(kv, 2, axis=1), np.repeat(vv, 2, axis=1)
    scores = qv @ np.swapaxes(k_expanded, -1, -2) / np.sqrt(4)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = probs / probs.sum(axis=-1, keepdims=True) @ v_expanded
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_fp16_tile_attention_is_exact_under_tc_optimizer_boundary(self):
    """The composite path stays numerically correct when TC_OPT is enabled.

    The composite WMMA boundary is deliberately fail-closed today: QK is
    consumed as scalar scores by online-softmax.  This gate protects that
    contract while standalone matmul WMMA remains covered by the AMD ISA
    compilation tests.
    """
    rng = np.random.default_rng(7)
    qv = rng.standard_normal((1, 1, 16, 64), dtype=np.float32)
    kv = rng.standard_normal((1, 1, 16, 64), dtype=np.float32)
    vv = rng.standard_normal((1, 1, 16, 64), dtype=np.float32)
    got = shared_prefill_attention(Tensor(qv, dtype=dtypes.float16),
                                   Tensor(kv, dtype=dtypes.float16),
                                   Tensor(vv, dtype=dtypes.float16)).numpy()
    scores = qv @ np.swapaxes(kv, -1, -2) / np.sqrt(64)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = probs / probs.sum(axis=-1, keepdims=True) @ vv
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)
