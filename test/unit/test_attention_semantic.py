import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import AttentionSpec
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention

class TestAttentionSemantic(unittest.TestCase):
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

  def test_generic_scheduler_reproduction_keeps_qk_and_pv_in_distinct_kernels(self):
    """Minimal generic multi-reduce scheduler reproduction.

    A QK reduction feeds both the online max and probability expressions. The
    current rangeify ownership rules realize it before the later PV reduction,
    so postrange receives one compatible contraction per kernel. This pins the
    boundary a future generic multi-reduction scheduler must remove.
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
    self.assertGreaterEqual(len(contraction_calls), 2)
    self.assertGreater(len(set(contraction_calls)), 1)

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

  def test_semantic_marker_fail_closes_to_ordinary_sdpa(self):
    attention = shared_prefill_attention(
      Tensor.empty(1, 1, 2, 4, dtype=dtypes.float32),
      Tensor.empty(1, 1, 65, 4, dtype=dtypes.float32),
      Tensor.empty(1, 1, 65, 4, dtype=dtypes.float32))
    linear = attention.schedule_linear()
    self.assertFalse(any(u.op is Ops.ATTENTION for u in linear.toposort()))

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
