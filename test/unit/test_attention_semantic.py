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

  def test_bounded_online_primitive_matches_attention(self):
    rng = np.random.default_rng(0)
    q = Tensor(rng.standard_normal((1, 2, 3, 4), dtype=np.float32))
    k = Tensor(rng.standard_normal((1, 2, 5, 4), dtype=np.float32))
    v = Tensor(rng.standard_normal((1, 2, 5, 4), dtype=np.float32))
    got = shared_prefill_attention(q, k, v).numpy()
    scores = q.numpy() @ np.swapaxes(k.numpy(), -1, -2) / np.sqrt(4)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = probs / probs.sum(axis=-1, keepdims=True) @ v.numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
