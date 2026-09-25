"""Nemotron-H model attention uses the batched sampler's rounding points."""
import unittest

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.llm.nemotron_h import _attention


def _mask(length: int) -> Tensor:
  return Tensor(np.triu(np.full((length, length), -np.inf, dtype=np.float32), 1)).reshape(1, 1, length, length)


class TestNemotronHAttentionRounding(unittest.TestCase):
  def test_float_keys_match_sdpa(self):
    Tensor.manual_seed(0)
    q, k, v = Tensor.randn(2, 4, 5, 8), Tensor.randn(2, 2, 5, 8), Tensor.randn(2, 2, 5, 8)
    expected = q.scaled_dot_product_attention(k, v, attn_mask=_mask(5), enable_gqa=True).numpy()
    np.testing.assert_allclose(_attention(q, k, v, _mask(5)).numpy(), expected, rtol=1e-5, atol=1e-5)

  def test_bf16_keys_round_query_and_weights(self):
    Tensor.manual_seed(1)
    q = Tensor.randn(1, 2, 3, 8)
    k, v = Tensor.randn(1, 1, 3, 8).cast(dtypes.bfloat16), Tensor.randn(1, 1, 3, 8).cast(dtypes.bfloat16)
    out = _attention(q, k, v, _mask(3)).numpy()
    bf = lambda t: t.cast(dtypes.bfloat16).cast(dtypes.float)
    qs = bf(q * (1 / 8 ** 0.5)).numpy()
    kf, vf = k.float().numpy()[0, 0], v.float().numpy()[0, 0]
    scores = qs[0] @ kf.T + _mask(3).numpy()[0]
    weights = bf(Tensor(np.exp(scores - scores.max(-1, keepdims=True)))).numpy()
    expected = (weights @ vf) / weights.sum(-1, keepdims=True)
    self.assertEqual(out.dtype, np.float32)
    np.testing.assert_allclose(out[0], expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
