import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.codegen.late.flash_attn import merge_online_softmax_tile, normalize_online_softmax_state
from tinygrad.codegen.late.composite_combines import COMBINE_REGISTRY

class TestOnlineSoftmaxStateSplit(unittest.TestCase):
  def test_state_combine_is_registered_separately(self):
    self.assertIn("online_softmax_state", COMBINE_REGISTRY)
    self.assertIsNot(COMBINE_REGISTRY["online_softmax_state"], COMBINE_REGISTRY["online_softmax"])

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

if __name__ == "__main__": unittest.main()
