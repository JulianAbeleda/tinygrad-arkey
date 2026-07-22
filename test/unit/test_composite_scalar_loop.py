import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention


def test_semantic_composite_scalar_loop_q16_kv16_hd64_cpu():
  rng = np.random.default_rng(23)
  qv = rng.standard_normal((1, 1, 16, 64), dtype=np.float32).astype(np.float16)
  kv = rng.standard_normal((1, 1, 16, 64), dtype=np.float32).astype(np.float16)
  vv = rng.standard_normal((1, 1, 16, 64), dtype=np.float32).astype(np.float16)

  got = shared_prefill_attention(Tensor(qv, dtype=dtypes.float16), Tensor(kv, dtype=dtypes.float16),
                                 Tensor(vv, dtype=dtypes.float16)).numpy().astype(np.float32)
  scores = qv.astype(np.float32) @ kv.astype(np.float32).swapaxes(-2, -1) / np.sqrt(64)
  weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
  expected = (weights / weights.sum(axis=-1, keepdims=True)) @ vv.astype(np.float32)
  np.testing.assert_allclose(got, expected, rtol=3e-2, atol=3e-2)
