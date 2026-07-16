import numpy as np

from extra.qk.layout import q6_k_reference
from extra.qk.prefill.q6_direct_packed_qualification import make_finite_q6k_bytes, q6k_dequantize_selected_positions
from tinygrad import Tensor


def test_selected_q6_decoder_matches_canonical_reference():
  n, k = 3, 512
  raw = make_finite_q6k_bytes(n, k, 7)
  positions = np.array([0, 15, 16, 63, 127, 128, 191, 255, 256, 511])
  got = q6k_dequantize_selected_positions(raw, positions)
  reference = q6_k_reference(Tensor(raw.reshape(-1)), n*k).reshape(n, k).numpy()[:, positions]
  np.testing.assert_array_equal(got, reference)


def test_q6_fixture_is_deterministic_and_finite():
  first = make_finite_q6k_bytes(2, 512, 3); second = make_finite_q6k_bytes(2, 512, 3)
  np.testing.assert_array_equal(first, second)
  assert np.isfinite(q6k_dequantize_selected_positions(first, np.arange(512))).all()
