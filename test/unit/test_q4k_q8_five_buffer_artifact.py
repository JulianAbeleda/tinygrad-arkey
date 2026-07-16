import numpy as np
import pytest

from extra.qk.mmq_q4k_q8_reference import (Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q8_1_MMQ_DS4_LAYOUT,
  describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_ds4_tile_reference)
from extra.qk.prefill.q4k_q8_five_buffer_artifact import (BUFFER_NAMES, build_q4k_q8_five_buffer_artifact,
  save_q4k_q8_five_buffer_artifact)
from extra.qk.q4k_q8_fixture import make_finite_q4k_bytes, q4k_dequantize_selected_positions


def test_selected_q4k_dequant_is_vectorized_and_exact():
  raw = make_finite_q4k_bytes(16, 512, 7)
  positions = np.array([511, 0, 31, 32, 255, 256, 31])
  got = q4k_dequantize_selected_positions(raw, positions)
  from tinygrad import Tensor
  from extra.qk.layout import q4_k_reference
  dense = q4_k_reference(Tensor(raw.reshape(-1).copy()), 16 * 512).reshape(16, 512).numpy()
  np.testing.assert_array_equal(got, dense[:, positions])


@pytest.mark.parametrize("shape", [(16, 16, 256), (32, 32, 512)])
def test_artifact_matches_canonical_full_reference_and_saves_exact_buffers(tmp_path, shape):
  m, n, k = shape
  artifact = build_q4k_q8_five_buffer_artifact(m, n, k, seed=23)
  q4 = artifact.q4_packed_words.view(np.uint8).reshape(n, k // 256, 144)
  ds4 = Q81MMQDS4Activation(artifact.q8_ds4_values.reshape(k // 128, m, 128),
    artifact.q8_scales.reshape(k // 128, m, 4), artifact.q8_weighted_sums.reshape(k // 128, m, 4),
    Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
  spec = describe_q4k_q8_1_mmq_tile(role="bounded_test", m=m, n=n, k=k, m_tile=m, n_tile=n,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  canonical = q4k_q8_1_mmq_ds4_tile_reference(q4, ds4, spec)
  assert artifact.reference.shape == (m, n)
  np.testing.assert_array_equal(artifact.reference, canonical)
  out = tmp_path / "five.npz"
  metadata = save_q4k_q8_five_buffer_artifact(out, artifact)
  with np.load(out, allow_pickle=False) as saved:
    assert tuple(saved.files) == BUFFER_NAMES
    for name in BUFFER_NAMES:
      np.testing.assert_array_equal(saved[name], getattr(artifact, name))
      assert metadata["buffers"][name]["nbytes"] == saved[name].nbytes


@pytest.mark.parametrize("shape", [(0, 16, 256), (15, 16, 256), (16, 17, 256), (16, 16, 128)])
def test_artifact_rejects_nonpositive_or_unaligned_shapes(shape):
  with pytest.raises(ValueError, match="positive and aligned"):
    build_q4k_q8_five_buffer_artifact(*shape)
