import numpy as np
import pytest

from extra.qk.mmq_coop_tile import BOUNDED_SHAPE, compute_q4k_q8_1_coop_tile, owner_writeback
from extra.qk.mmq_q4k_q8_reference import (Q4KQ81MMQTileSpec, q8_1_mmq_ds4_from_row_major_reference,
                                          q4k_q8_1_mmq_ds4_tile_reference)
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words


def _inputs(seed=7):
  words, _ = _make_q4k_words(16, 256, seed)
  raw = words.numpy().view(np.uint8)
  x = np.random.default_rng(seed + 1).standard_normal((16, 256)).astype(np.float32)
  ds4 = q8_1_mmq_ds4_from_row_major_reference(x, np.ones((16, 8), np.float32))
  spec = Q4KQ81MMQTileSpec(role="test", m=16, n=16, k=256, m_tile=16, n_tile=16,
                           activation_layout="q8_1_mmq_ds4_transposed_blocks")
  return raw, ds4, spec


def test_coop_tile_matches_reference_and_stages_independently():
  raw, ds4, spec = _inputs()
  got = compute_q4k_q8_1_coop_tile(raw, ds4, spec)
  assert got.shape == BOUNDED_SHAPE[:2]
  assert np.isfinite(got).all()
  np.testing.assert_allclose(got, q4k_q8_1_mmq_ds4_tile_reference(raw, ds4, spec), rtol=1e-6, atol=1e-5)


def test_owner_writeback_is_exactly_owner_only():
  tile = np.arange(256, dtype=np.float32).reshape(16, 16)
  writes = []
  assert owner_writeback(tile, [(0, 1), (15, 15)], lambda m, n, v: writes.append((m, n, v))) == 2
  assert [(m, n) for m, n, _ in writes] == [(0, 1), (15, 15)]


def test_owner_writeback_rejects_duplicate_or_out_of_bounds_owner():
  tile = np.zeros((16, 16), dtype=np.float32)
  with pytest.raises(ValueError, match="duplicate"):
    owner_writeback(tile, [(1, 2), (1, 2)], lambda *_: None)
  with pytest.raises(ValueError, match="outside"):
    owner_writeback(tile, [(16, 0)], lambda *_: None)


@pytest.mark.parametrize("shape", [(8, 16, 256), (16, 16, 512)])
def test_coop_tile_fails_closed(shape):
  raw, ds4, spec = _inputs()
  bad = Q4KQ81MMQTileSpec(role="test", m=shape[0], n=shape[1], k=shape[2], m_tile=shape[0], n_tile=shape[1],
                          activation_layout="q8_1_mmq_ds4_transposed_blocks")
  with pytest.raises(ValueError): compute_q4k_q8_1_coop_tile(raw, ds4, bad)
