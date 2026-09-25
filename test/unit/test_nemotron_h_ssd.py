"""The chunked SSD Mamba-2 scan equals `NemotronHMamba2.cached`, including a carried state and ragged lengths."""
import unittest

import numpy as np

from tinygrad import Tensor
from tinygrad.llm.nemotron_h_ssd import ssd_cached

from test.unit.test_nemotron_h_model import tiny_model


class TestNemotronHSSD(unittest.TestCase):
  def _check(self, lengths: tuple[int, ...], chunk: int, precision: str = "float", tol: float = 1e-4):
    model = tiny_model()
    block = model.blk[0]
    Tensor.manual_seed(3)
    ref_cache = ssd_cache = None
    for length in lengths:
      hidden = Tensor.randn(1, length, model.config.dim)
      ref, ref_cache = block.cached(hidden, ref_cache, keep_graph=True)
      out, ssd_cache = ssd_cached(block, hidden, ssd_cache, chunk=chunk, precision=precision)
      np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=tol, atol=tol)
      for key in ("conv", "state"):
        np.testing.assert_allclose(ssd_cache[key].numpy(), ref_cache[key].numpy(), rtol=tol, atol=tol)

  def test_single_piece_lengths(self):
    for length in (1, 3, 8, 16, 37):
      with self.subTest(length=length):
        self._check((length,), chunk=8)

  def test_carried_state_across_pieces(self):
    self._check((16, 11, 32, 5), chunk=8)

  def test_one_chunk_covers_piece(self):
    self._check((13, 20), chunk=64)

  def test_split_precision(self):
    self._check((24, 9), chunk=8, precision="split", tol=1e-3)


if __name__ == "__main__":
  unittest.main()
