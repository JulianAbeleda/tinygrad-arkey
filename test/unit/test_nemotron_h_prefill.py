"""Chunked TinyJit prefill equals the model's own cached prefix."""
import unittest

import numpy as np

from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
from test.unit.test_nemotron_h_model import tiny_model


class TestNemotronHPrefill(unittest.TestCase):
  def test_pieces_cover_length_with_powers_of_two(self):
    prefill = NemotronHPrefill(tiny_model(), capacity=64, piece=8)
    self.assertEqual(prefill.pieces(29), [8, 8, 8, 4, 1])
    self.assertEqual(prefill.pieces(16), [8, 8])

  def test_attention_reads_a_power_of_two_key_bound_not_capacity(self):
    prefill = NemotronHPrefill(tiny_model(), capacity=64, piece=8)
    prefill(list(range(1, 22)))
    # pieces 8, 8, 4, 1 end at 8, 16, 20, 21: key bounds 8, 16, 32, 32
    self.assertEqual(sorted(prefill.graphs), [(1, 32), (4, 32), (8, 8), (8, 16)])

  def test_matches_cached_prefix_across_jit_replays(self):
    model = tiny_model()
    prefill = NemotronHPrefill(model, capacity=64, piece=8)
    rng = np.random.default_rng(2)
    # three prompts of one length: the graphs capture, then replay
    for _ in range(3):
      prompt = [int(v) for v in rng.integers(0, 64, 29)]
      hidden = prefill(prompt).numpy()
      expected, caches = model.prefix(prompt, through=len(model.blk) - 1)
      np.testing.assert_allclose(hidden, expected.numpy()[:, -1:], rtol=1e-4, atol=1e-4)
      for block, buffer, cache in zip(model.blk, prefill.buffers, caches):
        if block.block_type == "attention":
          for key in ("k", "v"):
            np.testing.assert_allclose(buffer[key].numpy()[:, :, :len(prompt)], cache[key].numpy(), rtol=1e-4, atol=1e-4)
        elif block.block_type == "mamba":
          for key in ("conv", "state"):
            np.testing.assert_allclose(buffer[key].numpy(), cache[key].numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
