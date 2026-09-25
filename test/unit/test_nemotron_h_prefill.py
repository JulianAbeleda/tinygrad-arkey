"""Chunked TinyJit prefill equals the model's own cached prefix."""
import unittest

import numpy as np

from tinygrad import Tensor
from tinygrad.llm.nemotron_h import NemotronHConfig, NemotronHModel
from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
from tinygrad.llm.nemotron_h_prefill_attention import fused_prefill_supported
from test.unit.test_nemotron_h_model import tiny_model


def flash_geometry_model() -> NemotronHModel:
  """Small widths, but the attention geometry of Nemotron 3 Nano (Hq=40, Hkv=8, Hd=128)."""
  config = NemotronHConfig(num_blocks=3, dim=64, vocab_size=64, norm_eps=1e-5, max_context=2048,
                           block_types=("mamba", "attention", "mlp"),
                           head_counts=(0, 40, 0), kv_head_counts=(0, 8, 0), ffn_dims=(0, 0, 96),
                           head_dim=128, ssm_inner=64, ssm_state=8, ssm_groups=2, ssm_heads=4,
                           conv_kernel=4, scan_chunk=16)
  Tensor.manual_seed(0)
  model = NemotronHModel(config)
  mamba = model.blk[0]
  mamba.ssm_a = -(Tensor.rand(config.ssm_heads, 1) + 0.1)
  mamba.ssm_d = Tensor.rand(config.ssm_heads, 1)
  mamba.ssm_dt = {"bias": Tensor.rand(config.ssm_heads) - 0.5}
  mamba.ssm_conv1d.weight = Tensor.randn(*mamba.ssm_conv1d.weight.shape) * 0.3
  mamba.ssm_norm.weight = Tensor.ones(*mamba.ssm_norm.weight.shape)
  return model


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
    for mamba in ("ssd", "scan"):
      with self.subTest(mamba=mamba):
        self._matches_cached_prefix(mamba)

  def _matches_cached_prefix(self, mamba: str):
    model = tiny_model()
    # ssd_chunk 4 < piece 8: several SSD chunks per piece plus a carried state between pieces
    prefill = NemotronHPrefill(model, capacity=64, piece=8, mamba=mamba, ssd_chunk=4)
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


  def test_grouped_tensor_core_attention_matches_sdpa(self):
    from tinygrad.llm.nemotron_h_prefill_attention import grouped_prefill_attention
    Tensor.manual_seed(1)
    q, k, v = Tensor.randn(1, 8, 16, 32), Tensor.randn(1, 2, 64, 32), Tensor.randn(1, 2, 64, 32)
    position, keys = 20, 64  # rows 20..35 over a key bound past the last visible key
    got = grouped_prefill_attention(q, k, v, position, keys).numpy()
    allowed = Tensor.arange(keys).reshape(1, keys) <= (Tensor.arange(16) + position).reshape(16, 1)
    mask = allowed.where(0.0, float("-inf")).reshape(1, 1, 16, keys)
    want = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)

  def test_template_binding_follows_the_bound_weights(self):
    from tinygrad.llm.dense_candidate_gemm import CandidateBinding
    model = tiny_model()
    Tensor.manual_seed(3)
    for index in (1, 3):  # the two MLP blocks share one block graph
      for name in ("ffn_up", "ffn_down"):
        layer = getattr(model.blk[index], name)
        layer.weight = Tensor.randn(*layer.weight.shape).realize()
        layer._candidate = CandidateBinding(name, layer.weight)
    prefill = NemotronHPrefill(model, capacity=64, piece=8)
    kind = prefill.kinds[3]
    self.assertEqual(prefill.templates[kind], 1)
    template = prefill._bind(kind, prefill.bases[3])
    try:
      for name in ("ffn_up", "ffn_down"):
        np.testing.assert_array_equal(getattr(template, name)._candidate.padded.numpy(),
                                      getattr(model.blk[3], name).weight.numpy())
    finally:
      prefill._bind(kind, None)
    for name in ("ffn_up", "ffn_down"):
      np.testing.assert_array_equal(getattr(template, name)._candidate.padded.numpy(), getattr(template, name).weight.numpy())

  def test_unadmitted_geometry_stays_on_sdpa(self):
    self.assertFalse(NemotronHPrefill(tiny_model(), capacity=64, piece=8).fused)


@unittest.skipUnless(fused_prefill_supported(flash_geometry_model()), "fused flash prefill needs a promoted GPU target")
class TestNemotronHFusedPrefill(unittest.TestCase):
  def test_fused_attention_matches_cached_prefix_across_blocks(self):
    model = flash_geometry_model()
    prefill = NemotronHPrefill(model, capacity=1500, piece=512)
    self.assertTrue(prefill.fused)
    self.assertEqual(prefill.capacity, 1536)
    rng = np.random.default_rng(3)
    for _ in range(2):
      # pieces 512, 512, 256, 128, 64, 16: two full blocks, then tails at offsets 0..496 of block 2
      prompt = [int(v) for v in rng.integers(0, 64, 1488)]
      hidden = prefill(prompt).numpy()
      expected, caches = model.prefix(prompt, through=len(model.blk) - 1)
      # the flash kernel reads fp16 Q/K/V
      np.testing.assert_allclose(hidden, expected.numpy()[:, -1:], rtol=2e-2, atol=2e-2)
      for key in ("k", "v"):
        np.testing.assert_allclose(prefill.buffers[1][key].numpy()[:, :, :len(prompt)], caches[1][key].numpy(),
                                   rtol=2e-2, atol=2e-2)
    # graphs are keyed (piece, -1 - block): one per 512-token block, shared by every piece in it
    self.assertEqual(sorted(prefill.graphs), [(16, -3), (64, -3), (128, -3), (256, -3), (512, -2), (512, -1)])

  def test_piece_larger_than_a_block_runs_one_flash_call_per_block(self):
    model = flash_geometry_model()
    prefill = NemotronHPrefill(model, capacity=1500, piece=1024, ssd_precision="split")
    prompt = [int(v) for v in np.random.default_rng(4).integers(0, 64, 1488)]
    hidden = prefill(prompt).numpy()
    expected, caches = model.prefix(prompt, through=len(model.blk) - 1)
    np.testing.assert_allclose(hidden, expected.numpy()[:, -1:], rtol=2e-2, atol=2e-2)
    for key in ("k", "v"):
      np.testing.assert_allclose(prefill.buffers[1][key].numpy()[:, :, :len(prompt)], caches[1][key].numpy(),
                                 rtol=2e-2, atol=2e-2)
    for key in ("conv", "state"):
      np.testing.assert_allclose(prefill.buffers[0][key].numpy(), caches[0][key].numpy(), rtol=1e-3, atol=1e-3)
    # pieces 1024 (blocks 0-1 in one graph), then 256, 128, 64, 16 in block 2
    self.assertEqual(sorted(prefill.graphs), [(16, -3), (64, -3), (128, -3), (256, -3), (1024, -1)])


if __name__ == "__main__":
  unittest.main()
