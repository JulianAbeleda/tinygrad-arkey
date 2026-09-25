"""Nemotron-H hybrid model: the cached prefix/advance path equals the full forward."""
import unittest

import numpy as np

from tinygrad import Tensor
from tinygrad.llm.nemotron_h import NemotronHConfig, NemotronHModel


def tiny_model() -> NemotronHModel:
  config = NemotronHConfig(num_blocks=4, dim=32, vocab_size=64, norm_eps=1e-5, max_context=64,
                           block_types=("mamba", "mlp", "attention", "mlp"),
                           head_counts=(0, 0, 4, 0), kv_head_counts=(0, 0, 2, 0), ffn_dims=(0, 48, 0, 48),
                           head_dim=8, ssm_inner=32, ssm_state=8, ssm_groups=2, ssm_heads=4,
                           conv_kernel=4, scan_chunk=4)
  Tensor.manual_seed(0)
  model = NemotronHModel(config)
  mamba = model.blk[0]
  mamba.ssm_a = -(Tensor.rand(config.ssm_heads, 1) + 0.1)
  mamba.ssm_d = Tensor.rand(config.ssm_heads, 1)
  mamba.ssm_dt = {"bias": Tensor.rand(config.ssm_heads) - 0.5}
  mamba.ssm_conv1d.weight = Tensor.randn(*mamba.ssm_conv1d.weight.shape) * 0.3
  mamba.ssm_norm.weight = Tensor.ones(*mamba.ssm_norm.weight.shape)
  return model


class TestNemotronH(unittest.TestCase):
  def test_cached_prefix_and_advance_match_full_forward(self):
    model = tiny_model()
    ids = [int(v) for v in np.random.default_rng(1).integers(0, 64, 13)]
    hidden = model.token_embd(Tensor([ids])).float()
    for block in model.blk:
      hidden = block(hidden)
    full = hidden.numpy()[0]
    prefix_hidden, caches = model.prefix(ids[:7], through=len(model.blk) - 1)
    advanced, _ = model.advance(ids[7:], caches, through=len(model.blk) - 1)
    joined = np.concatenate([prefix_hidden.numpy()[0], advanced.numpy()[0]])
    np.testing.assert_allclose(joined, full, rtol=1e-4, atol=1e-4)


class TestNemotronHBatchSampler(unittest.TestCase):
  def test_batched_jit_logprobs_match_sequential_recompute(self):
    from tinygrad.llm.nemotron_h_sampler import NemotronHBatchSampler
    model = tiny_model()
    prompt = [3, 17, 5, 42, 9, 11, 2]
    sampler = NemotronHBatchSampler(model, batch=3, capacity=32)
    rollouts = sampler.generate(prompt, steps=6, temperature=1.0)
    self.assertEqual(len(rollouts), 3)
    for tokens, logprobs in rollouts:
      _, caches = model.prefix(prompt, through=len(model.blk) - 1)
      hidden = model.prefix(prompt, through=len(model.blk) - 1)[0][:, -1:]
      if len(tokens) > 1:
        hidden = hidden.cat(model.advance(tokens[:-1], caches, through=len(model.blk) - 1)[0], dim=1)
      reference = model.output(model.output_norm(hidden))[0].float().log_softmax(-1).numpy()
      expected = reference[np.arange(len(tokens)), tokens]
      np.testing.assert_allclose(np.asarray(logprobs), expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
