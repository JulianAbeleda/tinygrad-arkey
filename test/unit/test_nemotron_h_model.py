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


def tiny_metadata(config: NemotronHConfig) -> dict:
  arch = "nemotron_h"
  return {"general.architecture": arch, "general.file_type": 32, f"{arch}.block_count": config.num_blocks,
          f"{arch}.attention.head_count": list(config.head_counts),
          f"{arch}.attention.head_count_kv": list(config.kv_head_counts),
          f"{arch}.feed_forward_length": list(config.ffn_dims), f"{arch}.context_length": config.max_context,
          f"{arch}.embedding_length": config.dim, "tokenizer.ggml.tokens": ["t"] * config.vocab_size,
          f"{arch}.attention.layer_norm_rms_epsilon": config.norm_eps, f"{arch}.attention.key_length": config.head_dim,
          f"{arch}.ssm.inner_size": config.ssm_inner, f"{arch}.ssm.state_size": config.ssm_state,
          f"{arch}.ssm.group_count": config.ssm_groups, f"{arch}.ssm.time_step_rank": config.ssm_heads,
          f"{arch}.ssm.conv_kernel": config.conv_kernel}


class TestNemotronHLoad(unittest.TestCase):
  def test_loaded_weights_are_realized_buffers(self):
    # GGUF tensors arrive as lazy decodes of the file bytes; left lazy, every
    # consumer kernel re-decodes them. load_state must materialize each one.
    from tinygrad import dtypes
    from tinygrad.llm.nemotron_h import load_state
    from tinygrad.nn.state import get_parameters, get_state_dict
    config = tiny_model().config
    state = {name: (Tensor.ones(*value.shape) * 2).cast(dtypes.bfloat16)
             for name, value in get_state_dict(NemotronHModel(config)).items()}
    self.assertFalse(any(value.uop.is_realized for value in state.values()))
    model = load_state(tiny_metadata(config), state)
    for weight in get_parameters(model):
      self.assertTrue(weight.uop.is_realized)
      self.assertEqual(weight.dtype, dtypes.bfloat16)
    self.assertEqual(model.blk[1].ffn_up.weight.float().numpy()[0, 0], 2.0)


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

  def test_step_reads_each_projection_weight_once(self):
    # a lazily chained residual stream makes every consumer of a block's output recompute its projection
    from tinygrad.nn.state import get_parameters
    from tinygrad.llm.nemotron_h_sampler import NemotronHBatchSampler
    from tinygrad.uop.ops import Ops
    model = tiny_model()
    for weight in get_parameters(model): weight.replace(weight.contiguous().realize())
    sampler = NemotronHBatchSampler(model, batch=2, capacity=32)
    sampler.generate([3, 17, 5, 42], steps=4)
    calls = [c for c in sampler.step.captured.linear.toposort() if c.op is Ops.CALL]
    def buffer_of(u):
      try: return u.buffer
      except Exception: return None
    def readers(weight: Tensor) -> int:
      return sum(1 for c in calls if any(buffer_of(x) is weight.uop.buffer for x in c.src[1:]))
    for weight in (model.output.weight, model.token_embd.weight, model.blk[0].ssm_in.weight, model.blk[0].ssm_out.weight,
                   model.blk[1].ffn_up.weight, model.blk[1].ffn_down.weight,
                   model.blk[2].attn_output.weight):
      self.assertEqual(readers(weight), 1)


if __name__ == "__main__":
  unittest.main()
