"""NemotronHRolloutSampler: slot refill (forced lengths, occupancy stats, parity) and captured tail inputs."""
import unittest

import numpy as np

from tinygrad import Tensor

from test.unit import test_nemotron_h_model as model_tests
from test.unit.test_nemotron_h_model import tiny_model
from tinygrad.llm.nemotron_h import NemotronHConfig, NemotronHModel
from tinygrad.llm.nemotron_h_sampler import NemotronHRolloutSampler


def tiny_model_of(types: tuple[str, ...]) -> NemotronHModel:
  """`tiny_model`'s sizes with another block order (e.g. a Mamba block above the attention block)."""
  config = NemotronHConfig(num_blocks=len(types), dim=32, vocab_size=64, norm_eps=1e-5, max_context=64,
                           block_types=types,
                           head_counts=tuple(4 if t == "attention" else 0 for t in types),
                           kv_head_counts=tuple(2 if t == "attention" else 0 for t in types),
                           ffn_dims=tuple(48 if t == "mlp" else 0 for t in types), head_dim=8, ssm_inner=32,
                           ssm_state=8, ssm_groups=2, ssm_heads=4, conv_kernel=4, scan_chunk=4)
  Tensor.manual_seed(0)
  model = NemotronHModel(config)
  for block in model.blk:
    if block.block_type == "mamba":
      block.ssm_a = -(Tensor.rand(config.ssm_heads, 1) + 0.1)
      block.ssm_d = Tensor.rand(config.ssm_heads, 1)
      block.ssm_dt = {"bias": Tensor.rand(config.ssm_heads) - 0.5}
      block.ssm_conv1d.weight = Tensor.randn(*block.ssm_conv1d.weight.shape) * 0.3
      block.ssm_norm.weight = Tensor.ones(*block.ssm_norm.weight.shape)
  return model


class TestNemotronHSamplerRefill(unittest.TestCase):
  def test_forced_lengths_refill_lanes_and_match_recompute(self):
    model = tiny_model()
    prompts = [[3, 17, 5, 42, 9], [7, 7, 1], [60, 2, 33, 4, 8, 19, 25], [11, 4], [5, 6]]
    limits = [[2, 13], [9, 1], [5, 5], [12, 3], [7, 4]]
    # 3 lanes, 2 slots of 2 rows, a 4-step window: short rollouts retire mid-window and their lanes take other
    # prompts' rollouts while a sibling (13, 12 tokens) still holds its slot; no stop tokens, lengths are forced
    sampler = NemotronHRolloutSampler(model, batch=3, capacity=16, prefix_capacity=8, prompts=2, rows=2, ring=2,
                                      window=4)
    stats = {}
    results = sampler.generate([(p, 2, l) for p, l in zip(prompts, limits)], max_new=12, stats=stats)
    check = model_tests.TestNemotronHRolloutSampler._check
    for prompt, limit, rollouts in zip(prompts, limits, results):
      # rollouts come back in finishing order: compare the multiset of lengths, capped by max_new
      self.assertEqual(sorted(len(t) for t, _ in rollouts), sorted(min(x, 12) for x in limit))
      for tokens, logprobs in rollouts:
        check(self, model, prompt, tokens, logprobs)
    self.assertEqual(stats["active_lane_steps"], sum(min(x, 12) for l in limits for x in l))
    self.assertLessEqual(stats["steady_active_lane_steps"], stats["steady_lane_steps"])
    self.assertLessEqual(stats["steady_lane_steps"], stats["lane_steps"])

  def test_limits_must_cover_every_rollout(self):
    sampler = NemotronHRolloutSampler(tiny_model(), batch=2, capacity=8, prefix_capacity=8, prompts=1, rows=2, ring=2,
                                      window=2)
    with self.assertRaises(ValueError):
      sampler.generate([([1, 2], 2, [3])], max_new=4)
    with self.assertRaises(ValueError):
      sampler.generate([([1, 2], 1, [0])], max_new=4)

  def test_default_prompt_slots_cover_stragglers(self):
    sampler = NemotronHRolloutSampler(tiny_model(), batch=4, capacity=8, prefix_capacity=8, rows=2, ring=2, window=2)
    self.assertEqual(sampler.prompts, 2 * 4 // 2 + 2)

  def test_captured_hidden_is_the_tail_input(self):
    model = tiny_model()
    capture = len(model.blk) - 1  # the final MLP block, the LoRA tail
    prompts = [[3, 17, 5, 42, 9], [7, 7, 1], [60, 2, 33, 4, 8, 19, 25], [11, 4]]
    limits = [[6, 11], [3, 9], [10, 2], [7, 7]]
    sampler = NemotronHRolloutSampler(model, batch=3, capacity=16, prefix_capacity=8, prompts=2, rows=2, ring=2,
                                      window=4, capture=capture)
    temperature = 0.7
    results = sampler.generate([(p, 2, l) for p, l in zip(prompts, limits)], max_new=12, temperature=temperature)
    rows = []
    for prompt, rollouts in zip(prompts, results):
      for tokens, logprobs, hidden in rollouts:
        self.assertEqual(hidden.shape, (len(tokens), model.config.dim))
        self.assertEqual(hidden.dtype, np.float32)
        # position i's captured row is the input to the final block at the position that sampled tokens[i]: the
        # prompt's last token for i = 0, then tokens[i - 1]
        reference, caches = model.prefix(prompt, through=capture - 1)
        reference = reference[:, -1:]
        if len(tokens) > 1:
          reference = reference.cat(model.advance(tokens[:-1], caches, through=capture - 1)[0], dim=1)
        np.testing.assert_allclose(hidden, reference[0].numpy(), rtol=1e-4, atol=1e-4)
        rows += [(row, token, logprob) for row, token, logprob in zip(hidden, tokens, logprobs)]
    # the tail recomputed from the captures, `batch` rows at a time as the step ran it, gives the sampled bits
    for first in range(0, len(rows), sampler.batch):
      chunk = rows[first:first + sampler.batch]
      hidden = np.zeros((sampler.batch, 1, model.config.dim), np.float32)
      hidden[:len(chunk), 0] = [row for row, _, _ in chunk]
      logprobs = sampler.tail_logprobs(Tensor(hidden), temperature).numpy()
      got = np.array([logprobs[i, token] for i, (_, token, _) in enumerate(chunk)], np.float32)
      want = np.array([logprob for _, _, logprob in chunk], np.float32)
      np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

  def test_replay_of_every_block_window_is_bit_exact(self):
    prompts = [[3, 17, 5, 42, 9], [7, 7, 1], [60, 2, 33, 4, 8, 19, 25], [11, 4]]
    limits = [[6, 11], [3, 9], [10, 2], [7, 7]]
    temperature = 0.7
    # both orders put Mamba, attention and MLP blocks inside and outside some window [k, end)
    for types in (("mamba", "mlp", "attention", "mlp"), ("attention", "mlp", "mamba", "mlp")):
      model = tiny_model_of(types)
      for capture in range(len(types)):
        with self.subTest(types=types, capture=capture):
          sampler = NemotronHRolloutSampler(model, batch=3, capacity=16, prefix_capacity=8, prompts=2, rows=2, ring=2,
                                            window=4, capture=capture)
          stats = {}
          results = sampler.generate([(p, 2, l) for p, l in zip(prompts, limits)], max_new=12,
                                     temperature=temperature, stats=stats)
          for prompt, rollouts in zip(prompts, results):
            tokens, _, hidden = rollouts[0]
            if capture:  # the capture is the input to block `capture`, at the positions that sampled the tokens
              reference, caches = model.prefix(prompt, through=capture - 1)
              reference = reference[:, -1:]
              if len(tokens) > 1:
                reference = reference.cat(model.advance(tokens[:-1], caches, through=capture - 1)[0], dim=1)
            else:
              reference = model.token_embd(Tensor([[prompt[-1]] + tokens[:-1]])).float()
            np.testing.assert_allclose(hidden, reference[0].numpy(), rtol=1e-4, atol=1e-4)
          replayed = sampler.replay([(p, [(t, h) for t, _, h in r]) for p, r in zip(prompts, results)], temperature,
                                    stats=stats)
          got = np.concatenate([values for rollouts in replayed for values in rollouts])
          want = np.array([v for rollouts in results for _, logprobs, _ in rollouts for v in logprobs], np.float32)
          np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

  def test_bucketed_ring_reads_keep_parity_and_replay_bits(self):
    model = tiny_model()  # block 2 is attention
    prompts = [[3, 17, 5, 42, 9], [7, 7, 1], [60, 2, 33, 4, 8, 19, 25], [11, 4], [5, 6, 7]]
    limits = [[6, 11], [3, 9], [10, 2], [7, 7], [12, 5]]
    # a 32-row ring read through 8- and 16-row buckets (the whole ring past 16), wrapping around
    sampler = NemotronHRolloutSampler(model, batch=3, capacity=32, prefix_capacity=8, prompts=2, rows=2, ring=2,
                                      window=4, capture=1, min_bucket=4)
    stats = {}
    results = sampler.generate([(p, 2, l) for p, l in zip(prompts, limits)], max_new=12, temperature=0.7,
                               stats=stats)
    buckets, window = stats["buckets"], sampler.window
    self.assertLess(min(buckets), sampler.capacity)
    self.assertGreater(stats["steps"], sampler.capacity)
    self.assertTrue(any(b < sampler.capacity and (k * window) % sampler.capacity < b - 1 for k, b in enumerate(buckets)))
    self.assertTrue(any(b < sampler.capacity and (k * window) % sampler.capacity >= b - 1 for k, b in enumerate(buckets)))
    for prompt, rollouts in zip(prompts, results):
      for tokens, logprobs, _ in rollouts:
        reference = model.prefix(prompt, through=len(model.blk) - 1)
        hidden, caches = reference[0][:, -1:], reference[1]
        if len(tokens) > 1:
          hidden = hidden.cat(model.advance(tokens[:-1], caches, through=len(model.blk) - 1)[0], dim=1)
        expected = (model.output(model.output_norm(hidden))[0].float() / 0.7).log_softmax(-1).numpy()
        np.testing.assert_allclose(logprobs, expected[np.arange(len(tokens)), tokens], rtol=1e-4, atol=1e-4)
    replayed = sampler.replay([(p, [(t, h) for t, _, h in r]) for p, r in zip(prompts, results)], 0.7, stats=stats)
    got = np.concatenate([values for rollouts in replayed for values in rollouts])
    want = np.array([v for rollouts in results for _, logprobs, _ in rollouts for v in logprobs], np.float32)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

  def test_prefill_primed_slots_match_the_prefill_state_and_recompute(self):
    from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
    model = tiny_model()
    prefill = NemotronHPrefill(model, capacity=16, piece=8)
    sampler = NemotronHRolloutSampler(model, batch=2, capacity=8, prefix_capacity=12, prompts=1, rows=2, ring=2,
                                      window=2, prefill=prefill)
    for prompt in ([4, 9, 13, 2, 7, 30, 1, 5, 8, 12, 3], [6, 21, 3, 17]):  # a long prompt, then a short one: stale rows
      sampler._prime(0, prompt)
      length = len(prompt) - 1
      for attention, state, cache in zip(sampler.attention, sampler.prompt_state, prefill.buffers):
        if attention is not None:
          for key in ("k", "v"):
            slot = attention.prefix[key].numpy()[0]
            np.testing.assert_array_equal(slot[:, :length], cache[key].numpy()[0, :, :length])
            self.assertFalse(slot[:, length:].any())
        elif state is not None:
          for key in ("conv", "state"):
            np.testing.assert_array_equal(state[key].numpy()[0], cache[key].numpy()[0])
    prompts = [[4, 9, 13, 2, 7, 30, 1, 5, 8, 12, 3], [6, 21, 3, 17]]
    results = sampler.generate([(p, 2) for p in prompts], max_new=6)
    for prompt, rollouts in zip(prompts, results):
      for tokens, logprobs in rollouts:
        model_tests.TestNemotronHRolloutSampler._check(self, model, prompt, tokens, logprobs)

  def test_bucket_step_graphs_share_one_arena(self):
    from tinygrad.uop.ops import Ops
    from tinygrad.dtype import dtypes
    model = tiny_model()
    sampler = NemotronHRolloutSampler(model, batch=3, capacity=32, prefix_capacity=8, prompts=2, rows=2, ring=2,
                                      window=4, min_bucket=4)
    sampler.warm()
    # buckets 4, 8, 16 in both forms, plus the whole ring: 7 graphs, all planned into the one pool
    self.assertEqual(len(sampler.graphs), 7)
    self.assertTrue(sampler.arenas)
    def arenas():  # every graph's planned intermediates live in these int8 buffers
      return {id(u) for graph in sampler.graphs.values() for u in graph.captured.linear.toposort()
              if u.op is Ops.BUFFER and u.dtype == dtypes.int8}
    pool = {id(arena) for arena in sampler.arenas.values()}
    # every graph plans into the pool alone: none keeps an arena of its own or an outgrown one
    self.assertEqual(arenas(), pool)
    # replay graphs join the same pool, and step graphs are recaptured if that had to grow it
    capture = NemotronHRolloutSampler(model, batch=3, capacity=32, prefix_capacity=8, prompts=2, rows=2, ring=2,
                                      window=4, min_bucket=4, capture=1)
    stats = {}
    results = capture.generate([([3, 17, 5], 2, [9, 12]), ([7, 1], 2, [5, 3])], max_new=12, stats=stats)
    capture.replay([(p, [(t, h) for t, _, h in r]) for p, r in zip(([3, 17, 5], [7, 1]), results)], stats=stats)
    sampler = capture
    self.assertEqual(arenas(), {id(arena) for arena in capture.arenas.values()})

  def test_one_mamba_state_serves_groups_larger_than_their_rows(self):
    model = tiny_model()
    # 3 rollouts per prompt on 2 rows: a prompt's last rollout waits for a row while lanes are free; the next prompt
    # must not be primed over its Mamba state before it starts
    sampler = NemotronHRolloutSampler(model, batch=4, capacity=16, prefix_capacity=8, prompts=3, rows=2, ring=2,
                                      window=2)
    self.assertEqual(sampler.prompt_state[0]["state"].shape[0], 1)
    prompts = [[3, 17, 5, 42, 9], [7, 7, 1], [60, 2, 33, 4], [11, 4]]
    limits = [[3, 8, 5], [2, 6, 4], [7, 2, 3], [5, 5, 1]]
    results = sampler.generate([(p, 3, l) for p, l in zip(prompts, limits)], max_new=8)
    for prompt, limit, rollouts in zip(prompts, limits, results):
      self.assertEqual(sorted(len(t) for t, _ in rollouts), sorted(limit))
      for tokens, logprobs in rollouts:
        model_tests.TestNemotronHRolloutSampler._check(self, model, prompt, tokens, logprobs)

  def test_replay_through_attention_needs_ring_rows(self):
    model = tiny_model()  # block 2 is attention
    sampler = NemotronHRolloutSampler(model, batch=2, capacity=8, prefix_capacity=8, prompts=1, rows=2, ring=2,
                                      window=2, capture=1)
    (rollouts,) = sampler.generate([([3, 17, 5], 2, [3, 4])], max_new=4)
    with self.assertRaisesRegex(ValueError, "needs generate.s stats"):
      sampler.replay([([3, 17, 5], [(t, h) for t, _, h in rollouts])])

  def test_prompts_need_two_tokens(self):
    sampler = NemotronHRolloutSampler(tiny_model(), batch=2, capacity=8, prefix_capacity=8, prompts=1, rows=2, ring=2,
                                      window=2)
    with self.assertRaises(ValueError):
      sampler.generate([([5], 1)], max_new=4)

  def test_stale_non_finite_state_does_not_reach_a_new_rollout(self):
    model = tiny_model()
    sampler = NemotronHRolloutSampler(model, batch=3, capacity=16, prefix_capacity=8, prompts=2, rows=2, ring=2,
                                      window=4)
    # a previous occupant that went NaN: every lane ring, replay ring and prompt-slot row holds NaN; the masked
    # entries meet weight 0, and 0 * NaN is NaN unless the lane and slot are reset when they are handed out
    for buffer in sampler.buffers:
      if buffer is not None:
        for key in ("ring_x", "ring_a", "ring_b"):
          buffer[key].assign(Tensor.full(buffer[key].shape, float("nan"), dtype=buffer[key].dtype)).realize()
    for attention in sampler.attention:
      if attention is not None:
        for value in (*attention.suffix.values(), *attention.prefix.values()):
          value.assign(Tensor.full(value.shape, float("nan"), dtype=value.dtype)).realize()
    prompts = [[3, 17, 5], [7, 7, 1, 9], [11, 4]]
    results = sampler.generate([(p, 2) for p in prompts], max_new=9)
    for prompt, rollouts in zip(prompts, results):
      for tokens, logprobs in rollouts:
        model_tests.TestNemotronHRolloutSampler._check(self, model, prompt, tokens, logprobs)

  def test_non_finite_logits_raise_instead_of_returning_vocab_ids(self):
    model = tiny_model()
    bias = np.zeros(model.config.vocab_size, np.float32)
    bias[5] = np.nan  # every row's log_softmax is NaN: argmax would return vocab_size with log probability 0
    sampler = NemotronHRolloutSampler(model, batch=2, capacity=8, prefix_capacity=8, prompts=1, rows=2, ring=2,
                                      window=2, bias=bias)
    with self.assertRaisesRegex(FloatingPointError, "non-finite logits: request 0"):
      sampler.generate([([3, 17, 5], 2)], max_new=4)


if __name__ == "__main__":
  unittest.main()
