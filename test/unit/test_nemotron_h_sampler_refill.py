"""NemotronHRolloutSampler: slot refill (forced lengths, occupancy stats, parity) and captured tail inputs."""
import unittest

import numpy as np

from tinygrad import Tensor

from test.unit import test_nemotron_h_model as model_tests
from test.unit.test_nemotron_h_model import tiny_model
from tinygrad.llm.nemotron_h_sampler import NemotronHRolloutSampler


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
