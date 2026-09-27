"""NemotronHRolloutSampler(compact=...): finished lanes shrink the step, parity and captures unchanged."""
import unittest

import numpy as np

from tinygrad import Tensor

from test.unit import test_nemotron_h_model as model_tests
from test.unit.test_nemotron_h_model import tiny_model
from tinygrad.llm.nemotron_h_sampler import NemotronHRolloutSampler


def tail_bits_match(test, sampler, model, results, temperature):
  """Every sampled log probability, recomputed by `tail_logprobs` from its capture `batch` rows at a time (as the
  RLOO update's Tail does, rows packed in any order), equals the sampled value bit for bit."""
  rows = [(h, t, v) for rollouts in results for tokens, logprobs, hidden in rollouts
          for h, t, v in zip(hidden, tokens, logprobs)]
  rows = rows[::-1]  # another packing than the sampler's lanes
  for first in range(0, len(rows), sampler.full):
    chunk = rows[first:first + sampler.full]
    hidden = np.zeros((sampler.full, 1, model.config.dim), np.float32)
    hidden[:len(chunk), 0] = [h for h, _, _ in chunk]
    values = sampler.tail_logprobs(Tensor(hidden), temperature).numpy()
    got = np.array([values[i, t] for i, (_, t, _) in enumerate(chunk)], np.float32)
    want = np.array([v for _, _, v in chunk], np.float32)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def reference(test, model, prompt, tokens, logprobs, temperature):
  hidden, caches = model.prefix(prompt, through=len(model.blk) - 1)
  hidden = hidden[:, -1:]
  if len(tokens) > 1:
    hidden = hidden.cat(model.advance(tokens[:-1], caches, through=len(model.blk) - 1)[0], dim=1)
  expected = (model.output(model.output_norm(hidden))[0].float() / temperature).log_softmax(-1).numpy()
  np.testing.assert_allclose(logprobs, expected[np.arange(len(tokens)), tokens], rtol=1e-4, atol=1e-4)


class TestNemotronHSamplerCompact(unittest.TestCase):
  def test_compacted_rollouts_match_recompute_and_keep_tail_parity(self):
    model = tiny_model()
    capture = len(model.blk) - 1  # the final MLP block, the LoRA tail
    prompts = [[3, 17, 5, 42, 9], [7, 7, 1], [60, 2, 33, 4, 8, 19, 25], [11, 4]]
    limits = [[3, 14], [2, 9], [16, 2], [5, 4]]  # skewed: a few stragglers outlive their siblings
    sampler = NemotronHRolloutSampler(model, batch=4, capacity=16, prefix_capacity=8, prompts=4, rows=2, ring=2,
                                      window=2, capture=capture, compact=(1, 2))
    stats, temperature = {}, 0.7
    results = sampler.generate([(p, 2, l) for p, l in zip(prompts, limits)], max_new=16, temperature=temperature,
                               stats=stats)
    self.assertLess(min(stats["sizes"]), sampler.full)
    self.assertGreater(stats["lane_moves"], 0)
    self.assertLess(stats["row_steps"], stats["lane_steps"])
    self.assertEqual(stats["active_lane_steps"], sum(min(x, 16) for l in limits for x in l))
    for prompt, limit, rollouts in zip(prompts, limits, results):
      self.assertEqual(sorted(len(t) for t, _, _ in rollouts), sorted(limit))
      for tokens, logprobs, hidden in rollouts:
        reference(self, model, prompt, tokens, logprobs, temperature)
        want, caches = model.prefix(prompt, through=capture - 1)
        want = want[:, -1:]
        if len(tokens) > 1:
          want = want.cat(model.advance(tokens[:-1], caches, through=capture - 1)[0], dim=1)
        np.testing.assert_allclose(hidden, want[0].numpy(), rtol=1e-4, atol=1e-4)
    tail_bits_match(self, sampler, model, results, temperature)

  COMPACT = (1, 2)

  def test_compacted_follow_ups_with_a_shared_prefix(self):
    from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
    model = tiny_model()
    rng = np.random.default_rng(5)
    envelope = [int(v) for v in rng.integers(1, 64, 14)]
    prompts = [envelope + [int(v) for v in rng.integers(1, 64, n)] for n in (3, 5)]
    added = []

    def follow(request, rollout):  # a second turn for every first-turn rollout, a third for some
      if request >= len(prompts) + 4 or len(rollout[0]) % 2:
        return None
      base = prompts[request] if request < len(prompts) else added[request - len(prompts)]
      prompt = base + [9, 9] + list(rollout[0][:2])
      added.append(prompt)
      return [(prompt, 1)]

    prefill = NemotronHPrefill(model, capacity=32, piece=8, ssd_chunk=4)
    sampler = NemotronHRolloutSampler(model, batch=4, capacity=16, prefix_capacity=16, prompts=4, rows=2, ring=2,
                                      window=2, capture=len(model.blk) - 1, prefill=prefill, shared_capacity=16,
                                      compact=self.COMPACT)
    Tensor.manual_seed(4)
    stats = {}
    results = sampler.generate([(p, 2, [11, 3]) for p in prompts], max_new=12, temperature=0.8, stats=stats,
                               follow=follow)
    self.assertEqual(len(results), len(prompts) + len(added))
    if self.COMPACT:
      self.assertLess(min(stats["sizes"]), sampler.full)
    for prompt, rollouts in zip(prompts + added, results):
      for tokens, logprobs, _ in rollouts:
        reference(self, model, prompt, tokens, logprobs, 0.8)
    tail_bits_match(self, sampler, model, results, 0.8)

  def test_compacted_without_capture_matches_recompute(self):
    model = tiny_model()
    prompts = [[3, 17, 5, 42, 9], [7, 7, 1], [60, 2, 33, 4], [11, 4]]
    limits = [[3, 8, 5], [2, 6, 4], [7, 2, 3], [5, 5, 1]]
    sampler = NemotronHRolloutSampler(model, batch=4, capacity=16, prefix_capacity=8, prompts=3, rows=2, ring=2,
                                      window=2, compact=(2,))
    stats = {}
    results = sampler.generate([(p, 3, l) for p, l in zip(prompts, limits)], max_new=8, stats=stats)
    self.assertIn(2, stats["sizes"])
    for prompt, limit, rollouts in zip(prompts, limits, results):
      self.assertEqual(sorted(len(t) for t, _ in rollouts), sorted(limit))
      for tokens, logprobs in rollouts:
        model_tests.TestNemotronHRolloutSampler._check(self, model, prompt, tokens, logprobs)

  def test_moves_need_no_memory_of_their_own(self):
    import gc
    from tinygrad.helpers import GlobalCounters
    model = tiny_model()
    sampler = NemotronHRolloutSampler(model, batch=4, capacity=16, prefix_capacity=8, prompts=4, rows=2, ring=2,
                                      window=2, capture=len(model.blk) - 1, compact=(2,))
    sampler.warm()
    self.assertIn(("move", "lane"), sampler.graphs)
    self.assertIn(("move", "slot"), sampler.graphs)
    gc.collect()
    before = GlobalCounters.mem_used
    for _ in range(3):  # the copies' one-lane temporaries live in the step graphs' arenas, captured at warm()
      sampler._move("lane", 0, 1)
      sampler._move("slot", 0, 1)
    gc.collect()
    self.assertEqual(GlobalCounters.mem_used, before)

  def test_compact_rows_are_checked(self):
    model = tiny_model()
    with self.assertRaises(ValueError):
      NemotronHRolloutSampler(model, batch=4, capacity=8, prefix_capacity=8, prompts=4, rows=2, compact=(4,))
    with self.assertRaises(ValueError):
      NemotronHRolloutSampler(model, batch=4, capacity=8, prefix_capacity=8, prompts=2, rows=2, compact=(3,))
    with self.assertRaisesRegex(ValueError, "stateless MLP tail"):
      NemotronHRolloutSampler(model, batch=4, capacity=8, prefix_capacity=8, prompts=4, rows=2, capture=1,
                              compact=(2,))


if __name__ == "__main__":
  unittest.main()
