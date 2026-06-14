import unittest

import numpy as np

from tinygrad import Tensor, nn

from extra.llm_adapter_suffix_train import (
  build_suffix_cache, parity_check, prefix_hidden_numpy, suffix_logits_from_hidden, suffix_start_from_targets,
)
from extra.llm_adapter_train import _plain_logits

class Identity:
  def __call__(self, x): return x

class ToySuffixBlock:
  def __init__(self, dim:int=4, hidden:int=6):
    self.attn_norm = Identity()
    self.ffn_norm = Identity()
    self.ffn_gate = nn.Linear(dim, hidden, bias=False)
    self.ffn_up = nn.Linear(dim, hidden, bias=False)
    self.ffn_down = nn.Linear(hidden, dim, bias=False)

  def _init_state(self, x): pass
  def _attention(self, x, start_pos): return x * 0.125
  def _feed_forward(self, x): return self.ffn_down(self.ffn_gate(x).silu() * self.ffn_up(x))

  def __call__(self, x, start_pos):
    self._init_state(x)
    h = x + self._attention(self.attn_norm(x), start_pos)
    return (h + self._feed_forward(self.ffn_norm(h))).contiguous()

class ToySuffixModel:
  def __init__(self, blocks:int=3, vocab:int=11, dim:int=4):
    self.token_embd = nn.Embedding(vocab, dim)
    self.blk = [ToySuffixBlock(dim=dim) for _ in range(blocks)]
    self.output_norm = Identity()
    self.output = nn.Linear(dim, vocab, bias=False)

  def logits(self, tokens, start_pos):
    x = self.token_embd(tokens).float()
    for block in self.blk: x = block(x, start_pos)
    return self.output(self.output_norm(x))

class TestLLMAdapterSuffixTrain(unittest.TestCase):
  def test_suffix_start_requires_last_ffn_group(self):
    model = ToySuffixModel(blocks=4)
    self.assertEqual(suffix_start_from_targets(model, ["last1_ffn"]), 3)
    self.assertEqual(suffix_start_from_targets(model, ["last3_ffn"]), 1)
    with self.assertRaisesRegex(ValueError, "requires a lastN_ffn"):
      suffix_start_from_targets(model, ["output"])
    with self.assertRaisesRegex(ValueError, "requested 9 blocks"):
      suffix_start_from_targets(model, ["last9_ffn"])

  def test_prefix_suffix_logits_match_plain_logits(self):
    Tensor.manual_seed(7)
    model = ToySuffixModel(blocks=3)
    ids = [1, 2, 3, 4]
    suffix_start = suffix_start_from_targets(model, ["last2_ffn"])
    full = _plain_logits(model, Tensor([ids], dtype="int32"), 0)[:, -1, :].numpy()
    hidden = prefix_hidden_numpy(model, ids, suffix_start, "float32")
    suffix = suffix_logits_from_hidden(model, Tensor(hidden), suffix_start)[:, -1, :].numpy()
    np.testing.assert_allclose(full, suffix, atol=1e-6)

  def test_build_suffix_cache_records_hidden_metadata(self):
    Tensor.manual_seed(8)
    model = ToySuffixModel(blocks=2)
    examples = [
      {"id": "a:tok0", "row_id": "a", "ids": [1, 2], "target": 3, "input_tokens": 2},
      {"id": "a:tok1", "row_id": "a", "ids": [1, 2, 3], "target": 4, "input_tokens": 3},
    ]
    cache, summary = build_suffix_cache(model, examples, suffix_start=1, cache_dtype="float32")
    self.assertEqual(len(cache), 2)
    self.assertEqual(summary["examples"], 2)
    self.assertEqual(summary["prefix_cache_entries"], 1)
    self.assertEqual(summary["dtype"], "float32")
    self.assertEqual(summary["hidden_shape_tail"], [4])
    self.assertGreater(summary["total_hidden_bytes"], 0)
    self.assertEqual(cache[0].hidden.shape, (1, 2, 4))

  def test_parity_check_fails_loudly_on_bad_suffix_boundary(self):
    Tensor.manual_seed(9)
    model = ToySuffixModel(blocks=3)
    examples = [{"id": "a", "ids": [1, 2, 3], "target": 4, "input_tokens": 3}]
    good = parity_check(model, examples, suffix_start=2, cache_dtype="float32", limit=1, tol=1e-6, device=None)
    self.assertEqual(good["status"], "pass")
    bad = parity_check(model, examples, suffix_start=1, cache_dtype="float16", limit=1, tol=0.0, device=None)
    self.assertEqual(bad["status"], "fail")

if __name__ == "__main__":
  unittest.main()
