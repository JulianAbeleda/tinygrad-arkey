"""Shared-prefix, length-bounded decode attention equals attention over the concatenated prompt and suffix."""
import unittest

import numpy as np

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.nemotron_h_attention import (SharedPrefixKV, decode_attention, load_prefix_from_prefill,
                                               shared_kv_for_model, suffix_bucket, suffix_buckets)
from test.unit.test_nemotron_h_model import tiny_model

try:
  from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
except ImportError:  # the chunked prefill lands separately
  NemotronHPrefill = None


def _reference(q: np.ndarray, prefix_k, prefix_v, suffix_k, suffix_v, prefix_length: int, step: int) -> np.ndarray:
  """Plain SDPA over [prefix, whole suffix buffer] with the causal mask of the query at position prefix_length+step."""
  batch = q.shape[0]
  keys = Tensor(np.concatenate([np.broadcast_to(prefix_k[:, :, :prefix_length], (batch, *prefix_k.shape[1:2],
                                                                                   prefix_length, prefix_k.shape[3])),
                                suffix_k], axis=2))
  values = Tensor(np.concatenate([np.broadcast_to(prefix_v[:, :, :prefix_length], (batch, *prefix_v.shape[1:2],
                                                                                     prefix_length, prefix_v.shape[3])),
                                  suffix_v], axis=2))
  total = keys.shape[2]
  mask = (Tensor.arange(total) <= prefix_length + step).reshape(1, 1, 1, total).where(0.0, float("-inf"))
  return Tensor(q).scaled_dot_product_attention(keys, values, attn_mask=mask, enable_gqa=True).numpy()


class TestSuffixBuckets(unittest.TestCase):
  def test_powers_of_two_end_at_capacity(self):
    self.assertEqual(suffix_buckets(4096, 64), [64, 128, 256, 512, 1024, 2048, 4096])
    self.assertEqual(suffix_buckets(100, 16), [16, 32, 64, 100])
    self.assertEqual(suffix_bucket(0, 4096), 64)
    self.assertEqual(suffix_bucket(64, 4096), 128)
    self.assertEqual(suffix_bucket(2047, 4096), 2048)
    self.assertEqual(suffix_bucket(99, 100, 16), 100)
    with self.assertRaises(ValueError):
      suffix_bucket(100, 100)


class TestTwoSegmentAttention(unittest.TestCase):
  def test_matches_concatenated_sdpa_across_positions_buckets_and_replays(self):
    for chunk in (1024, 4):  # one chunk per segment; several, including fully masked ones
      with self.subTest(chunk=chunk):
        self._check_parity(chunk)

  def _check_parity(self, chunk: int):
    config = tiny_model().config
    layer_index = config.block_types.index("attention")
    heads, kv_heads, width = config.head_counts[layer_index], config.kv_head_counts[layer_index], config.head_dim
    batch, prefix_capacity, suffix_capacity, minimum = 3, 16, 32, 4
    rng = np.random.default_rng(3)
    layer = SharedPrefixKV(batch, kv_heads, width, prefix_capacity, suffix_capacity, dtypes.float32, chunk)
    prefix_k, prefix_v = (rng.standard_normal((1, kv_heads, prefix_capacity, width)).astype(np.float32) for _ in "kv")
    layer.load_prefix(Tensor(prefix_k), Tensor(prefix_v))
    suffix_k, suffix_v = (rng.standard_normal((batch, kv_heads, suffix_capacity, width)).astype(np.float32) for _ in "kv")
    layer.suffix["k"].assign(Tensor(suffix_k)).realize()
    layer.suffix["v"].assign(Tensor(np.ascontiguousarray(suffix_v.transpose(0, 1, 3, 2)))).realize()
    prefix_var = UOp.variable("prefix_length", 1, prefix_capacity)
    step_var = UOp.variable("suffix_step", 0, suffix_capacity - 1)
    graphs: dict[int, TinyJit] = {}

    def run(q: Tensor, prefix_length: UOp, step: UOp, bucket: int) -> Tensor:
      return layer.attend(q, prefix_length, step, bucket).realize()

    # every bucket is visited at several steps and two prompt lengths, so each graph captures and then replays
    cases = [(prefix_length, step) for prefix_length in (11, 5, 2) for step in (0, 1, 3, 4, 6, 7, 9, 15, 16, 24, 31)]
    for prefix_length, step in cases:
      bucket = suffix_bucket(step, suffix_capacity, minimum)
      q = rng.standard_normal((batch, heads, 1, width)).astype(np.float32)
      graph = graphs.setdefault(bucket, TinyJit(run))
      out = graph(Tensor(q).realize(), prefix_var.bind(prefix_length), step_var.bind(step), bucket).numpy()
      expected = _reference(q, prefix_k, prefix_v, suffix_k, suffix_v, prefix_length, step)
      np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5, err_msg=f"prefix={prefix_length} step={step}")
    self.assertEqual(sorted(graphs), suffix_buckets(suffix_capacity, minimum))
    self.assertTrue(all(graph.cnt > 2 for graph in graphs.values()), "every bucket graph must replay")


class TestSharedPrefixDecode(unittest.TestCase):
  def _decode_matches_full_recompute(self, use_prefill: bool):
    model = tiny_model()
    last = len(model.blk) - 1
    batch, suffix_capacity, minimum, prefix_capacity = 3, 16, 4, 32
    rng = np.random.default_rng(4)
    prompt = [int(v) for v in rng.integers(0, 64, 13)]
    continuations = [[int(v) for v in rng.integers(0, 64, 9)] for _ in range(batch)]
    if use_prefill:
      prefill = NemotronHPrefill(model, capacity=prefix_capacity, piece=8)
      prefill(prompt)
      sources = prefill.buffers
    else:
      sources = model.prefix(prompt, through=last)[1]
    layers = shared_kv_for_model(model, batch, prefix_capacity, suffix_capacity)
    load_prefix_from_prefill(layers, sources)
    states = [None if block.block_type != "mamba" else
              {key: Tensor.empty(batch, *value.shape[1:], dtype=value.dtype).assign(
                value.expand(batch, *value.shape[1:])).realize() for key, value in source.items()}
              for block, source in zip(model.blk, sources)]
    prefix_var = UOp.variable("prefix_length", 1, prefix_capacity)
    step_var = UOp.variable("suffix_step", 0, suffix_capacity - 1)

    def step_fn(tokens: Tensor, prefix_length: UOp, step: UOp, bucket: int) -> Tensor:
      hidden = model.token_embd(tokens.reshape(batch, 1)).float()
      for block, layer, state in zip(model.blk, layers, states):
        if block.block_type == "attention":
          hidden = hidden + decode_attention(block, block.attn_norm(hidden), layer, prefix_length, step, bucket)
        elif block.block_type == "mamba":
          hidden, carried = block.cached(hidden, state, keep_graph=True)
          hidden, *new = (hidden.contiguous().realize(), *(carried[key].contiguous().realize() for key in state))
          for key, value in zip(state, new):
            state[key].assign(value).realize()
        else:
          hidden, _ = block.cached(hidden, None, keep_graph=True)
      return hidden.contiguous().realize()

    graphs: dict[int, TinyJit] = {}
    outputs = []
    for step in range(len(continuations[0])):
      bucket = suffix_bucket(step, suffix_capacity, minimum)
      tokens = Tensor([row[step] for row in continuations], dtype=dtypes.int32).realize()
      graph = graphs.setdefault(bucket, TinyJit(step_fn))
      outputs.append(graph(tokens, prefix_var.bind(len(prompt)), step_var.bind(step), bucket).numpy()[:, 0])
    self.assertEqual(sorted(graphs), [4, 8, 16])
    for index, continuation in enumerate(continuations):
      _, caches = model.prefix(prompt, through=last)
      expected = model.advance(continuation, caches, through=last)[0].numpy()[0]
      got = np.stack([output[index] for output in outputs])
      np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4, err_msg=f"sequence {index}")

  def test_decode_from_cached_prefix_matches_full_recompute(self):
    self._decode_matches_full_recompute(use_prefill=False)

  @unittest.skipIf(NemotronHPrefill is None, "chunked prefill not present")
  def test_decode_from_chunked_prefill_matches_full_recompute(self):
    self._decode_matches_full_recompute(use_prefill=True)


class TestExactProducts(unittest.TestCase):
  def test_bf16_operands_multiply_exactly(self):
    from tinygrad.llm.nemotron_h_attention import _exact_dot
    x = Tensor([[1.0078125]], dtype=dtypes.bfloat16)  # 1 + 2**-7: its square needs 15 mantissa bits, bf16 keeps 8
    self.assertEqual(_exact_dot(x, x).numpy()[0, 0], np.float32(1.0078125) ** 2)


class TestRolloutAttention(unittest.TestCase):
  def test_matches_softmax_over_prompt_slot_and_own_ring_keys(self):
    from tinygrad.llm.nemotron_h_attention import rollout_attention
    rng = np.random.default_rng(0)
    batch, heads, kv_heads, width, prompts, rows, span, ring = 5, 4, 2, 8, 3, 2, 16, 8
    q = rng.standard_normal((batch, heads, 1, width)).astype(np.float32)
    pk, pv = (rng.standard_normal((prompts, kv_heads, span, width)).astype(np.float32) for _ in range(2))
    sk, sv = (rng.standard_normal((batch, kv_heads, ring, width)).astype(np.float32) for _ in range(2))
    lane_prompt, prefix_lengths, lengths = [0, 1, 0, 1, 2], [5, 11, 7], [3, 1, 8, 5, 2]
    lane_rows, lane_slot = [0, 2, 1, 3, 4, 0], [0, 2, 1, 3, 4]  # slot g row j -> lane; lane -> g*rows + j
    for row, chunk in ((2, 4), (7, 16)):  # ring wrap-around, one and several key chunks
      out = rollout_attention(Tensor(q), Tensor(pk), Tensor(pv), Tensor(sk), Tensor(sv.transpose(0, 1, 3, 2).copy()),
                              Tensor(lane_rows, dtype=dtypes.int32), Tensor(lane_slot, dtype=dtypes.int32),
                              Tensor(prefix_lengths, dtype=dtypes.int32), Tensor(lengths, dtype=dtypes.int32), row,
                              chunk).numpy()
      for b in range(batch):
        g = lane_prompt[b]
        own = [w for w in range(ring) if (row - w) % ring < lengths[b]]
        keys = np.concatenate([pk[g, :, :prefix_lengths[g]], sk[b][:, own]], 1)
        values = np.concatenate([pv[g, :, :prefix_lengths[g]], sv[b][:, own]], 1)
        for h in range(heads):
          scores = keys[h // (heads // kv_heads)] @ q[b, h, 0] / np.sqrt(width)
          p = np.exp(scores - scores.max())
          np.testing.assert_allclose(out[b, h, 0], p @ values[h // (heads // kv_heads)] / p.sum(), rtol=1e-4, atol=1e-5)

  def test_bucketed_ring_read_matches_softmax_with_and_without_wrap(self):
    from tinygrad.llm.nemotron_h_attention import rollout_attention
    rng = np.random.default_rng(1)
    batch, heads, kv_heads, width, prompts, rows, span, ring, bucket = 4, 4, 2, 8, 2, 2, 8, 16, 4
    q = rng.standard_normal((batch, heads, 1, width)).astype(np.float32)
    pk, pv = (rng.standard_normal((prompts, kv_heads, span, width)).astype(np.float32) for _ in range(2))
    sk, sv = (rng.standard_normal((batch, kv_heads, ring, width)).astype(np.float32) for _ in range(2))
    lane_prompt, prefix_lengths, lengths = [0, 1, 0, 1], [5, 8], [3, 1, 4, 2]  # every length within the bucket
    lane_rows, lane_slot = [0, 2, 1, 3], [0, 2, 1, 3]
    for row, low in ((9, 6), (3, 0), (1, None), (0, None)):  # inside the ring, at its start, wrapping around
      out = rollout_attention(Tensor(q), Tensor(pk), Tensor(pv), Tensor(sk), Tensor(sv.transpose(0, 1, 3, 2).copy()),
                              Tensor(lane_rows, dtype=dtypes.int32), Tensor(lane_slot, dtype=dtypes.int32),
                              Tensor(prefix_lengths, dtype=dtypes.int32), Tensor(lengths, dtype=dtypes.int32), row,
                              4, bucket, low).numpy()
      for b in range(batch):
        g = lane_prompt[b]
        own = [w for w in range(ring) if (row - w) % ring < lengths[b]]
        keys = np.concatenate([pk[g, :, :prefix_lengths[g]], sk[b][:, own]], 1)
        values = np.concatenate([pv[g, :, :prefix_lengths[g]], sv[b][:, own]], 1)
        for h in range(heads):
          scores = keys[h // (heads // kv_heads)] @ q[b, h, 0] / np.sqrt(width)
          p = np.exp(scores - scores.max())
          np.testing.assert_allclose(out[b, h, 0], p @ values[h // (heads // kv_heads)] / p.sum(), rtol=1e-4, atol=1e-5)
    with self.assertRaises(ValueError):
      rollout_attention(Tensor(q), Tensor(pk), Tensor(pv), Tensor(sk), Tensor(sv.transpose(0, 1, 3, 2).copy()),
                        Tensor(lane_rows, dtype=dtypes.int32), Tensor(lane_slot, dtype=dtypes.int32),
                        Tensor(prefix_lengths, dtype=dtypes.int32), Tensor(lengths, dtype=dtypes.int32), 9, 4, 12, None)

  def test_shared_prefix_ahead_of_every_slot(self):
    from tinygrad.llm.nemotron_h_attention import rollout_attention
    rng = np.random.default_rng(2)
    batch, heads, kv_heads, width, prompts, rows, span, ring, shared = 4, 4, 2, 8, 2, 2, 8, 16, 12
    q = rng.standard_normal((batch, heads, 1, width)).astype(np.float32)
    pk, pv = (rng.standard_normal((prompts, kv_heads, span, width)).astype(np.float32) for _ in range(2))
    sk, sv = (rng.standard_normal((batch, kv_heads, ring, width)).astype(np.float32) for _ in range(2))
    hk, hv = (rng.standard_normal((1, kv_heads, shared, width)).astype(np.float32) for _ in range(2))
    lane_prompt, prefix_lengths, lengths, shared_length = [0, 1, 0, 1], [5, 8], [3, 1, 4, 2], 9
    lane_rows, lane_slot = [0, 2, 1, 3], [0, 2, 1, 3]
    for row, chunk, bucket, low in ((9, 4, None, None), (9, 4, 4, 6), (1, 4, 4, None)):
      out = rollout_attention(Tensor(q), Tensor(pk), Tensor(pv), Tensor(sk), Tensor(sv.transpose(0, 1, 3, 2).copy()),
                              Tensor(lane_rows, dtype=dtypes.int32), Tensor(lane_slot, dtype=dtypes.int32),
                              Tensor(prefix_lengths, dtype=dtypes.int32), Tensor(lengths, dtype=dtypes.int32), row,
                              chunk, bucket, low,
                              (Tensor(hk), Tensor(hv), Tensor([shared_length], dtype=dtypes.int32))).numpy()
      for b in range(batch):
        g = lane_prompt[b]
        own = [w for w in range(ring) if (row - w) % ring < lengths[b]]
        keys = np.concatenate([hk[0, :, :shared_length], pk[g, :, :prefix_lengths[g]], sk[b][:, own]], 1)
        values = np.concatenate([hv[0, :, :shared_length], pv[g, :, :prefix_lengths[g]], sv[b][:, own]], 1)
        for h in range(heads):
          scores = keys[h // (heads // kv_heads)] @ q[b, h, 0] / np.sqrt(width)
          p = np.exp(scores - scores.max())
          np.testing.assert_allclose(out[b, h, 0], p @ values[h // (heads // kv_heads)] / p.sum(), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
