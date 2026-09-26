import unittest

import numpy as np

from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.nemotron_h_decode import (mamba_decode_buffers, mamba_decode_step, mamba_replay_buffers,
                                             mamba_replay_buffers_step, mamba_replay_flush)
from test.unit.test_nemotron_h_model import tiny_model

BATCH, STEPS, RING = 3, 5, 4


def _setup(steps=STEPS):
  model = tiny_model()
  block = model.blk[0]
  rng = np.random.default_rng(7)
  # a real prefix gives a nonzero conv tail and state to start from
  hidden0 = Tensor(rng.standard_normal((BATCH, 6, model.config.dim)).astype(np.float32))
  _, cache = block.cached(hidden0, None)
  inputs = [rng.standard_normal((BATCH, 1, model.config.dim)).astype(np.float32) for _ in range(steps)]
  return block, cache, inputs


def _buffers(cache):
  return {key: Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize() for key, value in cache.items()}


def _reference(block, cache, inputs):
  outs, states = [], []
  for value in inputs:
    out, cache = block.cached(Tensor(value), cache, keep_graph=True)
    out, conv, state = (t.contiguous().realize() for t in (out, cache["conv"], cache["state"]))
    cache = {"conv": conv, "state": state}
    outs.append(out.numpy()); states.append((conv.numpy(), state.numpy()))
  return outs, states


def _rel(a, b): return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


class TestNemotronHDecode(unittest.TestCase):
  def _check(self, outs, states, ref_outs, ref_states):
    for step, (out, (conv, state), ref, (ref_conv, ref_state)) in enumerate(zip(outs, states, ref_outs, ref_states)):
      self.assertLessEqual(_rel(state, ref_state), 1e-6, f"state step {step}")
      np.testing.assert_allclose(conv, ref_conv, rtol=1e-6, atol=1e-7, err_msg=f"conv step {step}")
      np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5, err_msg=f"out step {step}")

  def test_step_matches_cached(self):
    block, cache, inputs = _setup()
    ref_outs, ref_states = _reference(block, cache, inputs)
    buffer = _buffers(cache)
    conv_buffer, state_buffer = buffer["conv"].uop.buffer, buffer["state"].uop.buffer
    outs, states = [], []
    for value in inputs:
      out, conv, state = mamba_decode_step(block, Tensor(value), buffer["conv"], buffer["state"])
      Tensor.realize(out, conv, state)
      self.assertIs(conv, buffer["conv"]); self.assertIs(state, buffer["state"])
      outs.append(out.numpy()); states.append((conv.numpy(), state.numpy()))
    # updated in place: the buffers never move
    self.assertIs(buffer["conv"].uop.buffer, conv_buffer)
    self.assertIs(buffer["state"].uop.buffer, state_buffer)
    self._check(outs, states, ref_outs, ref_states)

  def test_jit_replay_matches_cached(self):
    block, cache, inputs = _setup()
    ref_outs, ref_states = _reference(block, cache, inputs)
    buffer = _buffers(cache)
    step = TinyJit(lambda hidden: mamba_decode_buffers(block, hidden, buffer))
    outs, states = [], []
    for value in inputs:
      out = step(Tensor(value).realize())
      outs.append(out.numpy()); states.append((buffer["conv"].numpy(), buffer["state"].numpy()))
    self.assertIsNotNone(step.captured)
    self._check(outs, states, ref_outs, ref_states)

  def _replay(self, jit: bool, ring: int = RING):
    steps = 2 * ring + 3
    block, cache, inputs = _setup(steps)
    ref_outs, ref_states = _reference(block, cache, inputs)
    buffer = mamba_replay_buffers(block, cache["conv"], cache["state"], ring)
    if jit:
      slot_var, count_var = UOp.variable("slot", 0, ring - 1), UOp.variable("count", 1, ring)
      step_jit = TinyJit(lambda hidden, slot: mamba_replay_buffers_step(block, hidden, buffer, slot))
      flush_jit = TinyJit(lambda count: mamba_replay_flush(block, buffer, count))
      step = lambda hidden, slot: step_jit(hidden, slot_var.bind(slot))
      flush = lambda count: flush_jit(count_var.bind(count))
    else:
      step = lambda hidden, slot: mamba_replay_buffers_step(block, hidden, buffer, slot)
      flush = lambda count: mamba_replay_flush(block, buffer, count)
    flushed = []
    for index, value in enumerate(inputs):
      slot = index % ring
      out = step(Tensor(value).realize(), slot)
      np.testing.assert_allclose(out.numpy(), ref_outs[index], rtol=1e-5, atol=1e-5, err_msg=f"out step {index}")
      np.testing.assert_allclose(buffer["conv"].numpy(), ref_states[index][0], rtol=1e-6, atol=1e-7)
      if slot == ring - 1 or index == steps - 1:
        flush(slot + 1)
        flushed.append(index)
        self.assertLessEqual(_rel(buffer["state"].numpy(), ref_states[index][1]), 1e-6, f"state after step {index}")
    self.assertEqual(flushed, [ring - 1, 2 * ring - 1, steps - 1])

  def test_replay_matches_cached(self): self._replay(jit=False)
  def test_replay_jit_matches_cached(self): self._replay(jit=True)
  # the flush unrolls one term per ring slot; cover a ring size that is not a power of two
  def test_replay_jit_odd_ring_matches_cached(self): self._replay(jit=True, ring=3)

  def test_flush_reads_the_ring_in_place_with_the_packed_operands_bits(self):
    # the flush weights each ring column where it reads it; the earlier form built a packed, transposed copy of
    # x*dt times the weights first: the same products in the same order, so the folded state is bit-identical
    from tinygrad.llm.nemotron_h_decode import _heads, _ring_weights
    block, cache, _ = _setup()
    buffer = mamba_replay_buffers(block, cache["conv"].expand(BATCH, *cache["conv"].shape[1:]),
                                  cache["state"].expand(BATCH, *cache["state"].shape[1:]), RING)
    rng = np.random.default_rng(9)
    for key, scale in (("ring_x", 1.0), ("ring_a", -0.3), ("ring_b", 1.0)):
      buffer[key].assign(Tensor((rng.standard_normal(buffer[key].shape) * scale).astype(np.float32))).realize()
    heads = block.config.ssm_heads
    batch, _, head_dim, ring = buffer["ring_x"].shape
    for count in (RING, 3):
      weights, decay = _ring_weights(buffer["ring_a"], count)
      pack = (buffer["ring_x"] * weights.contiguous().unsqueeze(2)).transpose(-1, -2).reshape(
        batch, heads, ring * head_dim).cat(decay.reshape(batch, heads, 1), dim=-1).contiguous()
      b = _heads(buffer["ring_b"], heads)
      packed = pack[:, :, ring * head_dim:].reshape(batch, heads, 1, 1) * buffer["state"]
      for j in range(ring):
        packed = packed + pack[:, :, j * head_dim:(j + 1) * head_dim].reshape(batch, heads, head_dim, 1) * b[:, :, j:j + 1]
      packed = packed.numpy()
      got = mamba_replay_flush(block, buffer, count).numpy()
      np.testing.assert_array_equal(got.view(np.uint32), packed.view(np.uint32))


if __name__ == "__main__":
  unittest.main()
