import unittest

import numpy as np

from tinygrad import Tensor, TinyJit
from tinygrad.llm.nemotron_h_decode import mamba_decode_buffers, mamba_decode_step
from test.unit.test_nemotron_h_model import tiny_model

BATCH, STEPS = 3, 5


def _setup():
  model = tiny_model()
  block = model.blk[0]
  rng = np.random.default_rng(7)
  # a real prefix gives a nonzero conv tail and state to start from
  hidden0 = Tensor(rng.standard_normal((BATCH, 6, model.config.dim)).astype(np.float32))
  _, cache = block.cached(hidden0, None)
  inputs = [rng.standard_normal((BATCH, 1, model.config.dim)).astype(np.float32) for _ in range(STEPS)]
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


if __name__ == "__main__":
  unittest.main()
