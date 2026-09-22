"""Regression coverage for TinyJit multi-output replay ownership.

The second returned tensor is an internal producer which is also consumed by
the first result.  A replay must refresh both return buffers, rather than only
the terminal argmax result.
"""
from tinygrad import Tensor, TinyJit


def test_returned_internal_tensor_is_refreshed_on_each_jit_replay():
  @TinyJit
  def classified(x):
    logits = x * 3
    return logits.argmax(-1, keepdim=True), logits

  # Seven calls includes capture plus five genuine replays.  Move the maximum
  # each time: merely checking a changing magnitude would miss a stale logits
  # buffer whose original argmax happens to remain valid.
  for step in range(7):
    winner = step % 8
    values = [-10.0] * 8
    values[winner] = float(step + 1)
    sampled, logits = classified(Tensor(values).contiguous().realize())
    got_logits = logits.tolist()
    assert got_logits[winner] == float((step + 1) * 3)
    assert sampled.item() == winner == logits.argmax(-1).item()
