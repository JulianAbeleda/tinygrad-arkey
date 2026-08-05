"""CPU ownership proof for a copy-free, device-resident decode feedback ring.

One captured JIT has a fixed return allocation.  Feeding that return back into
the same capture aliases its next input and output, so CapturedJit's written-
input shadow is required.  Alternating two captures makes each input/output
pair physically disjoint: A always reads B and writes A, then B reads A and
writes B.  This is the bounded construction a production decode experiment
can use without weakening CapturedJit's generic alias firewall.
"""
from tinygrad import Tensor, TinyJit
from tinygrad.llm.feedback_pingpong import pingpong_capture_contract


def _step(x:Tensor) -> Tensor: return (x + 1).contiguous()


def test_one_capture_recurrent_feedback_needs_alias_shadow():
  step = TinyJit(_step)
  token = Tensor([0], dtype="int32", device="CPU").contiguous().realize()
  for _ in range(5): token = step(token).realize()
  assert token.item() == 5
  assert len(step.captured._written_input_shadows) == 1


def test_two_capture_pingpong_keeps_feedback_device_resident_without_shadow():
  steps = (TinyJit(_step), TinyJit(_step))
  token = Tensor([0], dtype="int32", device="CPU").contiguous().realize()
  observed = []
  for iteration in range(12):
    token = steps[iteration & 1](token).realize()
    observed.append(token.item())  # public streaming-token boundary remains

  assert observed == list(range(1, 13))
  assert all(not step.captured._written_input_shadows for step in steps)
  assert steps[0].captured.ret.uop.buf_uop is not steps[1].captured.ret.uop.buf_uop
  assert pingpong_capture_contract(steps)["admitted"]


def test_pingpong_fixed_return_identity_is_stable_across_replays():
  steps = (TinyJit(_step), TinyJit(_step))
  token = Tensor([0], dtype="int32", device="CPU").contiguous().realize()
  return_buffers = [set(), set()]
  for iteration in range(16):
    arm = iteration & 1
    token = steps[arm](token).realize()
    if steps[arm].captured is not None: return_buffers[arm].add(steps[arm].captured.ret.uop.buf_uop)

  assert all(len(buffers) == 1 for buffers in return_buffers)
  assert not return_buffers[0] & return_buffers[1]
  assert token.item() == 16


def test_pingpong_contract_fails_closed_before_both_arms_are_warm():
  steps = (TinyJit(_step), TinyJit(_step))
  assert pingpong_capture_contract(steps) == {"admitted": False, "reason": "capture_not_warm"}
