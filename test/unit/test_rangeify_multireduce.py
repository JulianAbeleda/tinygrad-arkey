"""Generic scheduler boundary for a reduction feeding two incompatible loops.

No semantic attention node is used here.  The score reduction feeds both a
KV-reducing max and a KV-preserving second contraction.  Today rangeify stages
the score because one value cannot own both range lifetimes.  A future generic
scoped-register/nested-reduction representation can change this expectation.
"""
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops


def _fp16_contraction(red):
  body = red.src[0]
  while body.op is Ops.CAST: body = body.src[0]
  return red.arg[0] is Ops.ADD and body.op is Ops.MUL and tuple(x.dtype.scalar() for x in body.src) == (dtypes.float16, dtypes.float16)


def test_rangeify_stages_value_with_reducing_and_preserving_consumers():
  x = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
  y = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)
  z = Tensor.empty(1, 1, 16, 16, dtype=dtypes.float16)

  scores = x @ y.transpose(-2, -1)
  row_max = scores.max(axis=-1, keepdim=True)
  weights = (scores - row_max).exp()
  out = weights @ z

  calls = out.schedule_linear().src
  contraction_calls = [i for i, call in enumerate(calls)
                       if any(_fp16_contraction(red) for red in call.src[0].toposort() if red.op is Ops.REDUCE)]
  assert len(contraction_calls) == 2
  assert contraction_calls[0] != contraction_calls[1]
