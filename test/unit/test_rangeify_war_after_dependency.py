"""D2 substrate lock (m4-resadd-rangeify-substrate-scope-20260806.md section 4, S2):

``fix_assign`` must skip the WAR edge when the reader kernel's own backward slice already
carries the writer's ``AFTER`` (precompiled-output identity: the read is ordered after the
write, so the edge would be a false cycle), and must still raise for the genuine crossunder
shape where a *different* writer of the reader's buffer holds ``AFTER(s)``.

The reader shape is reproduced through the real pipeline: a precompiled writer whose output
is consumed by a nested precompiled boundary (so the reader kernel reads the produced buffer
raw *and* carries the writer's ``AFTER`` as a dependency argument).  Before D2 this exact
graph raises ``cycle detected in assign graph``; after D2 it schedules.
"""
import pytest

from tinygrad import Tensor, UOp, dtypes
from tinygrad.function import function
from tinygrad.schedule import create_linear_with_vars
from tinygrad.uop.ops import KernelInfo, Ops


N = 16


def _reader_kernel(out: UOp, words: UOp) -> UOp:
  row = UOp.special(N, "gidx0")
  return out[row].store(words[row] * 2.0).sink(arg=KernelInfo(name="k_reader"))


@function(precompile=True)
def _writer(x): return (x * 2 + 1).contiguous()


@function(precompile=True, allow_implicit=True)
def _consumer(w):
  out = Tensor.empty(N, dtype=dtypes.float32).contiguous()
  ret = UOp.custom_kernel(out.uop, w.uop, fxn=_reader_kernel)
  return Tensor(ret[0]).contiguous()


@function(precompile=True, allow_implicit=True)
def _wrapper(x):
  return _consumer(_writer(x))


def _schedule(out: Tensor):
  from tinygrad.callify import transform_to_call
  from tinygrad.tensor import _apply_map_to_tensors
  big_sink, becomes = transform_to_call(UOp.sink(out.uop))
  _apply_map_to_tensors(becomes, name="buffers")
  return create_linear_with_vars(big_sink)


def _call_names(linear) -> list[str]:
  return [getattr(getattr(u.src[0], "arg", None), "name", None)
          for u in linear.toposort() if u.op is Ops.CALL]


def test_precompiled_output_identity_reader_skips_false_cycle():
  """The reader reads the writer's buffer raw AND carries AFTER(s): the WAR edge is a false
  cycle and must be skipped.  Without the D2 skip this exact graph raises
  ``cycle detected in assign graph`` (verified on the delta tree)."""
  w_raw = Tensor.empty(N, dtype=dtypes.float32).contiguous()
  linear, _ = _schedule(_wrapper(w_raw))
  assert len(linear.src) >= 1
  assert "k_reader" in _call_names(linear)


def test_crossunder_assign_raise_preserved():
  """Genuine crossunder (upstream test_crossunder_assign shape): the reader of a buffer is
  not itself AFTER-ordered, so the WAR cycle is real and the guard must still raise."""
  a = Tensor.full((4,), 2).contiguous().realize()
  b = Tensor.full((4,), 3).contiguous().realize()
  c = a + 9
  a += b
  b += c
  with pytest.raises(RuntimeError, match="cycle"):
    Tensor.realize(a, b)
