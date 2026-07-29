import inspect
import warnings

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, Ops, UOp


def _increment(out:UOp, src:UOp) -> UOp:
  idx = UOp.range(out.numel(), 0)
  return out.flatten()[idx].store(src.flatten()[idx] + 1).end(idx).sink(arg=KernelInfo(name="uop_program_increment"))


def _call(method:str, *, grad_fxn=None) -> list[Tensor]:
  output = Tensor.empty(4, dtype=dtypes.float, device="PYTHON")
  source = Tensor([1, 2, 3, 4], dtype=dtypes.float, device="PYTHON")
  return getattr(output, method)(source, fxn=_increment, grad_fxn=grad_fxn)


def test_uop_program_and_compatibility_signatures_match():
  assert inspect.signature(Tensor.uop_program) == inspect.signature(Tensor.custom_kernel)


def test_uop_program_is_lazy_multi_output_and_forwards_gradient_callback():
  def gradient(*args): return args
  outputs = _call("uop_program", grad_fxn=gradient)
  assert isinstance(outputs, list) and len(outputs) == 2
  assert [x.shape for x in outputs] == [(4,), (4,)]
  assert all(x.uop.op is Ops.AFTER for x in outputs)
  call = outputs[0].uop.src[1]
  assert call.op is Ops.CALL and outputs[1].uop.src[1] is call
  assert call.arg.grad_fxn is gradient
  np.testing.assert_array_equal(outputs[0].numpy(), np.array([2, 3, 4, 5], dtype=np.float32))


def test_custom_kernel_is_a_silent_equivalent_compatibility_wrapper():
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    legacy = _call("custom_kernel")
  canonical = _call("uop_program")
  assert caught == []
  assert isinstance(legacy, list) and len(legacy) == len(canonical) == 2
  assert [x.uop.op for x in legacy] == [x.uop.op for x in canonical] == [Ops.AFTER, Ops.AFTER]
  assert legacy[0].uop.src[1].op is canonical[0].uop.src[1].op is Ops.CALL
  np.testing.assert_array_equal(legacy[0].numpy(), canonical[0].numpy())
