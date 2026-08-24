import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.llm.packed_argmax import emit_native_finite_fp32_argmax, packed_argmax_finite_fp32


def _check(a:np.ndarray, axis:int):
  # Semantic matrix stays hermetic: some NV renderer versions do not support
  # arbitrary uint64 vector widths (for example an outer extent of six).  The
  # native [1,151936] target is qualified by the dedicated microgate instead.
  x = Tensor(a.astype(np.float32), device="PYTHON")
  for keepdim in (False, True):
    got = packed_argmax_finite_fp32(x, axis, keepdim).numpy()
    want = x.argmax(axis, keepdim).numpy()
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("shape", [(17,), (3, 19), (2, 3, 23), (2, 3, 4, 29)])
def test_packed_argmax_matches_ordinary_all_axes_random_finite(shape):
  # A fixed distribution plus explicit extremal finite values catches both
  # sign halves of the IEEE ordering without giving NaN a new policy.
  a = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
  a.flat[0], a.flat[-1] = np.finfo(np.float32).min, np.finfo(np.float32).max
  for axis in range(len(shape)): _check(a, axis)


def test_packed_argmax_first_index_ties_and_signed_zero():
  a = np.array([[3.0, 3.0, 2.0, 3.0], [-0.0, 0.0, -1.0, -0.0],
                [-4.0, -2.0, -2.0, -3.0]], dtype=np.float32)
  _check(a, 1)
  assert packed_argmax_finite_fp32(Tensor(a), 1).numpy().tolist() == [0, 0, 1]


def test_packed_argmax_axes_preserve_layout():
  a = np.array([[[1, 7, 3], [4, 2, 6]], [[9, 8, 0], [5, 11, 10]]], dtype=np.float32)
  for axis in (0, 1, 2, -1, -2, -3): _check(a, axis)


def test_packed_argmax_fail_closed_input_contract():
  with pytest.raises(ValueError, match="float32"):
    packed_argmax_finite_fp32(Tensor([1, 2], dtype=dtypes.float16))
  with pytest.raises(ValueError, match="non-scalar"):
    packed_argmax_finite_fp32(Tensor(1.0))
  with pytest.raises(ValueError, match="axis"):
    packed_argmax_finite_fp32(Tensor([1.0]), 1)


def test_native_argmax_emitter_contract():
  for threads in (256, 512, 1024): assert callable(emit_native_finite_fp32_argmax(151936, threads))
  with pytest.raises(ValueError, match="positive"): emit_native_finite_fp32_argmax(0)
  with pytest.raises(ValueError, match="threads"): emit_native_finite_fp32_argmax(17, 128)
