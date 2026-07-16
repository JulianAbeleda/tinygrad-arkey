import os

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import KernelInfo, ScheduleHints

from extra.qk.q4k_q8_mmq_uop import describe_q4k_q8_mmq_wmma, emit_q4k_q8_mmq_wmma
from test.unit.test_q4k_q8_mmq_uop import _fixture, independent_packed_byte_reference


pytestmark = pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")


def _n_upcast_axis(m: int) -> int:
  # TC consumes one 16-wide tile from each output axis.  A 16-row contraction has no outer M range, so outer N is
  # position zero; a multiworkgroup M grid retains outer M before outer N, making N position one.
  return 0 if m == 16 else 1


@pytest.mark.parametrize("m,n", ((16, 32), (32, 32), (32, 64)))
def test_generic_tc_n_upcast_multiworkgroup_matmul(m: int, n: int):
  rng = np.random.default_rng(20260715 + m + n)
  a = rng.integers(-16, 17, size=(m, 32), dtype=np.int8)
  b = rng.integers(0, 16, size=(32, n), dtype=np.int8)
  opts = (Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.UPCAST, _n_upcast_axis(m), 2))

  got = Tensor(a, device="AMD").matmul(Tensor(b, device="AMD"), dtype=dtypes.int32).contiguous(
    arg=ScheduleHints(opts_to_apply=opts, name=f"generic_tc_n_upcast_{m}x{n}")).realize().numpy()

  np.testing.assert_array_equal(got, a.astype(np.int32) @ b.astype(np.int32))


@pytest.mark.parametrize("m,n", ((16, 32), (32, 32), (32, 64)))
def test_q4k_q8_packed_random_byte_tc_n_upcast_grid(m: int, n: int):
  k = 512
  words, xq, xscale = _fixture(m, n, k)
  reference = independent_packed_byte_reference(words, xq, xscale, m=m, n=n, k=k)
  base = emit_q4k_q8_mmq_wmma(describe_q4k_q8_mmq_wmma(m=m, n=n, k=k))

  def kernel(out, packed, activation, scale):
    sink = base(out, packed, activation, scale)
    return sink.replace(arg=KernelInfo(name=f"q4k_q8_tc_n_upcast_{m}x{n}x{k}", opts_to_apply=(
      Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.UPCAST, _n_upcast_axis(m), 2))))

  got = Tensor.empty(m, n, dtype=dtypes.float32, device="AMD").custom_kernel(
    Tensor(words, device="AMD"), Tensor(xq.reshape(-1), device="AMD"),
    Tensor(xscale.reshape(-1), device="AMD"), fxn=kernel)[0].numpy()

  np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-3)
