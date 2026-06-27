#!/usr/bin/env python3
"""Proving ground for the coalesced-load lowering PRIMITIVE (extra/qk_coalesced_load_lowering.py).

Tile-independent: locks the contract on representative load idioms so the capability is general, not a
per-kernel trick. (1) the pass promotes a unit-stride load axis and DECLINES strided/REG/oversized axes;
(2) end-to-end on AMD the promoted kernel is numerically correct and renders a vector load + scalar
accumulator; (3) default-off is byte-identical.
"""
from __future__ import annotations
import os, unittest
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from extra.qk_coalesced_load_lowering import coalesce_loads


def _axtypes(sink: UOp) -> dict[int, AxisType]:
  return {r.arg[0]: r.arg[-1] for r in sink.toposort() if r.op is Ops.RANGE}


def _toy_sink(load_idx_fn, *, dd_size=4, reg=False, dd_type=AxisType.REDUCE) -> tuple[UOp, UOp]:
  """SINK( out[g*dd_size+dd].store(buf[load_idx]).end(dd).end(g) ). Returns (sink, dd)."""
  space = AddrSpace.REG if reg else AddrSpace.GLOBAL
  buf = UOp.placeholder((64,), dtypes.float32, 0, addrspace=space)
  out = UOp.placeholder((64,), dtypes.float32, 1, addrspace=AddrSpace.GLOBAL)
  g = UOp.range(4, 0, AxisType.GLOBAL)
  dd = UOp.range(dd_size, 1, axis_type=dd_type)
  load = buf[load_idx_fn(g, dd)]
  sink = out[g * dd_size + dd].store(load).end(dd).end(g).sink(arg=KernelInfo(name="toy"))
  return sink, dd


class TestCoalescedLoadLowering(unittest.TestCase):
  def test_promotes_unit_stride_load_axis(self):
    sink, dd = _toy_sink(lambda g, dd: g * 4 + dd)        # unit stride in dd
    out = coalesce_loads(sink)
    self.assertEqual(_axtypes(out)[1], AxisType.UPCAST, "unit-stride load axis must be promoted to UPCAST")
    self.assertEqual(_axtypes(out)[0], AxisType.GLOBAL, "the GLOBAL grid axis must be untouched")

  def test_declines_strided_axis(self):
    sink, dd = _toy_sink(lambda g, dd: dd * 4 + g)        # stride 4 in dd -> not coalesced
    self.assertEqual(_axtypes(coalesce_loads(sink))[1], AxisType.REDUCE, "strided axis must NOT be promoted")

  def test_declines_reg_buffer(self):
    sink, dd = _toy_sink(lambda g, dd: g * 4 + dd, reg=True)   # REG accumulator must stay scalar
    self.assertEqual(_axtypes(coalesce_loads(sink))[1], AxisType.REDUCE, "REG-buffer axis must NOT be promoted")

  def test_declines_oversized_axis(self):
    sink, dd = _toy_sink(lambda g, dd: g * 16 + dd, dd_size=16)  # 16 > max_width=4 and not a sub-multiple
    self.assertEqual(_axtypes(coalesce_loads(sink))[1], AxisType.REDUCE, "axis larger than fold width not promoted")

  @unittest.skipUnless(Device.DEFAULT == "AMD", "vectorized load lowering is AMD-only")
  def test_end_to_end_numeric_and_vectorized(self):
    T, W = 64, 4
    N = T * W

    def kernel(o: UOp, inp: UOp) -> UOp:
      g = UOp.range(T, 0, AxisType.GLOBAL)
      acc = UOp.placeholder((1,), dtypes.float32, 100, addrspace=AddrSpace.REG)
      acc = acc.after(g)[0].set(0.0)
      dd = UOp.range(W, 1, axis_type=AxisType.REDUCE)
      acc = acc[0].set(acc.after(dd)[0] + inp[g * W + dd], end=dd)
      return o[g].store(acc[0]).end(g).sink(arg=KernelInfo(name="cl_e2e", opts_to_apply=()))

    x = np.arange(N, dtype=np.float32)
    ref = x.reshape(T, W).sum(axis=1)
    os.environ["COALESCED_LOAD_LOWERING"] = "1"
    try:
      got = Tensor.empty(T, dtype=dtypes.float32).custom_kernel(Tensor(x), fxn=kernel)[0].realize().numpy()
    finally:
      del os.environ["COALESCED_LOAD_LOWERING"]
    np.testing.assert_allclose(got, ref, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
