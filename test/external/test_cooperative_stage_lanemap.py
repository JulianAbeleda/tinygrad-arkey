#!/usr/bin/env python3
"""Proving ground for CooperativeStageLaneMap (extra/qk_cooperative_stage_lanemap.py).

Locks the contract: the lane map's element index is unit-stride in the per-thread `w` axis (statically
coalescable), and an end-to-end staging kernel using it + COALESCED_LOAD_LOWERING renders a vectorized GLOBAL
load and is numerically correct -- for both the plain and the t<Tc-masked cache-row access.
"""
from __future__ import annotations
import contextlib, io, os, unittest
import numpy as np
from tinygrad import Tensor, dtypes, Device, Context
from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from extra.qk_cooperative_stage_lanemap import CooperativeStageLaneMap
from extra.qk_layout_coalesce_check import axis_stride, is_coalesced


class TestCooperativeStageLaneMap(unittest.TestCase):
  def test_validate_rejects_bad_config(self):
    with self.assertRaises(ValueError): CooperativeStageLaneMap(total=2000, threads=128, width=4).validate()  # 2000 % 512 != 0
    with self.assertRaises(ValueError): CooperativeStageLaneMap(total=2048, threads=128, width=3).validate()  # width not a fold width

  def test_elem_index_is_unit_stride_in_w(self):
    lm = CooperativeStageLaneMap(total=2048, threads=128, width=4)
    st, w = lm.axes()
    tid = UOp.special(128, "lidx0")
    i = lm.elem_index(st, tid, w)
    self.assertEqual(axis_stride(i, w), 1, "per-thread chunk axis must be unit-stride (coalescable)")
    self.assertTrue(is_coalesced(i, w))
    self.assertNotEqual(axis_stride(i, st), 1, "the stage axis is strided across the workgroup, not coalesced")

  @unittest.skipUnless(Device.DEFAULT == "AMD", "vectorized load lowering is AMD-only")
  def test_end_to_end_vectorized_and_correct(self):
    TK, Hd, THREADS, W = 16, 128, 128, 4
    total, MAXC, Tc = TK * Hd, 256, 130
    lm = CooperativeStageLaneMap(total=total, threads=THREADS, width=W)

    for masked in (False, True):
      def kernel(out: UOp, cache: UOp) -> UOp:
        tid = UOp.special(THREADS, "lidx0")
        def value(i: UOp) -> UOp:
          t, e = i // Hd, i % Hd
          if masked:
            t = (t < Tc).where(t, t.const_like(0))
          return cache[t * Hd + e]
        return lm.stage(out, tid, value).sink(arg=KernelInfo(name=f"coop_stage_{int(masked)}", opts_to_apply=()))

      cache = np.arange(MAXC * Hd, dtype=np.float32)
      buf = io.StringIO()
      os.environ["COALESCED_LOAD_LOWERING"] = "1"
      try:
        with contextlib.redirect_stdout(buf), Context(DEBUG=4):
          got = Tensor.empty(total, dtype=dtypes.float32).custom_kernel(Tensor(cache), fxn=kernel)[0].realize().numpy()
      finally:
        del os.environ["COALESCED_LOAD_LOWERING"]
      idx = np.arange(total); t, e = idx // Hd, idx % Hd
      tsafe = np.where(t < Tc, t, 0) if masked else t
      np.testing.assert_array_equal(got, cache[tsafe * Hd + e])
      self.assertIn("float4", buf.getvalue(), f"masked={masked}: global staging load must vectorize to float4")


if __name__ == "__main__":
  unittest.main()
