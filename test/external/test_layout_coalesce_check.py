#!/usr/bin/env python3
"""Seed of the layout/mapping IR (docs/layout-mapping-ir-design-20260625.md): the static coalescing predicate.
Proves coalescing is queryable statically from the RANGE/INDEX algebra (stride = addr(lane+1)-addr(lane))."""
import unittest
from tinygrad.uop.ops import UOp, AxisType
from extra.qk_layout_coalesce_check import axis_stride, is_coalesced, vector_width

class TestLayoutCoalesceCheck(unittest.TestCase):
  def setUp(self):
    self.lane = UOp.range(32, 0, AxisType.LOCAL)
    self.kred = UOp.range(16, 1, AxisType.REDUCE)

  def test_coalesced_unit_stride(self):
    idx = self.lane + self.kred*32                 # consecutive lanes -> consecutive addresses
    self.assertEqual(axis_stride(idx, self.lane), 1)
    self.assertTrue(is_coalesced(idx, self.lane))
    self.assertEqual(vector_width(idx, self.lane), 4)

  def test_strided_not_coalesced(self):
    idx = self.lane*256 + self.kred                # lane on output rows (our gate/up GEMV pattern) -> stride 256
    self.assertEqual(axis_stride(idx, self.lane), 256)
    self.assertFalse(is_coalesced(idx, self.lane))
    self.assertEqual(vector_width(idx, self.lane), 1)

  def test_reduce_axis_stride(self):
    idx = self.lane + self.kred*32
    self.assertEqual(axis_stride(idx, self.kred), 32)   # the K-reduce axis has stride 32 here

if __name__ == "__main__":
  unittest.main()
