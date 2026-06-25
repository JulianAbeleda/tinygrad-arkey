#!/usr/bin/env python3
import unittest

from tinygrad.dtype import dtypes
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite
from tinygrad.uop.symbolic import symbolic
from extra.qk_layout_fn import LayoutFn, LayoutFnError


def _buf(size=4096):
  return UOp.new_buffer("PYTHON", size, dtypes.float)


class TestLayoutFn(unittest.TestCase):
  def test_coeff_matches_shrink_permute_expand_outputs(self):
    r0, r1 = UOp.range(8, 0, AxisType.LOOP), UOp.range(16, 1, AxisType.LOOP)
    s0, s1 = apply_movement_op(Ops.SHRINK, (16, 16), ((3, 11), (0, 16)), (r0, r1))
    lf = LayoutFn.from_index(_buf().index(s0*16 + s1))
    self.assertEqual(lf.coeff(r0), 16)
    self.assertEqual(lf.coeff(r1), 1)

    p0, p1 = apply_movement_op(Ops.PERMUTE, (8, 16), (1, 0), (r1, r0))
    lf = LayoutFn.from_index(_buf().index(p0*16 + p1))
    self.assertEqual(lf.coeff(r0), 16)
    self.assertEqual(lf.coeff(r1), 1)

    e0, e1 = apply_movement_op(Ops.EXPAND, (1, 16), (8, 16), (r0, r1))
    lf = LayoutFn.from_index(_buf().index(e0*16 + e1))
    self.assertEqual(lf.coeff(r0), 0)
    self.assertEqual(lf.coeff(r1), 1)

  def test_coeff_matches_admissible_reshape_output(self):
    r0 = UOp.range(32, 0, AxisType.LOOP)
    # Flattening a single axis is affine/admissible; mixed-radix reshape cases are rejected below.
    (a0,) = apply_movement_op(Ops.RESHAPE, (32,), (32,), (r0,))
    lf = LayoutFn.from_index(_buf().index(a0*4))
    self.assertEqual(lf.coeff(r0), 4)

  def test_compose_matches_manual_substitution_matmul_style(self):
    row, col, lane = UOp.range(32, 0, AxisType.LOOP), UOp.range(64, 1, AxisType.LOOP), UOp.range(32, 2, AxisType.THREAD)
    a = LayoutFn.from_index(_buf().index(row*64 + col))
    b = LayoutFn.from_expr(lane + UOp.const(dtypes.weakint, 8))
    got = a.compose(b, col).idx
    want = graph_rewrite((row*64 + col).substitute({col: lane + UOp.const(dtypes.weakint, 8)}), symbolic)
    self.assertEqual(got.tuplize, want.tuplize)
    self.assertEqual(LayoutFn.from_expr(got).coeff(lane), 1)

  def test_compose_matches_manual_substitution_gemv_style(self):
    out, k, lane = UOp.range(12288, 0, AxisType.LOOP), UOp.range(4096, 1, AxisType.REDUCE), UOp.range(32, 2, AxisType.THREAD)
    weights = LayoutFn.from_index(_buf(12288*4096).index(out*4096 + k))
    thread_k = LayoutFn.from_expr(lane*4 + UOp.const(dtypes.weakint, 1))
    got = weights.compose(thread_k, k).idx
    want = graph_rewrite((out*4096 + k).substitute({k: lane*4 + UOp.const(dtypes.weakint, 1)}), symbolic)
    self.assertEqual(got.tuplize, want.tuplize)
    self.assertEqual(LayoutFn.from_expr(got).coeff(lane), 4)

  def test_masked_pad_raises(self):
    r0 = UOp.range(8, 0, AxisType.LOOP)
    (p0,) = apply_movement_op(Ops.PAD, (8,), ((1, 10),), (r0,))
    with self.assertRaises(LayoutFnError):
      LayoutFn.from_index(_buf().index(p0)).coeff(r0)

  def test_mixed_radix_reshape_raises(self):
    r0, r1 = UOp.range(4, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOOP)
    a0, _a1 = apply_movement_op(Ops.RESHAPE, (4, 8), (2, 16), (r0, r1))
    with self.assertRaises(LayoutFnError):
      LayoutFn.from_index(_buf().index(a0)).coeff(r0)

  def test_multi_range_compose_requires_rng(self):
    r0, r1 = UOp.range(8, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOOP)
    with self.assertRaises(LayoutFnError):
      LayoutFn.from_expr(r0*8 + r1).compose(LayoutFn.from_expr(r0))


if __name__ == "__main__":
  unittest.main()
