#!/usr/bin/env python3
import unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp, graph_rewrite
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.schedule.rangeify import pm_mops
from extra.amd_warp_reduce import WARP
from extra.qk_coalesce_search import score_layout_transform

class TestLayoutTransform(unittest.TestCase):
  def test_layout_transform_is_movement(self):
    self.assertIn(Ops.LAYOUT_TRANSFORM, GroupOp.Movement)

  def test_shape_preserved_and_unknown_rejected(self):
    u = UOp.new_buffer("PYTHON", 64, dtypes.float).reshape((8, 8))
    lt = UOp(Ops.LAYOUT_TRANSFORM, u.dtype, (u,), "q4k_lane_partition")
    self.assertEqual(lt.shape, u.shape)
    with self.assertRaises(ValueError):
      _ = UOp(Ops.LAYOUT_TRANSFORM, u.dtype, (u,), "bad_transform").shape

  def test_tensor_sugar_preserves_shape(self):
    t = Tensor.empty(4, 4).layout_transform("q4k_lane_partition")
    self.assertEqual(t.shape, (4, 4))

  def test_apply_movement_op_identity_for_known_transform(self):
    r0, r1 = UOp.range(4, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOOP)
    self.assertEqual(apply_movement_op(Ops.LAYOUT_TRANSFORM, (4, 8), "q4k_lane_partition", (r0, r1)), (r0, r1))
    with self.assertRaises(RuntimeError):
      apply_movement_op(Ops.LAYOUT_TRANSFORM, (4, 8), "bad_transform", (r0, r1))

  def test_mop_index_rewrite_strips_inert_transform(self):
    buf = UOp.new_buffer("PYTHON", 64, dtypes.float).reshape((8, 8))
    lt = UOp(Ops.LAYOUT_TRANSFORM, buf.dtype, (buf,), "q4k_lane_partition")
    r0, r1 = UOp.range(8, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOOP)
    out = graph_rewrite(lt.index(r0, r1), pm_mops)
    self.assertEqual(out.op, Ops.INDEX)
    self.assertIs(out.src[0], buf.src[0])
    self.assertEqual(out.src[1].tuplize, (r0*8 + r1).tuplize)

  def test_layout_fn_survives_identity_transform(self):
    lane = UOp.range(WARP, 0, AxisType.WARP)
    score = score_layout_transform("q4k_lane_partition", lane)
    self.assertEqual(score.candidate.name, "lane_partition_q4k")
    self.assertEqual(score.stride, 1)
    self.assertGreater(score.score, 1000)

if __name__ == "__main__":
  unittest.main()
