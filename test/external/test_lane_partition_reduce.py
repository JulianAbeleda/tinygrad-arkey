#!/usr/bin/env python3
import unittest
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from extra.qk_layout_coalesce_check import axis_stride
from extra.qk_lane_partition_reduce import LanePartition, LanePartitionError, lane_partition_reduce_sum, q4k_packed_word_index
from extra.amd_warp_reduce import WARP

_DEV_OK = Device.DEFAULT == "AMD"
NB = 4

class TestLanePartitionReduceStructural(unittest.TestCase):
  def test_partition_expressions(self):
    lane = UOp.range(WARP, 0, AxisType.WARP)
    part = LanePartition(lane)
    self.assertEqual(part.block_groups, 4)
    self.assertEqual(part.lane_expr().substitute({lane: lane.const_like(19)}).simplify().arg, 19)
    self.assertEqual(part.block_group.substitute({lane: lane.const_like(19)}).simplify().arg, 2)
    self.assertEqual(part.word_col.substitute({lane: lane.const_like(19)}).simplify().arg, 3)

  def test_q4k_word_index_is_stride_one_inside_subgroup(self):
    lane = UOp.range(WARP, 0, AxisType.WARP)
    part = LanePartition(lane)
    idx = q4k_packed_word_index(UOp.const(dtypes.weakint, 128), 3, part)
    for l in range(7):
      a = idx.substitute({lane: lane.const_like(l)}).simplify().arg
      b = idx.substitute({lane: lane.const_like(l+1)}).simplify().arg
      self.assertEqual(b-a, 1)
    self.assertEqual(axis_stride(idx, lane), 1)

  def test_lower_sum_emits_bpermute_customi(self):
    lane = UOp.range(WARP, 0, AxisType.WARP)
    part = LanePartition(lane)
    out = lane_partition_reduce_sum(lane.cast(dtypes.float32), part)
    nodes = list(out.toposort())
    self.assertFalse(any(u.op is Ops.REDUCE for u in nodes))
    self.assertEqual(len([u for u in nodes if u.op is Ops.CUSTOMI and "ds_bpermute" in str(u.arg)]), 5)

  def test_rejects_unsupported_partition(self):
    lane = UOp.range(WARP, 0, AxisType.WARP)
    with self.assertRaises(LanePartitionError): LanePartition(lane, lane_extent=16).validate()
    with self.assertRaises(LanePartitionError): LanePartition(lane, words_per_group=7).validate()
    with self.assertRaises(LanePartitionError): q4k_packed_word_index(UOp.const(dtypes.weakint, 0), 8, LanePartition(lane))


def _partition_sum_kernel(name="lane_partition_sum"):
  def k(y:UOp, x:UOp) -> UOp:
    gid = UOp.special(NB, "gidx0")
    lane = UOp.special(WARP, "lidx0")
    part = LanePartition(lane)
    vals = x.reshape(NB, 4, 8)[gid]
    partial = vals[part.block_group, part.word_col].cast(dtypes.float32)
    total = lane_partition_reduce_sum(partial, part)
    return y.reshape(NB, WARP)[gid, lane].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return k

@unittest.skipUnless(_DEV_OK, "lane-partition reduction uses AMD wave32 ds_bpermute")
class TestLanePartitionReduceAMD(unittest.TestCase):
  def test_partitioned_sum_correct(self):
    xnp = np.random.default_rng(0).standard_normal((NB, 4, 8)).astype(np.float32)
    y = Tensor.empty(NB * WARP, dtype=dtypes.float32).custom_kernel(Tensor(xnp).realize(), fxn=_partition_sum_kernel())[0]
    got = y.realize().numpy().reshape(NB, WARP)
    ref = np.broadcast_to(xnp.sum(axis=(1, 2), keepdims=False)[:, None], (NB, WARP))
    self.assertTrue(np.allclose(got, ref, atol=1e-4), f"partitioned sum wrong, max_err={np.abs(got-ref).max()}")

  def test_emits_special_lidx_and_bpermute(self):
    out = Tensor.empty(NB * WARP, dtype=dtypes.float32).custom_kernel(
      Tensor(np.ones((NB, 4, 8), dtype=np.float32)).realize(), fxn=_partition_sum_kernel("lane_partition_src"))[0]
    src = ""
    for call in compile_linear(out.schedule_linear()).src:
      p = call.src[0]
      if p.op is Ops.PROGRAM and "lane_partition_src" in p.arg.name:
        src = next((u.arg for u in p.toposort() if u.op is Ops.SOURCE), "")
        break
    self.assertIn("lidx0", src)
    self.assertIn("ds_bpermute", src)

if __name__ == "__main__":
  unittest.main()
