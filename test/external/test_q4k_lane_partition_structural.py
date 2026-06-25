#!/usr/bin/env python3
import unittest
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops
from extra.q4_k_gemv_primitive import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK
from extra.qk_q4k_lane_partition_gemv import q4k_lane_partition_gemv_kernel

_DEV_OK = Device.DEFAULT == "AMD"
ROWS, K = 4, 1024

def _src():
  kb = K // Q4_K_BLOCK_ELEMS
  words = Tensor(np.zeros(ROWS * kb * Q4K_WORDS_PER_BLOCK, dtype=np.uint32), dtype=dtypes.uint32).realize()
  x = Tensor(np.ones(K, dtype=np.float16)).realize()
  out = Tensor.empty(ROWS, dtype=dtypes.float32).custom_kernel(words, x, fxn=q4k_lane_partition_gemv_kernel(ROWS, K))[0]
  for call in compile_linear(out.schedule_linear()).src:
    p = call.src[0]
    if p.op is Ops.PROGRAM and "q4k_lane_partition_gemv" in p.arg.name:
      return next((u.arg for u in p.toposort() if u.op is Ops.SOURCE), "")
  return ""

@unittest.skipUnless(_DEV_OK, "q4k lane-partition structural source is AMD wave32 gated")
class TestQ4KLanePartitionStructural(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.src = _src()
    cls.tight = cls.src.replace(" ", "")

  def test_lane_partition_shapes_render(self):
    self.assertIn("lidx0", self.src)
    self.assertTrue("%8" in self.tight or "&7" in self.tight, "word_col lane%8 did not render")
    self.assertTrue("/8" in self.tight or ">>3" in self.tight, "block_group lane//8 did not render")

  def test_wave_reduce_and_single_output_store_render(self):
    self.assertIn("ds_bpermute", self.src)
    self.assertEqual(self.src.count("*(data0_4+gidx0)"), 1, "expected one global output store instruction")

  def test_packed_word_loads_are_lane_offset(self):
    self.assertIn("data1_", self.src)
    self.assertIn("alu3+4", self.tight)
    self.assertIn("alu3+12", self.tight)
    self.assertIn("alu3+20", self.tight)
    self.assertIn("alu3+28", self.tight)

if __name__ == "__main__":
  unittest.main()
