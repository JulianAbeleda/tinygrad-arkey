import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from tinygrad import Device
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import GraphAdmissionReason, GraphRunner


class TestMetalGraphAdmissionSynthetic(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from tinygrad.runtime.graph.metal import MetalGraph
    cls.MetalGraph = MetalGraph

  @staticmethod
  def metal_buf(offset, *, size=1):
    buf = MagicMock()
    buf.op, buf.device, buf.dtype, buf.size = Ops.SLICE, "METAL", dtypes.uint8, size
    base = MagicMock(dtype=dtypes.uint8)
    buf.src = (base, UOp.const(dtypes.weakint, offset))
    return buf

  @staticmethod
  def call(*bufs):
    return MagicMock(src=(MagicMock(op=Ops.PROGRAM),) + tuple(bufs))

  def admission(self, *bufs):
    with patch.object(GraphRunner, "_all_devs", return_value=[object()]):
      return self.MetalGraph.admission([], self.call(*bufs))

  def test_uint32_boundary_and_first_invalid_byte_without_large_allocation(self):
    assert self.admission(self.metal_buf(0xFFFFFFFF)).reason is GraphAdmissionReason.ADMITTED
    rejected = self.admission(self.metal_buf(0x100000000, size=64))
    assert rejected.reason is GraphAdmissionReason.BACKEND_BUFFER_OFFSET_WIDTH
    assert rejected.capability == "icb_buffer_offset_bits" and rejected.limit == 0xFFFFFFFF
    assert rejected.observed == 0x100000000
    assert len(rejected.resources) == 1
    resource = rejected.resources[0]
    assert resource.buffer_arg_index == 0 and resource.byte_offset == 0x100000000 and resource.byte_span == 64

  def test_reports_every_offending_argument_and_boolean_delegates(self):
    bufs = (self.metal_buf(0), self.metal_buf(0x100000000), self.metal_buf(0x100000100))
    with patch.object(GraphRunner, "_all_devs", return_value=[object()]):
      admission = self.MetalGraph.admission([], self.call(*bufs))
      assert [resource.buffer_arg_index for resource in admission.resources] == [1, 2]
      assert self.MetalGraph.supports_uop([], self.call(*bufs)) is False

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset):
    buf = MagicMock()
    if offset > 0:
      buf.op = Ops.SLICE
      src = MagicMock()
      src.dtype = dtypes.uint8
      buf.src = (src, UOp.const(dtypes.weakint, offset))
      buf.dtype = dtypes.uint8
    else:
      buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    return buf

  def call(self, *bufs):
    c = MagicMock()
    c.src = (MagicMock(op=Ops.PROGRAM),) + tuple(bufs)
    return c

  def test_supports_uop_normal_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF))) is True

  def test_supports_uop_overflow_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(0x100000000))) is False

  def test_supports_uop_nonmetal_buf(self):
    # non-SLICE ops should not be checked for offset
    buf = MagicMock()
    buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    self.MetalGraph.supports_uop([self.dev], self.call(buf))

if __name__ == "__main__":
  unittest.main()
