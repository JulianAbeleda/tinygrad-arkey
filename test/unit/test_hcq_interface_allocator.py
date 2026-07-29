from tinygrad.runtime.support.hcq import HCQBuffer, HCQInterfaceAllocator
from tinygrad.runtime.ops_amd import AMDAllocator

class FakeIface:
  def __init__(self): self.freed, self.mapped = [], []
  def free(self, buf): self.freed.append(buf)
  def map(self, buf): self.mapped.append(buf); return "mapping"

class FakeDevice:
  def __init__(self): self.iface = FakeIface()

def test_interface_allocator_forwards_base_buffer_map_and_free():
  dev, allocator = FakeDevice(), object.__new__(HCQInterfaceAllocator)
  allocator.dev = dev
  base = HCQBuffer(0x1000, 0x100, owner=dev)
  view = base.offset(0x20, 0x20)

  assert allocator._map(view) == "mapping"
  allocator._do_free(base)
  assert dev.iface.mapped == [base]
  assert dev.iface.freed == [base]


def test_amd_interface_allocator_publishes_its_large_allocation_granularity():
  allocator = object.__new__(AMDAllocator)
  allocator.dev = type("FakeAMDDevice", (), {"is_am":lambda self:True})()
  assert allocator.allocation_granularity == 2 << 20
  allocator.dev = type("FakeKFDDevice", (), {"is_am":lambda self:False})()
  assert allocator.allocation_granularity is None
