from tinygrad.runtime.support.hcq import HCQBuffer, HCQInterfaceAllocator

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
