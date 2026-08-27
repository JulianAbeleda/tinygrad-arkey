from tinygrad.runtime.support.hcq import HCQBuffer, HCQInterfaceAllocator, _hcq_copyout_needs_presync
from tinygrad.runtime.ops_amd import AMDAllocator
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.llm import device_facts

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
  assert allocator.allocation_granularity == 4 << 10


def test_amd_interface_allocator_reports_live_allocatable_heap_bytes():
  heap = TLSFAllocator(1 << 20); heap.alloc(0x3000)
  mm = type("FakeMemoryManager", (), {"pa_allocator":heap})()
  impl = type("FakeAMDev", (), {"mm":mm})()
  iface = type("FakeIface", (), {"dev_impl":impl})()
  allocator = object.__new__(AMDAllocator)
  allocator.dev = type("FakeAMDDevice", (), {"is_am":lambda self:True, "iface":iface})()
  assert allocator.memory_stats() == (1 << 20, (1 << 20)-0x3000)


def test_default_memory_probe_falls_back_to_live_allocator(monkeypatch):
  expected = {"total_vram_bytes":16 << 30, "free_vram_bytes":15 << 30, "provenance":"live"}
  monkeypatch.setattr(device_facts, "_rocm_smi_memory_probe", lambda _device:(_ for _ in ()).throw(FileNotFoundError()))
  monkeypatch.setattr(device_facts, "_allocator_memory_probe", lambda _device:expected)
  assert device_facts._default_memory_probe("AMD") == expected

def test_copyout_presync_is_skipped_only_when_device_owns_source_ordering():
  assert _hcq_copyout_needs_presync(type("DefaultDevice", (), {})())
  assert not _hcq_copyout_needs_presync(type("OrderedCopyDevice", (), {"copyout_wait_orders_source":True})())
