from types import SimpleNamespace

from tinygrad.llm.device_facts import _tinygrad_target_probe
from tinygrad.llm.gguf_memory_scan import selected_gguf_backing_bytes
from tinygrad.runtime.ops_amd import AMDAllocator


def _allocator(is_am:bool) -> AMDAllocator:
  allocator = object.__new__(AMDAllocator)
  allocator.dev = SimpleNamespace(is_am=lambda: is_am)
  return allocator


def test_amd_allocator_reports_physical_default_vram_granularity():
  assert _allocator(False).allocation_granularity == 4 << 10
  assert _allocator(True).allocation_granularity == 2 << 20


def test_selected_large_am_gguf_uses_the_reported_hugepage_granularity(tmp_path):
  path = tmp_path/"selected.gguf"
  path.write_bytes(b"x" * ((8 << 20) + 1))
  assert selected_gguf_backing_bytes(path, _allocator(True).allocation_granularity) == 10 << 20


def test_target_probe_keeps_allocator_fact_without_rocminfo(monkeypatch):
  opened = SimpleNamespace(renderer=SimpleNamespace(arch="gfx1100", wave_size=32, max_workgroup_threads=None,
    max_workgroup_dimensions=None, shared_max=65536, lds_allocation_granularity=None),
    allocator=_allocator(True), is_aql=False)

  class FakeDevices:
    def __getitem__(self, _device): return opened

  import tinygrad.device, tinygrad.llm.device_facts
  monkeypatch.setattr(tinygrad.device, "Device", FakeDevices())
  monkeypatch.setattr(tinygrad.llm.device_facts.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
  facts = _tinygrad_target_probe("AMD")
  assert facts["backend"] == "AMD" and facts["architecture"] == "gfx1100" and facts["queue_mode"] == "PM4"
  assert facts["global_allocation_granularity"] == 2 << 20
