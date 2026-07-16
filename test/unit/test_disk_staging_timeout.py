from types import SimpleNamespace

import pytest

from tinygrad.runtime.ops_disk import DiskAllocator, DiskBuffer
import tinygrad.runtime.ops_disk as ops_disk


def _allocator(size=4096):
  allocator = object.__new__(DiskAllocator)
  allocator.dev = SimpleNamespace(size=size)
  return allocator


def test_sync_disk_staging_wait_times_out_with_context(monkeypatch):
  allocator = _allocator()
  src = DiskBuffer(allocator.dev, 16)
  ticks = iter(range(100))
  monkeypatch.setattr(ops_disk.time, "perf_counter", lambda: next(ticks) / 1000)

  with pytest.raises(RuntimeError) as exc:
    list(allocator._copyout_sharded(src, 16, lambda: None, 4096, use_ioring=False, timeout_ms=3,
                                    wait_info=lambda: "buffer=2, required timeline=9, current signal=4"))

  message = str(exc.value)
  assert "DISK copy staging wait timeout: 3 ms" in message
  assert "buffer=2" in message
  assert "required timeline=9" in message
  assert "current signal=4" in message


def test_sync_disk_staging_success_path_is_unchanged(monkeypatch):
  allocator = _allocator()
  src = DiskBuffer(allocator.dev, 16)
  class Staging:
    def __init__(self): self.data = memoryview(bytearray(4096))
    def view(self, size): return self.data[:size]
  staging = Staging()
  monkeypatch.setattr(DiskAllocator, "_copyout", lambda self, dest, src: dest.__setitem__(slice(None), bytes([7]) * len(dest)))

  batches = list(allocator._copyout_sharded(src, 16, lambda: (staging, 5), 4096, use_ioring=False, timeout_ms=3))

  assert batches == [((staging, 5), 0, 0, 16)]
  assert staging.data[:16].tolist() == [7] * 16
