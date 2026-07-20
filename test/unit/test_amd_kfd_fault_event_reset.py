import ctypes

import pytest

from tinygrad.runtime import ops_amd
from tinygrad.runtime.autogen import kfd


def _iface() -> ops_amd.KFDIface:
  iface = ops_amd.KFDIface.__new__(ops_amd.KFDIface)
  iface.queue_event_arr = (kfd.struct_kfd_event_data * 3)(
    kfd.struct_kfd_event_data(kfd_event_data_ext=0x101, event_id=11),
    kfd.struct_kfd_event_data(kfd_event_data_ext=0x102, event_id=12),
    kfd.struct_kfd_event_data(kfd_event_data_ext=0x103, event_id=13),
  )
  iface.queue_event_arr_ptr = ctypes.addressof(iface.queue_event_arr)
  return iface


def test_kfd_wait_does_not_join_stale_memory_fault_to_current_hw_exception(monkeypatch):
  iface = _iface()
  iface.queue_event_arr[1].memory_exception_data.gpu_id = 10727
  iface.queue_event_arr[1].memory_exception_data.va = 0xFFFFFFBFE000

  def wait_events(_fd, **kwargs):
    assert kwargs["events_ptr"] == iface.queue_event_arr_ptr
    assert [row.event_id for row in iface.queue_event_arr] == [11, 12, 13]
    assert [row.kfd_event_data_ext for row in iface.queue_event_arr] == [0x101, 0x102, 0x103]
    assert iface.queue_event_arr[1].memory_exception_data.gpu_id == 0
    assert iface.queue_event_arr[1].memory_exception_data.va == 0
    iface.queue_event_arr[2].hw_exception_data.gpu_id = 10727
    iface.queue_event_arr[2].hw_exception_data.memory_lost = 1

  monkeypatch.setattr(ops_amd.KFDIface, "kfd", object())
  monkeypatch.setattr(kfd, "AMDKFD_IOC_WAIT_EVENTS", wait_events)
  with pytest.raises(RuntimeError) as raised:
    iface.sleep(1)
  assert "MMU fault" not in str(raised.value)
  assert "HW fault:" in str(raised.value)
  assert "memory_lost=1" in str(raised.value)


def test_kfd_wait_does_not_join_stale_hw_exception_to_current_memory_fault(monkeypatch):
  iface = _iface()
  iface.queue_event_arr[2].hw_exception_data.gpu_id = 10727
  iface.queue_event_arr[2].hw_exception_data.memory_lost = 1

  def wait_events(_fd, **_kwargs):
    assert iface.queue_event_arr[2].hw_exception_data.gpu_id == 0
    assert iface.queue_event_arr[2].hw_exception_data.memory_lost == 0
    iface.queue_event_arr[1].memory_exception_data.gpu_id = 10727
    iface.queue_event_arr[1].memory_exception_data.va = 0x100000000
    iface.queue_event_arr[1].memory_exception_data.failure.NotPresent = 1

  monkeypatch.setattr(ops_amd.KFDIface, "kfd", object())
  monkeypatch.setattr(kfd, "AMDKFD_IOC_WAIT_EVENTS", wait_events)
  with pytest.raises(RuntimeError) as raised:
    iface.sleep(1)
  assert "MMU fault: 0x100000000" in str(raised.value)
  assert "NotPresent=1" in str(raised.value)
  assert "HW fault" not in str(raised.value)
