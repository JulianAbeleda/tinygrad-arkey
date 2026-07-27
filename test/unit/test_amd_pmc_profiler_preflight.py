import errno

import pytest

from tinygrad.runtime import ops_amd
from tinygrad.runtime.autogen import kfd


def test_pmc_profiler_preflight_locks_requested_gpu(monkeypatch):
  calls = []

  def profiler(fd, **kwargs):
    calls.append((fd, kwargs))

  monkeypatch.setattr(kfd, "AMDKFD_IOC_PROFILER", profiler)
  fd = object()
  ops_amd._lock_pmc_profiler(fd, 10727)

  assert len(calls) == 1
  assert calls[0][0] is fd
  assert calls[0][1]["op"] == kfd.KFD_IOC_PROFILER_PMC
  assert calls[0][1]["pmc"].gpu_id == 10727
  assert calls[0][1]["pmc"].lock == 1
  assert calls[0][1]["pmc"].perfcount_enable == 1


def test_pmc_profiler_preflight_rejects_permission_failure(monkeypatch):
  def profiler(*_args, **_kwargs):
    raise OSError(errno.EPERM, "Operation not permitted")

  monkeypatch.setattr(kfd, "AMDKFD_IOC_PROFILER", profiler)
  with pytest.raises(RuntimeError, match="AMDKFD_IOC_PROFILER PMC ioctl.*disable PMC or fix the KFD profiler interface") as raised:
    ops_amd._lock_pmc_profiler(object(), 10727)
  assert isinstance(raised.value.__cause__, PermissionError)
