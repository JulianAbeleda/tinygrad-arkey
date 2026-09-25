"""Hermetic policy tests for relaxed Blackwell internal dependent-QMD membars."""
from tinygrad.runtime.ops_nv import QMD, _nv_relax_internal_membar

def _fake_dev():
  class _Iface: compute_class = 0xcdc0
  class _Dev: iface = _Iface()
  return _Dev()

def test_blackwell_internal_qmd_membar_relaxes_by_default(monkeypatch):
  monkeypatch.delenv("NV_RELAX_INTERNAL_QMD_MEMBAR", raising=False)
  qmd = QMD(dev=_fake_dev(), cwd_membar_type=1)
  assert _nv_relax_internal_membar(qmd)
  assert qmd.read("cwd_membar_type") == 0

def test_internal_qmd_membar_has_explicit_rollback(monkeypatch):
  monkeypatch.setenv("NV_RELAX_INTERNAL_QMD_MEMBAR", "0")
  qmd = QMD(dev=_fake_dev(), cwd_membar_type=1)
  assert not _nv_relax_internal_membar(qmd)
  assert qmd.read("cwd_membar_type") == 1

def test_qmd_carrying_a_release_keeps_its_membar(monkeypatch):
  # another queue (or the host) waiting on the release reads this grid's writes with no other barrier
  monkeypatch.delenv("NV_RELAX_INTERNAL_QMD_MEMBAR", raising=False)
  qmd = QMD(dev=_fake_dev(), cwd_membar_type=1, release0_enable=1)
  assert not _nv_relax_internal_membar(qmd)
  assert qmd.read("cwd_membar_type") == 1

def test_signal_restores_membars_relaxed_since_the_previous_signal(monkeypatch):
  # a relaxed k1 -> k2(release) chain let a second compute queue read k1's output stale (sampler NaN logits)
  from tinygrad.runtime.ops_nv import NVComputeQueue
  monkeypatch.delenv("NV_RELAX_INTERNAL_QMD_MEMBAR", raising=False)
  q = NVComputeQueue()
  chain = [QMD(dev=_fake_dev(), cwd_membar_type=1) for _ in range(3)]
  for qmd in chain[:2]:
    assert _nv_relax_internal_membar(qmd)
    q.relaxed_qmds.append(qmd)
  q.active_qmd = None  # the semaphore-method path; restoring happens before either path
  class _Sig: value_addr = 0x1000
  q.signal(_Sig(), 1)
  assert [qmd.read("cwd_membar_type") for qmd in chain] == [1, 1, 1] and q.relaxed_qmds == []
