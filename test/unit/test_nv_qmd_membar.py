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
