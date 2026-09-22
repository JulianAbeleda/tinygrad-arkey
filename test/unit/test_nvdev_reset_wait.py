"""Regression for the NV reset-wait ordering (upstream 7eb197b1b).

The old code only waited for the falcon when WPR2 was up and a full reset had
just been issued. On a clean boot it could race ahead before the falcon was
ready. These tests pin the new invariant: `_early_ip_init` always waits for the
reset, and still issues a full PCI reset only when WPR2 is actually up.

Hermetic: the fake device touches no RM or GPU hardware.
"""
import pytest

from tinygrad.runtime.support.nv import nvdev


class _FakeReg:
  def __init__(self, value=0, bitfields=None):
    self._value = value
    self._bitfields = bitfields

  def read(self): return self._value
  def read_bitfields(self): return self._bitfields


class _FakePCIDev:
  def __init__(self):
    self.reset_calls = 0

  def write_config_flush(self, reg, value, width): pass
  def read_config(self, reg, width): return 0
  def reset(self): self.reset_calls += 1


class _FakeIP:
  def __init__(self, nvdev):
    self.nvdev = nvdev
    self.wait_calls = 0

  def wait_for_reset(self): self.wait_calls += 1


@pytest.fixture
def make_dev(monkeypatch):
  monkeypatch.setattr(nvdev, "NV_FLCN", _FakeIP)
  monkeypatch.setattr(nvdev, "NV_FLCN_COT", _FakeIP)
  monkeypatch.setattr(nvdev, "NV_GSP", _FakeIP)
  monkeypatch.setattr(nvdev.time, "sleep", lambda _s: None)

  def _make(wpr2_up):
    dev = nvdev.NVDev.__new__(nvdev.NVDev)
    dev.reg_names = set()
    dev.reg_offsets = {}
    dev.include = lambda *a, **k: None
    dev.pci_dev = _FakePCIDev()

    def reg(name):
      if name == "NV_PFB_PRI_MMU_WPR2_ADDR_HI":
        return _FakeReg(value=1 if wpr2_up else 0)
      if name == "NV_PMC_BOOT_0":
        return _FakeReg(value=0x1b)
      if name == "NV_PMC_BOOT_42":
        return _FakeReg(bitfields={"architecture": 0x1b, "implementation": 0x02})
      raise AssertionError(f"unexpected register: {name}")

    dev.reg = reg
    return dev

  return _make


def test_wait_for_reset_is_always_called_on_clean_boot(make_dev):
  dev = make_dev(wpr2_up=False)
  dev._early_ip_init()
  assert dev.pci_dev.reset_calls == 0
  assert dev.flcn.wait_calls == 1


def test_reset_and_wait_when_wpr2_is_up(make_dev):
  dev = make_dev(wpr2_up=True)
  dev._early_ip_init()
  assert dev.pci_dev.reset_calls == 1
  assert dev.flcn.wait_calls == 1
