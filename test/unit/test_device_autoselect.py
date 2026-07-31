from types import SimpleNamespace
from unittest.mock import patch

from tinygrad.device import _Device


def test_autoselect_probes_nv_before_cpu():
  calls = []

  def open_device(_self, device):
    calls.append(device)
    if device == "AMD": raise RuntimeError("AMD unavailable")
    return SimpleNamespace(device=device)

  with patch.object(_Device, "__getitem__", open_device):
    assert list(_Device().get_available_devices()) == ["NV", "METAL", "CPU"]

  assert calls == ["NV", "METAL", "AMD", "CPU"]
