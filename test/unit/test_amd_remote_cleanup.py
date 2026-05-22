import unittest

from tinygrad.runtime.ops_amd import AMDDevice, PCIIface
from tinygrad.runtime.support.amd import AMD_RUNTIME_DEVICES
from tinygrad.runtime.support.system import APLRemotePCIDevice, PCIDevice, PCIIfaceBase, RemotePCIDevice

class TestAMDRemoteCleanup(unittest.TestCase):
  def test_amd_runtime_devices_includes_rx_7900_xtx(self):
    self.assertIn(0x744c, AMD_RUNTIME_DEVICES)

  def test_remote_pci_capability_flag(self):
    self.assertFalse(PCIDevice.is_remote)
    self.assertTrue(RemotePCIDevice.is_remote)
    self.assertTrue(APLRemotePCIDevice.is_remote)

  def test_pci_iface_locality_uses_remote_capability(self):
    iface = object.__new__(PCIIfaceBase)
    iface.pci_dev = type("FakeLocalPCI", (), {})()
    self.assertTrue(iface.is_local())

    iface.pci_dev = type("FakeRemotePCI", (), {"is_remote": True})()
    self.assertFalse(iface.is_local())

  def test_amd_device_remote_detection_uses_capability(self):
    iface = object.__new__(PCIIface)
    iface.pci_dev = type("RenamedRemotePCI", (), {"is_remote": True})()
    dev = object.__new__(AMDDevice)
    dev.iface = iface
    self.assertTrue(dev.is_remote_pci())

    iface.pci_dev = type("RemotePCIDevice", (), {"is_remote": False})()
    self.assertFalse(dev.is_remote_pci())

if __name__ == "__main__":
  unittest.main()
