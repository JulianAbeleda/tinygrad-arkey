import unittest

from tinygrad.runtime.support.am.amdev import AMDev

class FakePCI:
  is_remote = True

class TestAMDVramWrite(unittest.TestCase):
  def test_remote_sparse_write_is_bounded(self):
    adev = object.__new__(AMDev)
    adev.pci_dev = FakePCI()
    writes = []
    adev.wreg = lambda reg, val: writes.append((reg, val))

    adev._write_vram(0x1000, b"\x00" * 0x1000, allow_remote_sparse=True)
    self.assertEqual(len(writes), 0x1000 // 4 * 3)

    with self.assertRaisesRegex(RuntimeError, "remote AMD indirect VRAM writes are disabled"):
      adev._write_vram(0x1000, b"\x00" * 0x1004, allow_remote_sparse=True)

if __name__ == "__main__":
  unittest.main()
