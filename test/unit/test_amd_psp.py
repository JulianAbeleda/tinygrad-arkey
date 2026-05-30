import unittest
from unittest import mock

from tinygrad.runtime.support.am.ip import AM_PSP

class FakeReg:
  addr = (0x16063,)
  def read(self): return 0

class FakeAdev:
  devfmt = "fake"
  def reg(self, name): return FakeReg()

class TestAMDPSP(unittest.TestCase):
  def test_bootloader_wait_timeout_uses_last_read_with_trace_disabled(self):
    psp = object.__new__(AM_PSP)
    psp.adev = FakeAdev()
    psp.reg_pref = "regMP0_SMN_C2PMSG"
    times = iter([0.0, 0.0, 0.001, 11.0, 11.0])

    with mock.patch("tinygrad.runtime.support.am.ip.time.perf_counter", side_effect=lambda: next(times, 11.0)):
      with self.assertRaisesRegex(TimeoutError, "condition not met: 0 != 2147483648"):
        psp._wait_for_bootloader()

if __name__ == "__main__":
  unittest.main()
