import os, unittest
from unittest import mock

from tinygrad.runtime.support.am.ip import AM_PSP

class FakeReg:
  def __init__(self, name="regMP0_SMN_C2PMSG_35", reads=None):
    self.name, self.addr, self.reads = name, (0x16063,), reads if reads is not None else []
  def read(self):
    self.reads.append(self.name)
    return 0

class FakeAdev:
  devfmt = "fake"
  def reg(self, name): return FakeReg()

class FakeGMC:
  def __init__(self): self.flushes = 0
  def flush_hdp(self): self.flushes += 1

class FakeMsg1View:
  def __init__(self, data:bytes):
    self.data = bytearray(data)
    self.nbytes = len(self.data)
  def __getitem__(self, idx): return bytes(self.data[idx])
  def __setitem__(self, idx, val): self.data[idx] = val

class TestAMDPSP(unittest.TestCase):
  def test_bootloader_wait_timeout_uses_last_read_with_trace_disabled(self):
    psp = object.__new__(AM_PSP)
    psp.adev = FakeAdev()
    psp.reg_pref = "regMP0_SMN_C2PMSG"
    times = iter([0.0, 0.0, 0.001, 11.0, 11.0])

    with mock.patch("tinygrad.runtime.support.am.ip.time.perf_counter", side_effect=lambda: next(times, 11.0)):
      with self.assertRaisesRegex(TimeoutError, "condition not met: 0 != 2147483648"):
        psp._wait_for_bootloader()

  def test_linux_pre_bootloader_status_reads_c2pmsg81_before_wait(self):
    reads = []
    adev = FakeAdev()
    adev.fw = type("FakeFW", (), {"sos_fw": {123: b""}})()
    adev.reg = lambda name: FakeReg(name, reads)
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.reg_pref = "regMP0_SMN_C2PMSG"
    order = []

    def stop_at_wait():
      order.append("wait")
      raise RuntimeError("stop")

    with mock.patch.dict(os.environ, {"AM_PSP_LINUX_PRE_BL_STATUS": "1"}):
      with mock.patch.object(psp, "_wait_for_bootloader", side_effect=stop_at_wait):
        with self.assertRaisesRegex(RuntimeError, "stop"):
          psp._bootloader_load_component(123, 456)

    self.assertEqual(reads, ["regMP0_SMN_C2PMSG_81"])
    self.assertEqual(order, ["wait"])

  def test_msg1_visibility_probe_restores_original_buffer(self):
    gmc = FakeGMC()
    adev = FakeAdev()
    adev.gmc = gmc
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.msg1_kind = "sysmem-gart"
    psp.msg1_addr = 0x7fff00700000
    psp.msg1_view = FakeMsg1View(bytes(range(256)) * 16)
    psp.msg1_paddrs = [0x100000 + i * 0x1000 for i in range(256)]

    with mock.patch.dict(os.environ, {"AM_PSP_TRACE": "1"}):
      psp._msg1_visibility_probe()

    self.assertEqual(psp.msg1_view.data, bytearray(bytes(range(256)) * 16))
    self.assertEqual(gmc.flushes, 2)

if __name__ == "__main__":
  unittest.main()
