import unittest

from extra.remote.amd_repro import classify_psp_clean_gate

BASELINE = {
  "C2PMSG33_VMBX": 0x80000000,
  "C2PMSG35_BL": 0x80000000,
  "C2PMSG36_ADDR": 0,
  "C2PMSG81_SOS": 0,
}

class TestAMDCleanGate(unittest.TestCase):
  def test_clean_pre_kdb_baseline(self):
    status, reasons = classify_psp_clean_gate(dict(BASELINE))
    self.assertEqual(status, "CLEAN")
    self.assertEqual(reasons, ["PSP mailbox is at pre-KDB ready baseline"])

  def test_dirty_when_bootloader_mailbox_stuck_zero(self):
    vals = dict(BASELINE, C2PMSG35_BL=0)
    status, reasons = classify_psp_clean_gate(vals)
    self.assertEqual(status, "DIRTY")
    self.assertIn("C2PMSG35_BL=0", reasons[0])

  def test_dirty_when_sos_already_alive(self):
    vals = dict(BASELINE, C2PMSG81_SOS=1)
    status, reasons = classify_psp_clean_gate(vals)
    self.assertEqual(status, "DIRTY")
    self.assertIn("C2PMSG81_SOS", reasons[0])

  def test_dirty_when_mailbox_reads_all_ones(self):
    for reg in ("C2PMSG33_VMBX", "C2PMSG35_BL"):
      vals = dict(BASELINE, **{reg: 0xffffffff})
      status, reasons = classify_psp_clean_gate(vals)
      self.assertEqual(status, "DIRTY")
      self.assertEqual(reasons, ["PSP mailbox returned all-ones MMIO"])

  def test_unknown_when_ready_registers_are_unexpected(self):
    for reg, val in (("C2PMSG33_VMBX", 0x1), ("C2PMSG35_BL", 0x1)):
      vals = dict(BASELINE, **{reg: val})
      status, reasons = classify_psp_clean_gate(vals)
      self.assertEqual(status, "UNKNOWN")
      self.assertIn(reg, reasons[0])

  def test_unknown_when_ready_with_nonzero_msg_addr(self):
    vals = dict(BASELINE, C2PMSG36_ADDR=0x5fff)
    status, reasons = classify_psp_clean_gate(vals)
    self.assertEqual(status, "UNKNOWN")
    self.assertIn("C2PMSG36_ADDR=0x00005fff", reasons[0])

if __name__ == "__main__":
  unittest.main()
