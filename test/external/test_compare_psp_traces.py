"""Pure-CPU regression tests for the PSP trace line parsers in
extra/amdpci/compare_psp_traces.py. Exercises parse_linux_lines / parse_tinygrad_lines
with synthetic log snippets -- no capture archives, no firmware fetch, no hardware.
"""
import unittest

from extra.amdpci.compare_psp_traces import parse_linux_lines, parse_tinygrad_lines

class TestParseLinuxLines(unittest.TestCase):
  def test_bl_load_enter(self):
    out = parse_linux_lines(["100 bl_load enter dev cmd=0x1 fw_pri_mc=0x2000 c2p36=0x16064 size=0x400"])
    self.assertEqual(out["bl"], [{"t": 100, "cmd": 0x1, "fw_pri_mc": 0x2000, "c2p36": 0x16064, "size": 0x400}])

  def test_bl_load_enter_deduplicates(self):
    line = "100 bl_load enter dev cmd=0x1 fw_pri_mc=0x2000 c2p36=0x16064 size=0x400"
    self.assertEqual(len(parse_linux_lines([line, line])["bl"]), 1)

  def test_wait_bl_ret(self):
    out = parse_linux_lines(["200 wait_bl enter", "250 wait_bl ret=0 duration_ns=50 reads=5"])
    self.assertEqual(out["waits"], [{"start": 200, "end": 250, "ret": 0, "duration_ns": 50, "reads": 5}])

  def test_wait_bl_ret_duration_inferred_when_absent(self):
    out = parse_linux_lines(["200 wait_bl enter", "275 wait_bl ret=-110"])
    self.assertEqual(out["waits"], [{"start": 200, "end": 275, "ret": -110, "duration_ns": 75, "reads": None}])

  def test_c2pmsg_write_and_read_extraction(self):
    # reg 0x16063 == C2PMSG_BASE(0x16040) + idx 35
    out = parse_linux_lines(["300 wreg dev reg=0x16063 val=0xabc", "310 rreg dev reg=0x16063 val=0xdef"])
    self.assertEqual(out["c2pmsg_events"], [
      {"t": 300, "op": "wreg", "idx": 35, "reg": 0x16063, "val": 0xabc},
      {"t": 310, "op": "rreg", "idx": 35, "reg": 0x16063, "val": 0xdef}])
    self.assertEqual([e["op"] for e in out["reg_events"]], ["wreg", "rreg"])

  def test_non_c2pmsg_reg_has_no_c2pmsg_event(self):
    out = parse_linux_lines(["400 wreg dev reg=0x9000 val=0x1"])
    self.assertEqual(out["reg_events"], [{"t": 400, "op": "wreg", "reg": 0x9000, "val": 0x1}])
    self.assertEqual(out["c2pmsg_events"], [])

  def test_source_defaults_empty_and_threads(self):
    self.assertEqual(parse_linux_lines([])["source"], "")
    self.assertEqual(parse_linux_lines([], source="cap.tar.gz")["source"], "cap.tar.gz")

class TestParseTinygradLines(unittest.TestCase):
  def test_write_msg1(self):
    out = parse_tinygrad_lines(["PSP write msg1 kind=KDB reg36=0x1 val=0x2000 msg1_addr=0x3000"])
    self.assertEqual(out["write_msg1"], {"kind": "KDB", "val": 0x2000, "addr": 0x3000})

  def test_write_compid(self):
    out = parse_tinygrad_lines(["PSP write compid reg35=0x1 val=0x10000003"])
    self.assertEqual(out["write_compid"], 0x10000003)

  def test_wait_bl(self):
    out = parse_tinygrad_lines(["PSP wait BL reg35=0x1 val=0x80000000", "PSP wait BL reg35=0x1 val=0x0"])
    self.assertEqual(out["wait_vals"], [0x80000000, 0x0])

  def test_timeout_detection(self):
    self.assertTrue(parse_tinygrad_lines(["PSP BL not ready after timeout"])["timeout"])
    self.assertFalse(parse_tinygrad_lines(["PSP wait BL reg35=0x1 val=0x0"])["timeout"])

  def test_msg1_readback(self):
    out = parse_tinygrad_lines(["PSP msg1 readback ok bytes=16 first=00112233 last=ccddeeff"])
    self.assertEqual(out["readback"], {"bytes": 16, "first": "00112233", "last": "ccddeeff"})

  def test_reg_readback_with_and_without_instance(self):
    out = parse_tinygrad_lines(["PSP reg regFoo=0x5", "PSP reg regBar[2]=0x6"])
    self.assertEqual(out["regs"], {"regFoo": 0x5, "regBar[2]": 0x6})

  def test_snapshot_scopes_regs(self):
    out = parse_tinygrad_lines([
      "PSP parity snapshot pre begin", "PSP reg regSnap=0x7", "PSP parity snapshot pre end", "PSP reg regAfter=0x8"])
    self.assertEqual(out["regs"], {"regSnap": 0x7, "regAfter": 0x8})
    self.assertEqual(out["snapshots"], [{"label": "pre", "regs": {"regSnap": 0x7}}])

  def test_source_defaults_empty_and_threads(self):
    self.assertEqual(parse_tinygrad_lines([])["source"], "")
    self.assertEqual(parse_tinygrad_lines([], source="run.tar.gz")["source"], "run.tar.gz")

if __name__ == "__main__":
  unittest.main()
