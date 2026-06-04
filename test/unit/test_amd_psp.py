import os, unittest
from unittest import mock

from tinygrad.helpers import getenv
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.am.ip import AM_PSP, AM_ReorderedMsg1View

class FakeReg:
  def __init__(self, name="regMP0_SMN_C2PMSG_35", reads=None):
    self.name, self.addr, self.reads, self.writes = name, (0x16063,), reads if reads is not None else [], []
  def read(self):
    self.reads.append(self.name)
    return 0
  def write(self, val): self.writes.append(val)

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

class FakeSyncMsg1View(FakeMsg1View):
  def __init__(self, data:bytes, off=0, syncs=None):
    super().__init__(data)
    self.off, self.syncs = off, [] if syncs is None else syncs
  def view(self, offset:int=0, size:int|None=None, fmt=None):
    view = FakeSyncMsg1View(bytes(self.data[offset:offset + (size or len(self.data) - offset)]), self.off + offset, self.syncs)
    view.nbytes = size or len(self.data) - offset
    return view
  def sync(self, invalidate=False): self.syncs.append((self.off, self.nbytes, invalidate))

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

  def test_msg1_sysmem_sync_uses_written_payload_range(self):
    gmc = FakeGMC()
    adev = FakeAdev()
    adev.gmc = gmc
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.msg1_kind = "sysmem-gart"
    psp.msg1_view = FakeSyncMsg1View(b"\x00" * 0x1000)

    with mock.patch.dict(os.environ, {"AM_PSP_MSG1_SYSMEM_SYNC": "1", "AM_PSP_MSG1_SYSMEM_SYNC_INVALIDATE": "1"}):
      getenv.cache_clear()
      try:
        psp._prep_msg1(memoryview(b"abc"))
      finally:
        getenv.cache_clear()

    self.assertEqual(psp.msg1_view.syncs, [(0, 16, True)])
    self.assertEqual(gmc.flushes, 1)

  def test_msg1_primary_sync_uses_full_primary_buffer(self):
    gmc = FakeGMC()
    adev = FakeAdev()
    adev.gmc = gmc
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.msg1_kind = "sysmem-gart"
    psp.msg1_view = FakeSyncMsg1View(b"\xff" * 0x1000)

    with mock.patch.dict(os.environ, {"AM_PSP_ZERO_MSG1": "1", "AM_PSP_MSG1_PRIMARY_SYNC": "1", "AM_PSP_TRACE": "1"}):
      getenv.cache_clear()
      try:
        psp._prep_msg1(memoryview(b"abc"))
      finally:
        getenv.cache_clear()

    self.assertEqual(psp.msg1_view.syncs, [(0, 0x1000, False)])
    self.assertEqual(gmc.flushes, 2)
    self.assertEqual(psp.msg1_view.data[:16], bytearray(b"abc\x00" + b"\x00" * 12))
    self.assertEqual(psp.msg1_view.data[16:], bytearray(b"\x00" * (0x1000 - 16)))

  def test_pre_kdb_gart_audit_stop_happens_before_mailbox_writes(self):
    gmc = FakeGMC()
    gmc.vmhubs = 0
    adev = FakeAdev()
    adev.gmc = gmc
    adev.fw = type("FakeFW", (), {"sos_fw": {am.PSP_FW_TYPE_PSP_KDB: b"abcdef"}})()
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.reg_pref = "regMP0_SMN_C2PMSG"
    psp.msg1_kind = "sysmem-gart"
    psp.msg1_addr = 0x7fff00700000
    psp.msg1_view = FakeSyncMsg1View(b"\x00" * 0x1000)
    psp.msg1_paddrs = [0x100000 + i * 0x1000 for i in range(256)]
    psp.msg1_gart_info = ([0] * 0x1000, 0x700, 0x100)

    with mock.patch.dict(os.environ, {"AM_PSP_PRE_KDB_GART_AUDIT": "1", "AM_PSP_PRE_KDB_GART_AUDIT_STOP": "1"}):
      getenv.cache_clear()
      try:
        with mock.patch.object(psp, "_wait_for_bootloader", return_value=0):
          with self.assertRaisesRegex(RuntimeError, "stopped before KDB mailbox writes"):
            psp._bootloader_load_component(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_KEY_DATABASE)
      finally:
        getenv.cache_clear()

    self.assertEqual(gmc.flushes, 2)

  def test_kdb_order_barrier_checks_msg1_and_traces_regs(self):
    gmc = FakeGMC()
    adev = FakeAdev()
    adev.gmc = gmc
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.msg1_kind = "sysmem-gart"
    psp.msg1_view = FakeSyncMsg1View(b"abc" + b"\x00" * 13)
    psp.msg1_gart_info = ([0x3000000000077, 0x3000000010077], 0, 2)

    with mock.patch.dict(os.environ, {"AM_PSP_KDB_ORDER_BARRIER": "1", "AM_PSP_TRACE": "1"}):
      getenv.cache_clear()
      try:
        psp._kdb_order_barrier("unit", b"abc" + b"\x00" * 13, FakeReg(), FakeReg())
      finally:
        getenv.cache_clear()

    self.assertEqual(psp.msg1_view.syncs, [(0, 16, False)])
    self.assertEqual(gmc.flushes, 1)

  def test_kdb_payload_audit_traces_hash_and_bounded_windows(self):
    psp = object.__new__(AM_PSP)
    psp.adev = FakeAdev()
    traces = []
    psp._trace = traces.append
    payload = bytes(range(32))
    padded = payload + b"\x00" * 16

    with mock.patch.dict(os.environ, {"AM_PSP_KDB_PAYLOAD_AUDIT": "1", "AM_PSP_KDB_PAYLOAD_AUDIT_BYTES": "8"}):
      getenv.cache_clear()
      try:
        psp._kdb_payload_audit(payload, padded)
      finally:
        getenv.cache_clear()

    self.assertEqual(len(traces), 3)
    self.assertIn("payload_size=0x20 padded_size=0x30", traces[0])
    self.assertIn("payload_sha256=", traces[0])
    self.assertIn("first8=0001020304050607", traces[1])
    self.assertIn("last8=0000000000000000", traces[1])
    self.assertIn("dwords_le", traces[2])

  def test_bootloader_payload_audit_traces_component_hash_and_bounded_windows(self):
    psp = object.__new__(AM_PSP)
    psp.adev = FakeAdev()
    traces = []
    psp._trace = traces.append
    payload = bytes(range(32))
    padded = payload + b"\x00" * 16

    with mock.patch.dict(os.environ, {"AM_PSP_BL_PAYLOAD_AUDIT": "1", "AM_PSP_BL_PAYLOAD_AUDIT_BYTES": "8"}):
      getenv.cache_clear()
      try:
        psp._bootloader_payload_audit(am.PSP_FW_TYPE_PSP_SYS_DRV, 0x10000, payload, padded)
      finally:
        getenv.cache_clear()

    self.assertEqual(len(traces), 3)
    self.assertIn("fw=PSP_FW_TYPE_PSP_SYS_DRV compid=0x10000", traces[0])
    self.assertIn("payload_size=0x20 padded_size=0x30", traces[0])
    self.assertIn("payload_sha256=", traces[0])
    self.assertIn("first8=0001020304050607", traces[1])
    self.assertIn("last8=0000000000000000", traces[1])
    self.assertIn("dwords_le", traces[2])

  def test_kdb_skip_prefix_applies_to_key_database_and_tos_spl_loads(self):
    adev = FakeAdev()
    adev.fw = type("FakeFW", (), {"sos_fw": {am.PSP_FW_TYPE_PSP_KDB: b"abcdef"}})()
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.reg_pref = "regMP0_SMN_C2PMSG"
    prepped = []

    def capture_prep(data):
      prepped.append(bytes(data))
      raise RuntimeError("stop after prep")

    with mock.patch.dict(os.environ, {"AM_PSP_KDB_SKIP_PREFIX": "2"}):
      getenv.cache_clear()
      try:
        with mock.patch.object(psp, "_wait_for_bootloader", return_value=0), mock.patch.object(psp, "_prep_msg1", side_effect=capture_prep):
          with self.assertRaisesRegex(RuntimeError, "stop after prep"):
            psp._bootloader_load_component(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_KEY_DATABASE)
          with self.assertRaisesRegex(RuntimeError, "stop after prep"):
            psp._bootloader_load_component(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_TOS_SPL_TABLE)
      finally:
        getenv.cache_clear()

    self.assertEqual(prepped, [b"cdef", b"cdef"])

  def test_kdb_slice_overrides_skip_prefix_for_kdb_loads(self):
    adev = FakeAdev()
    adev.fw = type("FakeFW", (), {"sos_fw": {am.PSP_FW_TYPE_PSP_KDB: b"abcdefghij"}})()
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.reg_pref = "regMP0_SMN_C2PMSG"
    prepped = []

    def capture_prep(data):
      prepped.append(bytes(data))
      raise RuntimeError("stop after prep")

    with mock.patch.dict(os.environ, {"AM_PSP_KDB_SKIP_PREFIX": "6", "AM_PSP_KDB_SLICE_OFFSET": "2", "AM_PSP_KDB_SLICE_SIZE": "4"}):
      getenv.cache_clear()
      try:
        with mock.patch.object(psp, "_wait_for_bootloader", return_value=0), mock.patch.object(psp, "_prep_msg1", side_effect=capture_prep):
          with self.assertRaisesRegex(RuntimeError, "stop after prep"):
            psp._bootloader_load_component(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_KEY_DATABASE)
          with self.assertRaisesRegex(RuntimeError, "stop after prep"):
            psp._bootloader_load_component(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_TOS_SPL_TABLE)
      finally:
        getenv.cache_clear()

    self.assertEqual(prepped, [b"cdef", b"cdef"])

  def test_reordered_msg1_view_maps_logical_pages_to_sorted_physical_pages(self):
    raw = FakeSyncMsg1View(b"\x00" * 0x3000)
    view = AM_ReorderedMsg1View(raw, [2, 0, 1])
    view[:0x1004] = b"a" * 0x1000 + b"bcde"

    self.assertEqual(raw.data[0x2000:0x2004], bytearray(b"aaaa"))
    self.assertEqual(raw.data[0x0000:0x0004], bytearray(b"bcde"))
    self.assertEqual(view[:0x1004], b"a" * 0x1000 + b"bcde")

    sub = view.view(0x1000, 4)
    self.assertEqual(sub[:], b"bcde")
    sub.sync(invalidate=True)
    self.assertEqual(raw.syncs, [(0, 0x3000, True)])

  def test_kdb_fail_capture_sampler_skips_missing_focus_regs(self):
    reads = []
    regs = {f"regMP0_SMN_C2PMSG_{idx}": FakeReg(f"regMP0_SMN_C2PMSG_{idx}", reads) for idx in [35, 36, 64, 67, 81, 90, 92]}
    adev = FakeAdev()
    adev.reg = lambda name: regs[name]
    for name, reg in regs.items(): setattr(adev, name, reg)
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.reg_pref = "regMP0_SMN_C2PMSG"

    with mock.patch.dict(os.environ, {"AM_PSP_TRACE": "1", "AM_PSP_KDB_FAIL_CAPTURE_READS": "1"}):
      psp._kdb_fail_capture_sample(regs["regMP0_SMN_C2PMSG_35"], regs["regMP0_SMN_C2PMSG_36"])

    self.assertNotIn("regMP0_SMN_C2PMSG_115", reads)
    self.assertIn("regMP0_SMN_C2PMSG_92", reads)

  def test_mailbox_visibility_samples_focus_regs_and_flushes_when_requested(self):
    reads = []
    regs = {f"regMP0_SMN_C2PMSG_{idx}": FakeReg(f"regMP0_SMN_C2PMSG_{idx}", reads) for idx in [35, 36, 64, 67, 81, 90, 92]}
    adev = FakeAdev()
    adev.reg = lambda name: regs[name]
    adev.gmc = FakeGMC()
    for name, reg in regs.items(): setattr(adev, name, reg)
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.reg_pref = "regMP0_SMN_C2PMSG"

    with mock.patch.dict(os.environ, {"AM_PSP_MAILBOX_VIS": "1", "AM_PSP_MAILBOX_VIS_READS": "2", "AM_PSP_MAILBOX_VIS_HDP_FLUSH": "1"}):
      getenv.cache_clear()
      try:
        psp._mailbox_visibility_sample("post-compid", regs["regMP0_SMN_C2PMSG_35"], regs["regMP0_SMN_C2PMSG_36"])
      finally:
        getenv.cache_clear()

    self.assertEqual(adev.gmc.flushes, 2)
    self.assertEqual(reads.count("regMP0_SMN_C2PMSG_35"), 2)
    self.assertEqual(reads.count("regMP0_SMN_C2PMSG_92"), 2)
    self.assertNotIn("regMP0_SMN_C2PMSG_115", reads)

  def test_mailbox_visibility_read_cap(self):
    adev = FakeAdev()
    adev.gmc = FakeGMC()
    psp = object.__new__(AM_PSP)
    psp.adev = adev
    psp.reg_pref = "regMP0_SMN_C2PMSG"

    with mock.patch.dict(os.environ, {"AM_PSP_MAILBOX_VIS": "1", "AM_PSP_MAILBOX_VIS_READS": "4097"}):
      getenv.cache_clear()
      try:
        with self.assertRaisesRegex(ValueError, "too large"):
          psp._mailbox_visibility_sample("post-compid", FakeReg(), FakeReg())
      finally:
        getenv.cache_clear()

if __name__ == "__main__":
  unittest.main()
