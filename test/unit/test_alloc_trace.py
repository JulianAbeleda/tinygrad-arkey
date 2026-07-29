"""Pins ALLOC_TRACE, the low-overhead fault-to-allocation/dispatch attribution ring
(tinygrad/device.py, next to KERNARGS_AUDIT / DISPATCH_TRACE).

Context: ~140 GPU fault addresses observed in the 0x00007xxx_xxxxx000 range -- exactly where
KFDIface.alloc's anon_mmap(0, ...) places tinygrad's own buffers (tinygrad/runtime/ops_amd.py:795-823:
tinygrad never chooses the GPU VA, the host mmap does, and the KFD ioctl echoes it back). ALLOC_TRACE
records every allocation's VA range and lifetime, and every dispatch's kernel identity/grid/kernarg
pointers, into a fixed-size preallocated ring -- no I/O and no synchronize() in the hot path -- so a fault
address can be attributed after the fact via the dev-tier extra/debug/gpu_fault_analysis/analyze_faults.py tool.

Unlike DISPATCH_TRACE (which deliberately serializes every dispatch and is far too expensive to leave on),
ALLOC_TRACE never blocks the device; it only touches disk at dump time (process exit or an explicit call).

Import note: this imports `tinygrad.device` normally. An earlier draft installed a stub `tinygrad`
module into sys.modules to import device.py without executing tinygrad/__init__.py. That stub carried
only __path__ -- no Tensor, no dtypes -- so every test module collected AFTER this one in the same
pytest session died with "cannot import name 'Tensor' from 'tinygrad' (unknown location)". It silently
cut suite collection from 1454 tests to 504 with 94 errors. A test must never mutate sys.modules for
the package under test.
"""
import pathlib, unittest, tempfile, os, json, time

import tinygrad.device as D


class _FakeBuf:
  """Stands in for HCQBuffer: ALLOC_TRACE only ever reads .va_addr and .size off it."""
  def __init__(self, va_addr, size): self.va_addr, self.size = va_addr, size


class _FakeSignal:
  def __init__(self, value): self.value = value


def _reset_ring():
  D._at_alloc_ring = None
  D._at_dispatch_ring = None
  D._at_alloc_count[0] = 0
  D._at_dispatch_count[0] = 0
  D._at_seq[0] = 0
  D._at_device_ids.clear(); D._at_device_names.clear()
  D._at_kernel_ids.clear(); D._at_kernel_names.clear()


class TestAllocTraceRecording(unittest.TestCase):
  def setUp(self):
    self._prev = D.ALLOC_TRACE.value
    D.ALLOC_TRACE.value = 1
    _reset_ring()

  def tearDown(self):
    D.ALLOC_TRACE.value = self._prev
    _reset_ring()

  def test_alloc_records_va_range_and_both_sizes(self):
    aid = D.alloc_trace_record_alloc("AMD", va_start=0x7f0000000000, mapped_size=0x2000, req_size=0x1800)
    self.assertEqual(aid, 0)
    rec = D._at_alloc_ring[0]
    self.assertEqual(rec.va_start, 0x7f0000000000)
    self.assertEqual(rec.va_end, 0x7f0000002000)  # va_start + mapped_size, NOT + req_size
    self.assertEqual(rec.req_size, 0x1800)
    self.assertEqual(rec.mapped_size, 0x2000)
    self.assertEqual(rec.free_seq, -1)  # not freed yet

  def test_alloc_ids_are_monotonic_across_calls(self):
    a0 = D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000)
    a1 = D.alloc_trace_record_alloc("AMD", 0x2000, 0x1000, 0x1000)
    a2 = D.alloc_trace_record_alloc("AMD", 0x3000, 0x1000, 0x1000)
    self.assertEqual((a0, a1, a2), (0, 1, 2))

  def test_free_sets_free_seq_and_ts_on_the_right_record(self):
    a0 = D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000)
    a1 = D.alloc_trace_record_alloc("AMD", 0x2000, 0x1000, 0x1000)
    D.alloc_trace_record_free(a0)
    self.assertGreaterEqual(D._at_alloc_ring[0].free_seq, 0)
    self.assertEqual(D._at_alloc_ring[1].free_seq, -1)  # a1 untouched
    self.assertGreater(D._at_alloc_ring[0].free_ts_ns, 0)

  def test_free_of_negative_id_is_a_silent_no_op(self):
    D.alloc_trace_record_free(-1)  # must not raise, must not touch the sequence clock in a way that breaks anything
    self.assertEqual(D._at_seq[0], 0)

  def test_disabled_recording_returns_sentinel_and_touches_nothing(self):
    D.ALLOC_TRACE.value = 0
    aid = D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000)
    self.assertEqual(aid, -1)
    self.assertIsNone(D._at_alloc_ring)  # ring never even allocated -- true zero cost when off

  def test_dispatch_records_grid_and_arg_va_size(self):
    bufs = [_FakeBuf(0x7f0000000000, 4096), _FakeBuf(0x7f0000001000, 8192)]
    D.alloc_trace_record_dispatch("AMD", "r_64_4n2", (64, 2, 1), (4, 1, 1), bufs, signal_target=7)
    rec = D._at_dispatch_ring[0]
    self.assertEqual((rec.gx, rec.gy, rec.gz), (64, 2, 1))
    self.assertEqual((rec.lx, rec.ly, rec.lz), (4, 1, 1))
    self.assertEqual(rec.nargs, 2)
    self.assertEqual(rec.arg_va[0], 0x7f0000000000)
    self.assertEqual(rec.arg_size[1], 8192)
    self.assertEqual(rec.signal_target, 7)
    self.assertEqual(D._at_kernel_names[rec.kernel_id], "r_64_4n2")

  def test_dispatch_args_beyond_max_are_truncated_not_crashed(self):
    bufs = [_FakeBuf(i, i) for i in range(1, D.ALLOC_TRACE_MAX_ARGS + 5)]
    D.alloc_trace_record_dispatch("AMD", "k", (1,1,1), (1,1,1), bufs, signal_target=1)
    self.assertEqual(D._at_dispatch_ring[0].nargs, D.ALLOC_TRACE_MAX_ARGS)

  def test_device_and_kernel_name_tables_dedupe(self):
    for _ in range(5): D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000)
    self.assertEqual(D._at_device_names, ["AMD"])  # registered once, not 5 times
    for _ in range(3): D.alloc_trace_record_dispatch("AMD", "same_kernel", (1,1,1), (1,1,1), [], 1)
    self.assertEqual(D._at_kernel_names, ["same_kernel"])

  def test_ring_wraps_without_crashing_and_free_on_stale_id_is_a_no_op(self):
    cap = 8
    prev_cap = D.ALLOC_TRACE_ALLOCS
    D.ALLOC_TRACE_ALLOCS = cap  # shrink for this test only, restored in tearDown-adjacent finally below
    try:
      _reset_ring()
      ids = [D.alloc_trace_record_alloc("AMD", i, 0x1000, 0x1000) for i in range(cap * 3)]
      # the ring only holds the most recent `cap` allocations; free() on a long-evicted id must not raise
      # and must not corrupt whatever now occupies that slot.
      D.alloc_trace_record_free(ids[0])
      last_slot = D._at_alloc_ring[(cap * 3 - 1) % cap]
      self.assertEqual(last_slot.alloc_id, cap * 3 - 1)
      self.assertEqual(last_slot.free_seq, -1)  # untouched by the stale free
    finally:
      D.ALLOC_TRACE_ALLOCS = prev_cap
      _reset_ring()


class TestAllocTraceDump(unittest.TestCase):
  def setUp(self):
    self._prev = D.ALLOC_TRACE.value
    D.ALLOC_TRACE.value = 1
    _reset_ring()
    self._tmpdir = tempfile.mkdtemp()

  def tearDown(self):
    D.ALLOC_TRACE.value = self._prev
    _reset_ring()

  def test_dump_writes_json_with_allocs_and_dispatches(self):
    a0 = D.alloc_trace_record_alloc("AMD", 0x7f0000000000, 0x2000, 0x1800)
    D.alloc_trace_record_free(a0)
    D.alloc_trace_record_dispatch("AMD", "r_k", (8,1,1), (2,1,1), [_FakeBuf(0x7f0000000000, 0x1800)], signal_target=3)
    path = os.path.join(self._tmpdir, "dump.json")
    ret = D.alloc_trace_dump(path)
    self.assertEqual(ret, path)
    data = json.load(open(path))
    self.assertEqual(data["format"], "tinygrad-alloc-trace-v1")
    self.assertEqual(len(data["allocs"]), 1)
    self.assertEqual(len(data["dispatches"]), 1)
    self.assertEqual(data["allocs"][0]["va_start"], 0x7f0000000000)
    self.assertIsNotNone(data["allocs"][0]["free_seq"])
    self.assertEqual(data["dispatches"][0]["args"][0]["va"], 0x7f0000000000)

  def test_dump_when_never_enabled_returns_none(self):
    D.ALLOC_TRACE.value = 0
    _reset_ring()
    self.assertIsNone(D.alloc_trace_dump(os.path.join(self._tmpdir, "should_not_exist.json")))

  def test_dump_is_idempotent_and_repeatable(self):
    D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000)
    p1 = os.path.join(self._tmpdir, "a.json")
    p2 = os.path.join(self._tmpdir, "b.json")
    D.alloc_trace_dump(p1)
    D.alloc_trace_record_alloc("AMD", 0x2000, 0x1000, 0x1000)
    D.alloc_trace_dump(p2)
    self.assertEqual(len(json.load(open(p1))["allocs"]), 1)
    self.assertEqual(len(json.load(open(p2))["allocs"]), 2)


class TestAllocTraceDefaultAndWiring(unittest.TestCase):
  def test_default_is_off(self):
    self.assertEqual(D.ALLOC_TRACE.value, 0, "ALLOC_TRACE must ship default-off")

  def test_hcq_call_site_still_exists(self):
    # The dispatch half of the ring is only as good as its one hook in HCQProgram.__call__; if that hook is
    # lost the ring silently stops recording dispatches while allocations keep flowing -- an asymmetric,
    # easy-to-miss failure. Pin both the import and the call.
    hcq_path = pathlib.Path(__file__).resolve().parents[2] / "tinygrad" / "runtime" / "support" / "hcq.py"
    src = hcq_path.read_text()
    self.assertIn("alloc_trace_record_dispatch", src, "HCQProgram.__call__ lost its ALLOC_TRACE dispatch hook")
    self.assertIn("from tinygrad.device import ALLOC_TRACE", src, "hcq.py lost its ALLOC_TRACE import")

  def test_buffer_allocate_deallocate_still_wired(self):
    dev_path = pathlib.Path(__file__).resolve().parents[2] / "tinygrad" / "device.py"
    src = dev_path.read_text()
    self.assertIn("alloc_trace_record_alloc", src)
    self.assertIn("alloc_trace_record_free", src)


class TestAllocTraceOverhead(unittest.TestCase):
  """Not a strict perf gate (CI machines vary) -- a sanity bound so a regression that makes ALLOC_TRACE
  expensive (e.g. an accidental print, or a dict rebuilt every call) fails loudly instead of silently."""
  def test_recording_is_fast_and_off_switch_is_near_free(self):
    _reset_ring()
    bufs = [_FakeBuf(0x1000, 0x1000)]

    D.ALLOC_TRACE.value = 0
    t0 = time.perf_counter()
    for i in range(20000):
      D.alloc_trace_record_alloc("AMD", i, 0x1000, 0x1000)
      D.alloc_trace_record_dispatch("AMD", "k", (1,1,1), (1,1,1), bufs, 1)
    off_ns_per_call = (time.perf_counter() - t0) / (20000 * 2) * 1e9

    D.ALLOC_TRACE.value = 1
    _reset_ring()
    t0 = time.perf_counter()
    for i in range(20000):
      aid = D.alloc_trace_record_alloc("AMD", i, 0x1000, 0x1000)
      D.alloc_trace_record_free(aid)
      D.alloc_trace_record_dispatch("AMD", "k", (1,1,1), (1,1,1), bufs, 1)
    on_ns_per_call = (time.perf_counter() - t0) / (20000 * 3) * 1e9
    D.ALLOC_TRACE.value = 0
    _reset_ring()

    print(f"\n=== ALLOC_TRACE overhead: OFF {off_ns_per_call:.1f} ns/call, ON {on_ns_per_call:.1f} ns/call ===")
    self.assertLess(off_ns_per_call, 200, "the OFF branch should be a single ContextVar bool check")
    self.assertLess(on_ns_per_call, 5000, "recording should stay well under a microsecond per call")


if __name__ == "__main__":
  unittest.main()
