"""Self-test for extra/gpu_fault_analysis/analyze_faults.py, run against SYNTHETIC data with known answers.

Required by the ALLOC_TRACE handoff: "a silent instrument is our most common failure mode" -- this proves
the analysis script's core claims (live-interval-at-fault-time, nearest-boundary distance, dispatch/pointer
attribution, kernel-log bitfield decode) are correct BEFORE the script is ever pointed at the ~140 real
fault addresses from `sudo journalctl -k -b -1`.

Does not import the `tinygrad` package (analyze_faults.py itself has no tinygrad dependency), so it is not
exposed to unrelated breakage elsewhere in the tree.
"""
import sys, pathlib, unittest, importlib.util

_MOD_PATH = pathlib.Path(__file__).resolve().parents[2] / "extra" / "gpu_fault_analysis" / "analyze_faults.py"
_spec = importlib.util.spec_from_file_location("analyze_faults", _MOD_PATH)
AF = importlib.util.module_from_spec(_spec)
sys.modules["analyze_faults"] = AF
_spec.loader.exec_module(AF)


class TestProtectionFaultStatusDecode(unittest.TestCase):
  def test_matches_the_verified_field_values_from_the_fault_scope_doc(self):
    # docs/fault-scope-for-review-20260726.md: "GCVM_L2_PROTECTION_FAULT_STATUS = 0x008012B1 decodes ... as
    # MORE_FAULTS=1, WALKER_ERROR=0, PERMISSION_FAULTS=0xb, MAPPING_ERROR=0, CID=9 (SQC inst), RW=0, VMID=8".
    # Shifts/masks below were read directly from
    # /usr/src/amdgpu-6.16.13-2341068.24.04/amd/include/asic_reg/gc/gc_11_0_0_sh_mask.h on this box, not
    # guessed -- this test is what proves that transcription didn't introduce an off-by-one.
    f = AF.decode_protection_fault_status(0x008012B1)
    self.assertEqual(f["MORE_FAULTS"], 1)
    self.assertEqual(f["WALKER_ERROR"], 0)
    self.assertEqual(f["PERMISSION_FAULTS"], 0xb)
    self.assertEqual(f["MAPPING_ERROR"], 0)
    self.assertEqual(f["CID"], 9)
    self.assertEqual(f["RW"], 0)
    self.assertEqual(f["VMID"], 8)


class TestKernelLogParsing(unittest.TestCase):
  LOG = (
    "1785072100.100000 host kernel: sq_intr: error, detail 0x00000000, type 2, sh {0,1}, priv 1\n"
    "1785072100.100010 host kernel: [gfxhub] page fault (src_id:0 ring:88 vmid:8 pasid:32774)\n"
    "1785072100.100020 host kernel:   in page starting at address 0x00007c7a20b4a000 from client 10\n"
    "1785072100.100030 host kernel: GCVM_L2_PROTECTION_FAULT_STATUS: 0x008012B1\n"
    "1785072100.100040 host kernel:   Faulty UTCL2 client ID: SQC (inst) (0x9)\n"
    "1785072100.100050 host kernel: -> Failed to evict queue 0 / Failed to quiesce KFD / GPU reset begin\n"
    # a second, unrelated event later in the same log
    "1785072200.000000 host kernel: [gfxhub] page fault (src_id:0 ring:12 vmid:3 pasid:99)\n"
    "1785072200.000010 host kernel:   in page starting at address 0x0000ffffffbfe000 from client 10\n"
  )

  def test_extracts_both_events_with_correct_fields(self):
    events = AF.parse_kernel_log(self.LOG)
    self.assertEqual(len(events), 2)
    e0, e1 = events
    self.assertEqual(e0.addr, 0x00007c7a20b4a000)
    self.assertEqual(e0.vmid, 8)
    self.assertEqual(e0.pasid, 32774)
    self.assertEqual(e0.ih_client, 10)
    self.assertEqual(e0.utcl2_client_name, "SQC (inst)")
    self.assertEqual(e0.utcl2_client_id, 0x9)
    self.assertEqual(e0.status_raw, 0x008012B1)
    self.assertEqual(e0.status_fields["CID"], 9)
    self.assertEqual(e0.ts_ns, 1785072100100010000)  # from the address line's short-unix prefix

    self.assertEqual(e1.addr, 0x0000ffffffbfe000)
    self.assertEqual(e1.vmid, 3)
    self.assertEqual(e1.pasid, 99)
    self.assertIsNone(e1.status_raw)  # this synthetic event has no STATUS/UTCL2 lines -- must not crash, must stay None

  def test_no_page_fault_lines_yields_empty(self):
    self.assertEqual(AF.parse_kernel_log("nothing to see here\njust noise\n"), [])

  def test_addr_list_parsing_tolerates_blank_lines_comments_and_bare_hex(self):
    text = "0x1000\n  # a comment\n\n2000\n0X3000\n"
    self.assertEqual(AF.parse_addr_list(text), [0x1000, 0x2000, 0x3000])


class TestAttribution(unittest.TestCase):
  """Synthetic ALLOC_TRACE dump with a hand-worked ground truth:

    alloc A: [0x...0000, 0x...1000)  req=0x800  alloc_ts=1000  freed at free_ts=5000
    alloc B: [0x...1000, 0x...2000)  req=0x1000 alloc_ts=1000  never freed
    dispatch D1 'r_test', arg0 = (va=0x...1000, size=0x800), submitted at ts=4000

  and three fault addresses with known answers:
    F1 = 0x...1100  -- inside B (live), and inside D1's arg0            -> exact hit, live
    F2 = 0x...0500  -- inside A, but A was freed (5000) before F2's ts (6000) -> NOT live
    F3 = 0x...2050  -- 0x50 past B's end, no containing alloc            -> nearest_below = B, distance 0x50
  """
  BASE = 0x7f0000000000

  def setUp(self):
    b = self.BASE
    self.allocA = AF.AllocRec(alloc_id=0, device="AMD", va_start=b, va_end=b+0x1000, req_size=0x800, mapped_size=0x1000,
                               alloc_seq=0, free_seq=1, alloc_ts_ns=1000, free_ts_ns=5000)
    self.allocB = AF.AllocRec(alloc_id=1, device="AMD", va_start=b+0x1000, va_end=b+0x2000, req_size=0x1000, mapped_size=0x1000,
                               alloc_seq=2, free_seq=None, alloc_ts_ns=1000, free_ts_ns=None)
    self.allocs = [self.allocA, self.allocB]
    self.dispatch = AF.DispatchRec(dispatch_id=0, device="AMD", kernel="r_test", global_size=[1,1,1], local_size=[1,1,1],
                                    submit_seq=0, submit_ts_ns=4000, signal_target=1, completed=True,
                                    args=[{"va": b+0x1000, "size": 0x800}])
    self.dispatches = [self.dispatch]

  def test_f1_exact_hit_inside_live_allocation_and_dispatch_arg(self):
    addr = self.BASE + 0x1100
    r = AF.attribute(addr, ts_ns=6000, allocs=self.allocs, dispatches=self.dispatches)
    self.assertEqual(r["containing"], [self.allocB])
    self.assertEqual(r["live_containing"], [self.allocB])
    self.assertEqual(len(r["arg_hits"]), 1)
    d, i, dist = r["arg_hits"][0]
    self.assertIs(d, self.dispatch); self.assertEqual(i, 0); self.assertEqual(dist, 0)
    self.assertEqual(r["arg_near"], [])  # an exact hit shouldn't also show up as "near"

  def test_f2_inside_an_allocation_freed_before_the_fault(self):
    addr = self.BASE + 0x500
    r = AF.attribute(addr, ts_ns=6000, allocs=self.allocs, dispatches=self.dispatches)
    self.assertEqual(r["containing"], [self.allocA])       # structurally still "contains" the VA
    self.assertEqual(r["live_containing"], [])              # but NOT live at the fault's timestamp
    self.assertTrue(self.allocA.free_ts_ns is not None and self.allocA.free_ts_ns <= 6000)

  def test_f3_out_of_bounds_past_the_end_of_b(self):
    addr = self.BASE + 0x2050
    r = AF.attribute(addr, ts_ns=6000, allocs=self.allocs, dispatches=self.dispatches)
    self.assertEqual(r["containing"], [])
    self.assertIs(r["nearest_below"], self.allocB)
    self.assertIsNone(r["nearest_above"])
    self.assertEqual(addr - self.allocB.va_end, 0x50)
    # the dispatch arg (size 0x800, ending at BASE+0x1800) is further from F3 than B's own boundary is --
    # both are legitimate "nearest" signals; just confirm the arg shows up as a near (not exact) candidate.
    self.assertTrue(len(r["arg_near"]) >= 1)
    self.assertEqual(r["arg_near"][0][2], (self.BASE + 0x2050) - (self.BASE + 0x1000 + 0x800))

  def test_dispatch_submitted_after_the_fault_is_excluded(self):
    late = AF.DispatchRec(dispatch_id=1, device="AMD", kernel="r_late", global_size=[1,1,1], local_size=[1,1,1],
                           submit_seq=1, submit_ts_ns=999_999, signal_target=2, completed=True,
                           args=[{"va": self.BASE+0x1000, "size": 0x800}])
    r = AF.attribute(self.BASE + 0x1100, ts_ns=6000, allocs=self.allocs, dispatches=[self.dispatch, late])
    ids = {d.dispatch_id for d, _, _ in r["arg_hits"]}
    self.assertNotIn(1, ids, "a dispatch submitted after the fault's timestamp cannot be responsible for it")

  def test_no_timestamp_falls_back_to_dump_time_snapshot(self):
    # ts_ns=None models a fault address given with no kernel-log context (bare address list): liveness
    # becomes "unknown at fault time", and the caller must fall back to the dump-time snapshot instead of
    # silently claiming certainty it doesn't have.
    r = AF.attribute(self.BASE + 0x1100, ts_ns=None, allocs=self.allocs, dispatches=self.dispatches)
    self.assertIsNone(r["live_containing"])
    self.assertEqual(r["containing"], [self.allocB])


class TestEndToEndSyntheticDump(unittest.TestCase):
  """Round-trips a real alloc_trace_dump()-shaped JSON file (built by hand, not via tinygrad.device, to
  keep this test independent of tinygrad's import chain) through load_dump() + attribute()."""
  def test_load_dump_and_attribute_agree_with_hand_computed_answers(self):
    import json, tempfile, os
    base = 0x7c7a20000000
    dump = {
      "format": "tinygrad-alloc-trace-v1", "dumped_at_unix": 0.0,
      "alloc_ring_capacity": 8, "dispatch_ring_capacity": 8,
      "total_allocs_recorded": 1, "total_dispatches_recorded": 1,
      "allocs": [{"alloc_id": 0, "device": "AMD", "va_start": base, "va_end": base + 0x1000,
                  "req_size": 0x672000 // 10, "mapped_size": 0x1000, "alloc_seq": 0, "free_seq": None,
                  "alloc_ts_ns": 10, "free_ts_ns": None}],
      "dispatches": [{"dispatch_id": 0, "device": "AMD", "kernel": "r_conv", "global_size": [4,1,1], "local_size": [1,1,1],
                       "submit_seq": 0, "submit_ts_ns": 20, "signal_target": 1, "completed": True,
                       "args": [{"va": base, "size": 0x1000}]}],
    }
    with tempfile.TemporaryDirectory() as td:
      p = os.path.join(td, "dump.json")
      json.dump(dump, open(p, "w"))
      allocs, dispatches, meta = AF.load_dump(p)
    self.assertEqual(len(allocs), 1); self.assertEqual(len(dispatches), 1)
    r = AF.attribute(base + 0x30000, ts_ns=None, allocs=allocs, dispatches=dispatches)
    # base+0x30000 is well past the 0x1000 allocation -- must be a clean out-of-bounds report, not a crash.
    self.assertEqual(r["containing"], [])
    self.assertIs(r["nearest_below"], allocs[0])
    self.assertIsNone(r["nearest_above"])


if __name__ == "__main__":
  unittest.main()
