"""Pins KERNARGS_AUDIT, the detector for the live gfx1100 SQC (inst) page-fault signature.

dmesg classifies the live faults precisely (docs/gpu-page-fault-population-analysis-20260725.md): every
fault at 0x0000ffffffbfe000, 0x100000000 and 0x0 reports `Faulty UTCL2 client ID: SQC (inst)`. Those are
program counters, not data pointers. A wave's entry PC comes from the dispatch packet's kernel_object, and
that packet lives inside the kernargs allocation, which a wrapping BumpAllocator recycles with no check that
the memory belongs to an in-flight dispatch.

A detector that cannot fire is worse than no detector, because a silent run then reads as evidence. These
tests exist so that cannot happen quietly.
"""
import unittest
from tinygrad.runtime.support.memory import BumpAllocator
import tinygrad.device as D


class _Sig:
  def __init__(self, v): self.value = v


class _Dev:
  def __init__(self, observed, pending, name="AMD"):
    self.device, self.timeline_signal, self.timeline_value = name, _Sig(observed), pending + 1


class TestBumpAllocatorWrapCounter(unittest.TestCase):
  def test_counter_starts_at_zero_and_increments_only_on_wrap(self):
    a = BumpAllocator(64, wrap=True)
    a.alloc(32); self.assertEqual(a.wraps, 0)
    a.alloc(32); self.assertEqual(a.wraps, 0)   # exactly fills
    a.alloc(32); self.assertEqual(a.wraps, 1)   # must reuse from 0
    self.assertEqual(a.ptr, 32)

  def test_non_wrapping_allocator_still_raises(self):
    a = BumpAllocator(32, wrap=False)
    a.alloc(32)
    with self.assertRaises(RuntimeError): a.alloc(1)
    self.assertEqual(a.wraps, 0)


class TestKernargsWrapAudit(unittest.TestCase):
  def setUp(self):
    self._prev, D.KERNARGS_AUDIT.value = D.KERNARGS_AUDIT.value, 1
    D._kernargs_wrap_hits.clear(); D._kernargs_wrap_total[0] = 0

  def tearDown(self):
    D.KERNARGS_AUDIT.value = self._prev
    D._kernargs_wrap_hits.clear(); D._kernargs_wrap_total[0] = 0

  def test_wrap_with_undrained_work_is_recorded(self):
    D._audit_kernargs_wrap(_Dev(observed=18, pending=19), wrapped=True)
    self.assertEqual(D._kernargs_wrap_hits, [("AMD", 18, 19)])
    self.assertEqual(D._kernargs_wrap_total[0], 1)

  def test_wrap_on_a_drained_device_is_counted_but_not_a_hit(self):
    D._audit_kernargs_wrap(_Dev(observed=19, pending=19), wrapped=True)
    self.assertEqual(D._kernargs_wrap_hits, [])
    self.assertEqual(D._kernargs_wrap_total[0], 1, "a safe wrap must still be counted, so 0 hits is readable")

  def test_no_wrap_is_not_examined_at_all(self):
    D._audit_kernargs_wrap(_Dev(observed=1, pending=99), wrapped=False)
    self.assertEqual(D._kernargs_wrap_hits, [])
    self.assertEqual(D._kernargs_wrap_total[0], 0)

  def test_non_hcq_backend_is_a_no_op(self):
    class Bare: device = "METAL"
    D._audit_kernargs_wrap(Bare(), wrapped=True)   # no timeline_signal
    self.assertEqual(D._kernargs_wrap_hits, [])

  def test_disabled_is_a_no_op(self):
    D.KERNARGS_AUDIT.value = 0
    D._audit_kernargs_wrap(_Dev(observed=1, pending=99), wrapped=True)
    self.assertEqual(D._kernargs_wrap_total[0], 0)

  def test_level_2_raises(self):
    D.KERNARGS_AUDIT.value = 2
    with self.assertRaises(RuntimeError) as cm:
      D._audit_kernargs_wrap(_Dev(observed=1, pending=99), wrapped=True)
    self.assertIn("wild PC", str(cm.exception))

  def test_the_fix_is_on_by_default(self):
    # KERNARGS_WRAP_DRAIN is the fix: it drains before the kernargs allocator reuses memory an in-flight
    # dispatch may still be reading. Measured A/B at 512B: guard off -> [15,14,15,15] reuses-in-flight,
    # guard on -> [0,0,0,0]. If this default ever flips to 0 the hazard is live again.
    self.assertEqual(D.KERNARGS_WRAP_DRAIN.value, 1, "the kernargs wrap drain must ship enabled")

  def test_the_fix_and_its_audit_are_both_wired(self):
    import pathlib
    hcq = (pathlib.Path(D.__file__).parent/"runtime"/"support"/"hcq.py").read_text()
    self.assertIn("KERNARGS_WRAP_DRAIN", hcq, "fill_kernargs lost the wrap drain (the fix)")
    self.assertIn("self.dev.synchronize()", hcq, "the drain itself is gone")

  def test_the_call_site_is_still_hooked(self):
    # The detector is only as good as its one call site; if fill_kernargs loses it the detector goes silent
    # while looking healthy.
    import pathlib
    hcq = (pathlib.Path(D.__file__).parent/"runtime"/"support"/"hcq.py").read_text()
    self.assertIn("_audit_kernargs_wrap", hcq, "fill_kernargs lost its kernargs-wrap audit hook")


if __name__ == "__main__":
  unittest.main()
