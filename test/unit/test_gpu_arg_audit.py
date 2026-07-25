"""Pins GPU_ARG_AUDIT, the detector for the one live gfx1100 page-fault signature.

Context in docs/gpu-page-fault-population-analysis-20260725.md: 56 of 145 faults land on
0x0000ffffffbfe000 (the 48-bit sign-extension of the int32 -0x402000) and 27 more on exactly 0x0. Both are
what a kernel produces when its BASE POINTER IS ZERO and it then applies an offset. A null base can only
reach the GPU through a kernel-argument address slot, so the detector guards those two write sites.

These tests exist so the detector cannot silently rot into a no-op -- a detector that cannot fire is worse
than none, because a silent run then reads as evidence.
"""
import unittest
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
import tinygrad.device as D


class _Buf:
  def __init__(self, va): self.va_addr = va


class TestGPUArgAudit(unittest.TestCase):
  def setUp(self):
    self._prev, D.GPU_ARG_AUDIT.value = D.GPU_ARG_AUDIT.value, 1
    D._arg_audit_hits.clear(); D._arg_coverage_hits.clear()

  def tearDown(self):
    D.GPU_ARG_AUDIT.value = self._prev
    D._arg_audit_hits.clear(); D._arg_coverage_hits.clear()

  def test_null_address_is_detected(self):
    D._audit_kernarg_bufs("AMD", "k", (_Buf(0x7f0012340000), _Buf(0), _Buf(0x7f0012350000)))
    self.assertEqual(len(D._arg_audit_hits), 1)
    dev, prg, idx, nbufs = D._arg_audit_hits[0]
    self.assertEqual((dev, prg, idx, nbufs), ("AMD", "k", 1, 3))

  def test_real_addresses_do_not_fire(self):
    D._audit_kernarg_bufs("AMD", "k", (_Buf(0x7f0012340000), _Buf(0x200000000000)))
    self.assertEqual(D._arg_audit_hits, [])

  def test_symbolic_address_does_not_fire_and_does_not_raise(self):
    # Graph-captured buffers carry a symbolic UOp placeholder patched at replay. Truthiness on a UOp raises,
    # so the detector must type-check before testing the value. This regressed once and broke a real run.
    sym = UOp.variable("inp_0_0", 0, 0xffffffffffffffff, dtype=dtypes.ulong)
    D._audit_kernarg_bufs("AMD", "k", (_Buf(sym),))
    self.assertEqual(D._arg_audit_hits, [])

  def test_disabled_is_a_no_op(self):
    D.GPU_ARG_AUDIT.value = 0
    D._audit_kernarg_bufs("AMD", "k", (_Buf(0),))
    self.assertEqual(D._arg_audit_hits, [])

  def test_level_2_raises_on_first_hit(self):
    D.GPU_ARG_AUDIT.value = 2
    with self.assertRaises(RuntimeError) as cm:
      D._audit_kernarg_bufs("AMD", "some_kernel", (_Buf(0),))
    self.assertIn("some_kernel", str(cm.exception))

  # --- kernarg segment coverage: the second, cheaper form of the same failure ---

  def test_under_written_kernarg_segment_is_detected(self):
    # Kernel declares 64B of arguments, we only write 40 -> the trailing 24B are recycled bump-allocator
    # memory. A kernel reading an address out of that gap gets a stale or null base.
    D._audit_kernarg_coverage("k", 40, 64)
    self.assertEqual(D._arg_coverage_hits, [("k", 40, 64)])

  def test_exact_and_over_written_segments_do_not_fire(self):
    D._audit_kernarg_coverage("k", 64, 64)
    D._audit_kernarg_coverage("k", 72, 64)
    self.assertEqual(D._arg_coverage_hits, [])

  def test_unknown_declared_size_does_not_fire(self):
    # Non-AMD HCQ programs may not expose kernargs_segment_size; absence must not be read as a gap.
    D._audit_kernarg_coverage("k", 40, None)
    self.assertEqual(D._arg_coverage_hits, [])

  def test_coverage_level_2_raises(self):
    D.GPU_ARG_AUDIT.value = 2
    with self.assertRaises(RuntimeError) as cm: D._audit_kernarg_coverage("some_kernel", 40, 64)
    self.assertIn("uninitialised", str(cm.exception))

  def test_both_write_sites_are_still_hooked(self):
    # The detector is only as good as its call sites. If either write site loses its hook the detector goes
    # silent while looking healthy, so pin them textually.
    import pathlib
    root = pathlib.Path(D.__file__).parent
    hcq = (root/"runtime"/"support"/"hcq.py").read_text()
    graph = (root/"runtime"/"graph"/"hcq.py").read_text()
    self.assertIn("_audit_kernarg_bufs", hcq, "HCQArgsState (fill_kernargs path) lost its audit hook")
    self.assertIn("_audit_kernarg_coverage", hcq, "CLikeArgsState lost its kernarg-coverage hook")
    self.assertIn("_audit_kernarg_bufs", graph, "HCQGraph.__call__ (graph replay path) lost its audit hook")


if __name__ == "__main__":
  unittest.main()
