"""Pins DISPATCH_TRACE, the fault-to-dispatch correlation probe (docs/gpu-fault-fix-scope-20260725.md).

dmesg names the faulting VA and pid but never the kernel that was dispatched. Every indirect probe tried so
far came back ambiguous. DISPATCH_TRACE closes that gap directly: when on, it serializes so at most one
dispatch is ever in flight, records its name right before the wait that could fault on it, and if that wait
raises, the last recorded dispatch names the offender -- no inference.

A detector that cannot fire is worse than no detector, because a silent run then reads as evidence of
absence. These tests exist so DISPATCH_TRACE cannot silently lose its hook while looking healthy.
"""
import unittest
import tinygrad.device as D


class _Sig:
  def __init__(self, v): self.value = v


class _Dev:
  def __init__(self, name="AMD", timeline_value=7, has_signal=True):
    self.device = name
    self.timeline_value = timeline_value
    if has_signal: self.timeline_signal = _Sig(0)


class TestDispatchTraceRecording(unittest.TestCase):
  def setUp(self):
    self._prev, D.DISPATCH_TRACE.value = D.DISPATCH_TRACE.value, 1
    D._dispatch_trace_ring.clear(); D._dispatch_trace_inflight[0] = None

  def tearDown(self):
    D.DISPATCH_TRACE.value = self._prev
    D._dispatch_trace_ring.clear(); D._dispatch_trace_inflight[0] = None

  def test_before_records_the_inflight_dispatch(self):
    dev = _Dev()
    D._dispatch_trace_before(dev, "r_64_4", (64, 1, 1), (4, 1, 1))
    self.assertIsNotNone(D._dispatch_trace_inflight[0])
    device, name, pid, tv, gs, ls = D._dispatch_trace_inflight[0]
    self.assertEqual((device, name, gs, ls), ("AMD", "r_64_4", (64, 1, 1), (4, 1, 1)))
    self.assertEqual(len(D._dispatch_trace_ring), 1)

  def test_after_clears_inflight_on_clean_completion(self):
    dev = _Dev()
    D._dispatch_trace_before(dev, "r_64_4", (64, 1, 1), (4, 1, 1))
    D._dispatch_trace_after()
    self.assertIsNone(D._dispatch_trace_inflight[0])
    # history is retained even after a clean completion, for context
    self.assertEqual(len(D._dispatch_trace_ring), 1)

  def test_dump_names_the_dispatch_still_inflight_on_error(self):
    dev = _Dev()
    D._dispatch_trace_before(dev, "r_offending_kernel", (1, 1, 1), (1, 1, 1))
    # no _dispatch_trace_after() call -- simulates the synchronize() that raised
    with self.assertLogs(level="INFO") if False else _capture_stdout() as out:
      D._dispatch_trace_dump(dev, RuntimeError("MMU fault: 0x7c7a20b4a000"))
    self.assertIn("r_offending_kernel", out.getvalue())
    self.assertIn("MMU fault", out.getvalue())

  def test_dump_with_nothing_inflight_says_so_instead_of_guessing(self):
    dev = _Dev()
    with _capture_stdout() as out:
      D._dispatch_trace_dump(dev, RuntimeError("boom"))
    self.assertIn("no traced dispatch in flight", out.getvalue())

  def test_non_hcq_backend_is_a_no_op(self):
    dev = _Dev(has_signal=False)
    D._dispatch_trace_before(dev, "r_64_4", (64, 1, 1), (4, 1, 1))
    self.assertIsNone(D._dispatch_trace_inflight[0])
    self.assertEqual(len(D._dispatch_trace_ring), 0)

  def test_disabled_is_a_no_op(self):
    D.DISPATCH_TRACE.value = 0
    dev = _Dev()
    D._dispatch_trace_before(dev, "r_64_4", (64, 1, 1), (4, 1, 1))
    self.assertIsNone(D._dispatch_trace_inflight[0])

class TestDispatchTraceDefault(unittest.TestCase):
  def test_default_is_off(self):
    # DISPATCH_TRACE serializes every dispatch (a synchronize() per dispatch); if this default ever flips
    # to 1 every backend pays a correctness-probe tax that was only ever meant for deliberate fault-hunting.
    # NOTE: no setUp here on purpose -- TestDispatchTraceRecording forces the value to 1 for its own tests.
    self.assertEqual(D.DISPATCH_TRACE.value, 0, "DISPATCH_TRACE must ship default-off")


class TestDispatchTraceWired(unittest.TestCase):
  def test_call_sites_still_exist(self):
    # The detector is only as good as its two hooks: if HCQProgram.__call__ or HCQCompiled.synchronize lose
    # them, the probe goes silent while looking healthy.
    import pathlib
    hcq = (pathlib.Path(D.__file__).parent/"runtime"/"support"/"hcq.py").read_text()
    self.assertIn("_dispatch_trace_before", hcq, "HCQProgram.__call__ lost its dispatch-trace record hook")
    self.assertIn("_dispatch_trace_after", hcq, "HCQProgram.__call__ lost its dispatch-trace clear hook")
    self.assertIn("_dispatch_trace_dump", hcq, "HCQCompiled.synchronize lost its dispatch-trace dump hook")


import contextlib, io

@contextlib.contextmanager
def _capture_stdout():
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf): yield buf


if __name__ == "__main__":
  unittest.main()
