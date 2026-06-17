#!/usr/bin/env python3
"""Lock Phase 3: AMD same-process two-ring compute overlap is REAL. Two compute-bound kernels routed to ring 0
and ring 1 (AMD_COMPUTE_RINGS=2) run concurrently (~2x), while the SAME pairing on one ring serializes (~1x).
Skip-if-absent; the GPU-timestamp probe (extra/amd_two_compute_ring_probe.py) stays out of the suite."""
import json, pathlib, unittest

_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "amd-two-compute-ring-probe" / "result.json"

class TestAMDTwoComputeRing(unittest.TestCase):
  def test_two_ring_overlap_real(self):
    if not _ARTIFACT.exists(): self.skipTest(f"no artifact at {_ARTIFACT}")
    d = json.loads(_ARTIFACT.read_text())
    self.assertTrue(d["outputs_correct"], "two-ring kernel outputs incorrect")
    self.assertGreater(d["two_ring_overlap_x"], 1.2, "two rings did not overlap")
    self.assertLess(d["one_ring_control_x"], 1.15, "one-ring control should serialize (~1.0x)")
    self.assertTrue(d["passes"])

if __name__ == "__main__":
  unittest.main()
