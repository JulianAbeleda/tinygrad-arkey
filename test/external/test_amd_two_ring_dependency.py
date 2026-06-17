#!/usr/bin/env python3
"""Lock Phase 4: AMD cross-ring dependency/wait semantics work. A consumer on the other ring that WAITS on the
producer's signal always sees the write (both directions + copy-queue-after-compute); the no-wait control races
(reads stale). Skip-if-absent; the device probe (extra/amd_two_ring_dependency_probe.py) stays out of the suite."""
import json, pathlib, unittest

_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "amd-two-ring-dependency-probe" / "result.json"

class TestAMDTwoRingDependency(unittest.TestCase):
  def test_cross_ring_ordering(self):
    if not _ARTIFACT.exists(): self.skipTest(f"no artifact at {_ARTIFACT}")
    d = json.loads(_ARTIFACT.read_text()); reps = d["reps"]
    self.assertEqual(d["fwd_0to1_correct"], reps, "ring0->ring1 wait did not order")
    self.assertEqual(d["rev_1to0_correct"], reps, "ring1->ring0 wait did not order")
    self.assertEqual(d["copyq_after_compute_correct"], reps, "copy queue did not wait on compute signal")
    self.assertLess(d["nowait_control_correct"], reps, "no-wait control should race (wait must matter)")
    self.assertTrue(d["passes"])

if __name__ == "__main__":
  unittest.main()
