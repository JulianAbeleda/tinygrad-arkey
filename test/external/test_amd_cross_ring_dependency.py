#!/usr/bin/env python3
"""Lock Phase 4 (complete): AMD cross-ring dependency semantics. Deps order correctly both ways, independent
kernels overlap on two rings, and adding a B-waits-A dependency serializes them -- i.e. dependency cost appears
ONLY when required (multi-ring is schedulable safely). Magnitude is power/thermal-capped (~1.34x warm); gated
qualitatively (>1.15x direction). Skip-if-absent."""
import json, pathlib, unittest

_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "amd-cross-ring-dependency-probe" / "result.json"

class TestAMDCrossRingDependency(unittest.TestCase):
  def test_dependency_semantics(self):
    if not _ARTIFACT.exists(): self.skipTest(f"no artifact at {_ARTIFACT}")
    d = json.loads(_ARTIFACT.read_text()); reps = d["reps"]
    self.assertEqual(d["fwd_ring0_to_ring1_correct"], reps, "ring0->ring1 dependency incorrect")
    self.assertEqual(d["rev_ring1_to_ring0_correct"], reps, "ring1->ring0 dependency incorrect")
    self.assertGreater(d["twoRing_overlap_control_x"], 1.15, "independent kernels did not overlap on two rings")
    self.assertGreater(d["dependent_serialize_x"], 1.15, "B-waits-A dependency did not serialize")
    self.assertTrue(d["passes"])

if __name__ == "__main__":
  unittest.main()
