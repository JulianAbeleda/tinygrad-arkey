#!/usr/bin/env python3
"""Lock Phase 5: a minimal cross-ring dependency-DAG scheduler overlaps a dependent chain (A0->A1) on one ring
with independent work (B) on another, with correct cross-ring dependencies and a join. Proves the decode-overlap
scheduling CONCEPT outside the model. Skip-if-absent; the device probe (extra/amd_two_ring_dag_probe.py) stays
out of the suite."""
import json, pathlib, unittest

_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "amd-two-ring-dag-probe" / "result.json"

class TestAMDTwoRingDAG(unittest.TestCase):
  def test_dag_scheduler_overlaps_with_correct_deps(self):
    if not _ARTIFACT.exists(): self.skipTest(f"no artifact at {_ARTIFACT}")
    d = json.loads(_ARTIFACT.read_text())
    self.assertTrue(d["dependency_correct"], "A1 did not see A0's output (dependency broken)")
    self.assertTrue(d["independent_ran"], "independent task B did not run")
    self.assertGreater(d["overlap_x"], 1.2, "DAG scheduler did not overlap chain with independent work")
    self.assertTrue(d["passes"])

if __name__ == "__main__":
  unittest.main()
