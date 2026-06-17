#!/usr/bin/env python3
"""Lock Phase 7a finding: same-process two-stream decode overlap is BLOCKED by the dispatch path -- a real decode
with AMD_COMPUTE_RINGS=2 only ever uses ring 0 (no per-stream ring-routing API), and the shared global timeline
serializes dispatch. Skip-if-absent; the device probe (extra/amd_two_stream_decode_probe.py) stays out of the suite."""
import json, pathlib, unittest

_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "amd-two-stream-decode-probe" / "result.json"

class TestAMDTwoStreamDecode(unittest.TestCase):
  def test_routing_blocked(self):
    if not _ARTIFACT.exists(): self.skipTest(f"no artifact at {_ARTIFACT}")
    d = json.loads(_ARTIFACT.read_text())
    self.assertTrue(d["routing_blocked"], "decode unexpectedly used ring 1 -- re-examine the routing finding")
    self.assertEqual(d["decode_rings_used"], [0], "a real decode should only touch ring 0 (no routing API)")

if __name__ == "__main__":
  unittest.main()
