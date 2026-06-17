#!/usr/bin/env python3
"""Lock the multi-process decode serving-throughput finding: two concurrent Qwen3-8B decode processes deliver a
material aggregate tok/s gain over one process (cross-process hardware overlap; no runtime/dispatch changes),
with identical (sane) greedy output. Bounded by HBM bandwidth (~1.3x ceiling, Phase 6). Locks >=1.15x to be
robust to measurement noise (stable ~1.21x with the long window). Skip-if-absent."""
import json, pathlib, unittest

_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "amd-multiprocess-decode-throughput" / "result.json"

class TestAMDMultiprocessDecode(unittest.TestCase):
  def test_serving_throughput_gain(self):
    if not _ARTIFACT.exists(): self.skipTest(f"no artifact at {_ARTIFACT}")
    d = json.loads(_ARTIFACT.read_text())
    self.assertTrue(d["output_sane"], "concurrent decode corrupted output")
    self.assertGreaterEqual(d["aggregate_speedup"], 1.15, "cross-process decode should give a material aggregate gain")

if __name__ == "__main__":
  unittest.main()
