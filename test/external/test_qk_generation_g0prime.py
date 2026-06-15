import json, pathlib, unittest

from extra.qk_generation_g0prime import MODES, NOISE_BAND_PCT, _candidates

REPO = pathlib.Path(__file__).resolve().parents[2]
G0P = REPO / "bench/amd-decode-flywheel-proof-20260614/generation-g0prime"


class TestQKGenerationG0Prime(unittest.TestCase):
  def test_grid_sweeps_modes_and_occupancy(self):
    cands = _candidates()
    labels = [c["label"] for c in cands]
    self.assertEqual(len(labels), len(set(labels)))
    self.assertIn("v1_partial", labels)
    # genuinely different dequant/load kernels (not just ILP knobs) are swept
    for mode in ("packed_load", "vector_load", "grouped", "tile_custom"):
      self.assertTrue(any(c["mode"] == mode for c in cands), mode)
    # occupancy lever (parts) is varied
    self.assertTrue(any(c["parts"] > 1 for c in cands))

  def test_committed_g0prime_verdict(self):
    if not (G0P / "summary.json").exists():
      self.skipTest("G0' not run yet")
    d = json.loads((G0P / "summary.json").read_text())
    self.assertEqual(d["phase"], "Phase G0'")
    # any_real_headroom <=> the verdict is one of the win conclusions (not the null one).
    self.assertEqual(d["any_real_headroom"], not d["conclusion"].startswith("no_existing_kernel_beats"))
    for tensor, e in d["per_tensor"].items():
      self.assertIn("baseline_roofline_pct", e)
      self.assertIn("beats_baseline", e)
      # if a kernel beats baseline it must clear the noise band
      if e["beats_baseline"]:
        self.assertGreater(e["best_gain_vs_baseline_pct"], NOISE_BAND_PCT)


if __name__ == "__main__":
  unittest.main()
