import json, pathlib, unittest

from extra.qk_metric_audit import MECHANISM_SPEC, NOISE_BAND_PCT, PEAK_GBS_MEASURED

REPO = pathlib.Path(__file__).resolve().parents[2]
M0 = REPO / "bench/amd-decode-flywheel-proof-20260614/metric-audit-m0"


class TestQKMetricAudit(unittest.TestCase):
  def test_audit_constants_are_sane(self):
    # Measured peak is below the 960 datasheet and well above the kernel's achieved rate.
    self.assertLess(PEAK_GBS_MEASURED, 960.0)
    self.assertGreater(PEAK_GBS_MEASURED, 700.0)
    self.assertGreater(NOISE_BAND_PCT, 0.0)
    self.assertEqual(MECHANISM_SPEC["v1_partial"]["mode"], "partial")
    self.assertIn("UPCAST:0:2", MECHANISM_SPEC["row_upcast"]["opts"])

  def test_committed_m0a_confirms_4x_wins_were_noise(self):
    if not (M0 / "summary.json").exists():
      self.skipTest("M0a not run yet")
    d = json.loads((M0 / "summary.json").read_text())
    self.assertEqual(d["phase"], "Phase M0a")
    # Re-audited on the device metric, no 4.x raw_accept beats v1_partial beyond the noise band.
    self.assertEqual(d["real_device_wins"], 0)
    self.assertLess(d["median_device_gain_pct_of_4x_raw_accepts"], 0.0)
    self.assertTrue(d["conclusion"].startswith("4x_wins_confirmed_noise"))
    # Headroom is real and shape-dependent (attn_q further from peak than ffn_gate).
    roof = d["v1_partial_roofline_pct_by_tensor"]
    self.assertTrue(roof)
    self.assertTrue(all(0.0 < p < 100.0 for p in roof.values()))

  def test_committed_m0b_names_dequant_bottleneck(self):
    if not (M0 / "bottleneck-m0b.json").exists():
      self.skipTest("M0b not run yet")
    d = json.loads((M0 / "bottleneck-m0b.json").read_text())
    self.assertEqual(d["bottleneck"], "q4_k_dequant_compute_and_occupancy")
    # Loads are already wide -> width is not the bottleneck; ALU dominates.
    for k in d["per_kernel"].values():
      self.assertGreater(k["global_load_b128"], k["global_load_b32"])
      self.assertGreater(k["total_vector_alu"], 10 * (k["global_load_b128"] + k["global_load_b64"] + k["global_load_b32"]))
    self.assertIn("UPCAST", d["implications"]["drop_irrelevant_axes"])


if __name__ == "__main__":
  unittest.main()
