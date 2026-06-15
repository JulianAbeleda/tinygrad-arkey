import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
W1B = REPO / "bench/amd-decode-flywheel-proof-20260614/wmma-w1b"


class TestQKMarlinW1b(unittest.TestCase):
  def test_committed_w1b_marlin_primitive(self):
    if not (W1B / "summary.json").exists():
      self.skipTest("W1b not run yet")
    d = json.loads((W1B / "summary.json").read_text())
    self.assertEqual(d["phase"], "Phase W1b")
    g = d["gates"]
    # the three Track A gates: TC on a hand AST, TC on an LDS-staged operand, full Marlin correct
    self.assertTrue(g["a0a_tc_on_hand_ast"])
    self.assertTrue(g["a0b_tc_on_lds_staged_operand"])
    self.assertTrue(g["a1_marlin_fused_wmma_correct"])
    self.assertTrue(len(d["curve"]) >= 4)
    for c in d["curve"]:
      # fused marlin (reads compressed) AND the fp16 ceiling are both numerically correct
      self.assertTrue(c["marlin_correct"], c["shape"])
      self.assertTrue(c["ceiling_correct"], c["shape"])
      self.assertLess(c["marlin_rel_err"], 1e-2, c["shape"])
      # the whole weight tile fits in LDS (no K-tiling yet)
      self.assertLessEqual(c["lds_bytes"], 64 * 1024, c["shape"])
    # the W1b claim: fusing the dequant is ~free vs the materialized-fp16 WMMA ceiling
    mean_ratio = sum(c["marlin_vs_ceiling"] for c in d["curve"]) / len(d["curve"])
    self.assertGreater(mean_ratio, 0.9)


if __name__ == "__main__":
  unittest.main()
