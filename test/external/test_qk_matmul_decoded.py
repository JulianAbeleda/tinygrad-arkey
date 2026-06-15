import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
N0 = REPO / "bench/amd-decode-flywheel-proof-20260614/native-matmul-N0"


class TestQKMatmulDecoded(unittest.TestCase):
  def test_committed_n0a_matmul_decoded(self):
    f = N0 / "n0a_summary.json"
    if not f.exists():
      self.skipTest("N0a not run yet")
    d = json.loads(f.read_text())
    self.assertEqual(d["phase"], "Phase N0a")
    self.assertTrue(len(d["curve"]) >= 4)
    for c in d["curve"]:
      self.assertTrue(c["correct"], c["shape"])
      # H-N0: matmul_decoded (per-call, incl. the dequant pass) beats the fused split-K kernel
      self.assertGreater(c["percall_vs_fused"], 1.0, c["shape"])
      # native matmul lives on the rich opt space (reaches a real fraction of peak at batch)
    # at large batch the native matmul is well into the compute-bound band (>=20% peak)
    big = max(d["curve"], key=lambda c: c["shape"]["N"])
    self.assertGreater(big["matmul_pct_peak"], 20.0)


if __name__ == "__main__":
  unittest.main()
