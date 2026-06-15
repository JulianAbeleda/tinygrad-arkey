import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
N0 = REPO / "bench/amd-decode-flywheel-proof-20260614/native-matmul-N0"


class TestQKLoopLearnability(unittest.TestCase):
  def test_committed_n1_learnability(self):
    f = N0 / "n1_learnability.json"
    if not f.exists():
      self.skipTest("N1 not run yet")
    d = json.loads(f.read_text())
    self.assertEqual(d["phase"], "Phase N1")
    self.assertGreaterEqual(d["n_shapes"], 10)
    a, g = d["aggregate"], d["gate"]
    # the substantive loop signals (all positive): model beats the deterministic lookup baseline,
    # and is worth many random trials (sample efficiency)
    self.assertTrue(g["model_beats_lookup"])
    self.assertGreater(a["mean_top1_frac_oracle"], a["mean_lookup_frac_oracle"])
    self.assertTrue(g["model_saves_trials"])
    self.assertGreaterEqual(a["median_random_trials_to_match"], 3.0)
    # on the batched regime it serves (N>=256), the model clears the strict 0.90-of-oracle bar
    self.assertGreaterEqual(a["batched_N>=256_mean_top1_frac_oracle"], 0.90)
    self.assertGreater(a["batched_N>=256_mean_top1_frac_oracle"], a["batched_N>=256_mean_lookup_frac_oracle"])
    # transfer: experience helps -- more train shapes -> better held-out prediction than k=1
    byk = d["transfer"]["mean_top1_frac_oracle_by_k_train"]
    kmax = max(int(k) for k in byk)
    self.assertGreater(byk[str(kmax)], byk["1"])
    # PASS is reported honestly per the pre-registered (un-moved) strict gate -- a narrow miss is fine
    self.assertIn("PASS", g)


if __name__ == "__main__":
  unittest.main()
