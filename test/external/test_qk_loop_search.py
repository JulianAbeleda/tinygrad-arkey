import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
N0 = REPO / "bench/amd-decode-flywheel-proof-20260614/native-matmul-N0"


class TestQKLoopSearch(unittest.TestCase):
  def test_committed_n2_loop(self):
    f = N0 / "n2_loop_search.json"
    if not f.exists():
      self.skipTest("N2 not run yet")
    d = json.loads(f.read_text())
    self.assertEqual(d["phase"], "Phase N2")
    a = d["n2a_guided_vs_random"]
    bok = a["best_of_k_frac_oracle"]
    # model-guided beats random best-of-K at every budget
    for K, v in bok.items():
      self.assertGreater(v["model"], v["random"], K)
    # guided reaches >=95% of oracle by K=8; near-oracle by K=50
    self.assertGreaterEqual(bok["8"]["model"], 0.95)
    self.assertGreaterEqual(bok["50"]["model"], 0.99)
    # the loop's headline: far fewer trials to 95% than random
    self.assertLessEqual(a["median_guided_trials_to_95pct"], 5)
    self.assertGreaterEqual(a["median_random_trials_to_95pct"], 3 * a["median_guided_trials_to_95pct"])
    # online flywheel: best-of-5 improves as the corpus grows
    bc = d["n2b_online_accumulation"]["K5_bestofk_frac_oracle_by_corpus"]
    kmax = max(int(k) for k in bc)
    self.assertGreater(bc[str(kmax)], bc["1"])
    self.assertTrue(d["gate"]["PASS"])


class TestQKLoopGateClosed(unittest.TestCase):
  def test_n1_strict_gate_now_passes(self):
    f = N0 / "n1_learnability.json"
    if not f.exists():
      self.skipTest("N1 not run yet")
    d = json.loads(f.read_text())
    # after N1.1 small-N coverage, the merged-dataset run clears the pre-registered strict gate
    if d["n_shapes"] >= 24:
      self.assertGreaterEqual(d["aggregate"]["mean_top1_frac_oracle"], 0.90)
      self.assertTrue(d["gate"]["PASS"])


if __name__ == "__main__":
  unittest.main()
