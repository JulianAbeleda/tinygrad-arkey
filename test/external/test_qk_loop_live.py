"""Phase L: tests for the live autotuner (extra/qk_loop_live.py) -- pure-python logic, no device.

Covers the integrity + correctness properties of the harness: fresh shapes are genuinely held out, the
candidate feature encoding matches the corpus convention (the TC tc_level round-trip fix), the ranking
is deterministic under the fixed seed, numpy scalars serialize, and the random baseline is well-formed.
The live device-timing path (live_time_shape/evaluate_shape) is exercised by the L0/L1 runs, not here.
"""
import json, unittest
import numpy as np

from tinygrad.codegen.opt import Opt, OptOps
from extra.qk_loop_live import _cand_feature_rows, _pyify, _random_bestofk, FRESH_L0, FRESH_L1
from extra.qk_loop_learnability import load_merged, _train_predict, FEAT_KEYS


class TestQKLoopLive(unittest.TestCase):
  def test_fresh_shapes_absent_from_corpus(self):
    corpus_shapes = {r["shape"] for r in load_merged()}
    for s in [FRESH_L0, *FRESH_L1]:
      self.assertNotIn(s, corpus_shapes, f"{s} is in the corpus -- not a held-out test")

  def test_candidate_feature_encoding_matches_corpus_tc_convention(self):
    # corpus features come from jsonl where TC arg is a LIST -> _opt_feats reads tc_level = arg[1].
    # _cand_feature_rows must JSON round-trip so an in-process Opt tuple gives the SAME tc_level.
    cands = [[Opt(OptOps.TC, 0, (-1, 2, 1))], [Opt(OptOps.TC, 0, (-1, 0, 1))], []]
    rows = _cand_feature_rows(4096, 14336, 128, cands)
    tci = FEAT_KEYS.index("tc_level")
    has_tc_i = FEAT_KEYS.index("has_tc")
    self.assertEqual(rows[0]["x"][tci], 2)   # TC(-1,2,1) -> level 2
    self.assertEqual(rows[1]["x"][tci], 0)   # TC(-1,0,1) -> level 0
    self.assertEqual(rows[0]["x"][has_tc_i], 1)
    self.assertEqual(rows[2]["x"][has_tc_i], 0)  # no-opt baseline
    self.assertTrue(all(len(r["x"]) == len(FEAT_KEYS) for r in rows))

  def test_ranking_is_deterministic_under_fixed_seed(self):
    rows = load_merged()
    shapes = sorted({r["shape"] for r in rows})
    held = shapes[0]
    train = [r for r in rows if r["shape"] != held]
    test = [r for r in rows if r["shape"] == held]
    p1 = _train_predict(train, test)
    p2 = _train_predict(train, test)
    self.assertTrue(np.array_equal(p1, p2), "fixed-seed XGBoost predictions must be reproducible")

  def test_pyify_makes_numpy_json_serializable(self):
    obj = {"a": np.float64(0.5), "b": np.int64(3), "c": np.bool_(True),
           "d": [np.float32(1.0), {"e": np.bool_(False)}]}
    out = _pyify(obj)
    json.dumps(out)  # must not raise
    self.assertIsInstance(out["a"], float)
    self.assertIsInstance(out["b"], int)
    self.assertIsInstance(out["c"], bool)
    self.assertIsInstance(out["d"][1]["e"], bool)

  def test_random_bestofk_is_monotonic_and_bounded(self):
    tflops = [1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 8.0]
    oracle = max(tflops)
    res = _random_bestofk(tflops, oracle, [1, 2, 4, 8], draws=2000)
    Ks = sorted(res)
    fracs = [res[k] for k in Ks]
    self.assertTrue(all(0.0 < f <= 1.0 for f in fracs))
    self.assertTrue(all(fracs[i] <= fracs[i+1] + 1e-9 for i in range(len(fracs)-1)), "best-of-K rises in K")
    self.assertAlmostEqual(res[len(tflops)], 1.0, places=6)  # best-of-all = oracle


if __name__ == "__main__":
  unittest.main()
