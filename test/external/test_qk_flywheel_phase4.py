import hashlib, json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_cost_model import extract_feature_map
from extra.qk_flywheel_shadow import (
  FRESH_SPECS, LEAKAGE_TOKENS, build_fresh_candidates, dead_branch_metric, freeze_predictions, fresh_id,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SHADOW = REPO / "bench/amd-decode-flywheel-proof-20260614/shadow-v0"


class TestQKFlywheelPhase4(unittest.TestCase):
  def test_fresh_candidates_are_static_blind_and_leak_free(self):
    rows = build_fresh_candidates(REPO)
    self.assertEqual(len(rows), len(FRESH_SPECS))
    self.assertEqual([row["id"] for row in rows], [fresh_id(spec) for spec in FRESH_SPECS])
    for row in rows:
      self.assertEqual(row["schema_version"], "kernel_triage_v1")
      self.assertEqual(row["prediction_stage"], "after_static_before_microbench")
      self.assertFalse(row["outcome_known"])
      # Static-blind: no compile/source/microbench-derived context before running.
      ctx = row["pre_result_context"]
      self.assertNotIn("comparison", ctx)
      self.assertNotIn("wide_loads", ctx)
      self.assertNotIn("source_ok", ctx)
      feature_keys = "\n".join(extract_feature_map(row))
      for token in LEAKAGE_TOKENS:
        self.assertNotIn(token, feature_keys)

  def test_freeze_is_deterministic_and_predictions_join_candidates(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      first = freeze_predictions(REPO, out=td / "a")
      second = freeze_predictions(REPO, out=td / "b")
      self.assertEqual(first["predictions_sha256"], second["predictions_sha256"])
      self.assertTrue(first["leakage_audit"]["leak_free"])
      preds = [json.loads(line) for line in (td / "a" / "predictions.jsonl").read_text().splitlines()]
      cands = [json.loads(line) for line in (td / "a" / "candidates.jsonl").read_text().splitlines()]
      self.assertEqual({p["id"] for p in preds}, {c["id"] for c in cands})
      self.assertEqual(first["predictions_sha256"], hashlib.sha256((td / "a" / "predictions.jsonl").read_bytes()).hexdigest())

  def test_dead_branch_metric_counts_experiments_to_first_live(self):
    holdout = [{"id": "a", "label": "reject"}, {"id": "b", "label": "raw_accept_unconfirmed"}, {"id": "c", "label": "tie"}]
    dead_first = dead_branch_metric(holdout, [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}, {"id": "c", "score": 0.1}])
    self.assertEqual(dead_first["experiments_to_first_live"], 1)
    self.assertEqual(dead_first["live_candidates"], 1)
    live_first = dead_branch_metric(holdout, [{"id": "b", "score": 0.9}, {"id": "a", "score": 0.5}, {"id": "c", "score": 0.1}])
    self.assertEqual(live_first["experiments_to_first_live"], 0)
    no_live = dead_branch_metric([{"id": "a", "label": "reject"}], [{"id": "a", "score": 0.9}])
    self.assertIsNone(no_live["experiments_to_first_live"])

  def test_committed_freeze_predates_outcomes_and_scores_three_baselines(self):
    # Runs against committed shadow-v0 artifacts (available after the score step / Commit B).
    if not (SHADOW / "summary.json").exists():
      self.skipTest("shadow-v0 not scored yet")
    freeze = json.loads((SHADOW / "freeze.json").read_text())
    self.assertEqual(freeze["predictions_sha256"], hashlib.sha256((SHADOW / "predictions.jsonl").read_bytes()).hexdigest())
    self.assertTrue(freeze["leakage_audit"]["leak_free"])
    preds = [json.loads(line) for line in (SHADOW / "predictions.jsonl").read_text().splitlines()]
    outcomes = [json.loads(line) for line in (SHADOW / "outcomes.jsonl").read_text().splitlines()]
    candidates = [json.loads(line) for line in (SHADOW / "candidates.jsonl").read_text().splitlines()]
    ids = {p["id"] for p in preds}
    self.assertEqual(ids, {c["id"] for c in candidates})
    self.assertEqual(ids, {o["id"] for o in outcomes})
    summary = json.loads((SHADOW / "summary.json").read_text())
    self.assertIn("xgboost", summary["model_metrics"])
    for baseline in ("mechanism_prior", "simple_family_heuristic", "reject_all"):
      self.assertIn(baseline, summary["baseline_metrics"])


if __name__ == "__main__":
  unittest.main()
