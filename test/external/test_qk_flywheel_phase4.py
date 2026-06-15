import hashlib, json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_cost_model import extract_feature_map
from extra.qk_flywheel_shadow import (
  FRESH_SPECS, LEAKAGE_TOKENS, STAGED_OUT, _role_mechanism_prior, _safe_skips, _staged_candidate_rows,
  build_fresh_candidates, dead_branch_metric, freeze_predictions, fresh_id,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SHADOW = REPO / "bench/amd-decode-flywheel-proof-20260614/shadow-v0"
STAGED = REPO / "bench/amd-decode-flywheel-proof-20260614/shadow-staged"
STAGED_V2 = REPO / "bench/amd-decode-flywheel-proof-20260614/shadow-staged-v2"


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


class TestQKFlywheelPhase41Staged(unittest.TestCase):
  def test_staged_candidates_are_diverse_leak_free_and_unlabeled(self):
    rows = _staged_candidate_rows(REPO, STAGED)
    self.assertEqual(len(rows), 16)
    self.assertEqual({r["mechanism"] for r in rows}, {"row_upcast", "direct_output", "reduce_unroll", "two_dim_local"})
    self.assertEqual({r["role"] for r in rows}, {"ffn_gate", "attn_q"})  # cross-shape variation
    for r in rows:
      self.assertEqual(r["schema_version"], "kernel_triage_v1")
      self.assertFalse(r["outcome_known"])
      feature_keys = "\n".join(extract_feature_map(r))
      for token in LEAKAGE_TOKENS:
        self.assertNotIn(token, feature_keys)

  def test_safe_skips_keeps_full_live_recall(self):
    # Live candidate at score 0.5: only candidates scored below it are safe to skip.
    res = _safe_skips([(0.9, False), (0.5, True), (0.1, False)])
    self.assertEqual(res["safe_skips"], 1)
    self.assertEqual(res["live_candidates"], 1)
    # Live candidate ranked top: everything below is safe to skip.
    self.assertEqual(_safe_skips([(0.9, True), (0.5, False), (0.1, False)])["safe_skips"], 2)
    # No live candidates: everything is skippable (and the batch is degenerate).
    none_live = _safe_skips([(0.5, False), (0.2, False)])
    self.assertEqual(none_live["safe_skips"], 2)
    self.assertEqual(none_live["live_candidates"], 0)

  def test_committed_staged_freeze_predates_outcomes_and_compares_to_run_all(self):
    if not (STAGED / "summary.json").exists():
      self.skipTest("shadow-staged not scored yet")
    freeze = json.loads((STAGED / "freeze.json").read_text())
    self.assertEqual(freeze["predictions_sha256"], hashlib.sha256((STAGED / "predictions.jsonl").read_bytes()).hexdigest())
    self.assertTrue(freeze["leakage_audit"]["leak_free"])
    preds = [json.loads(line) for line in (STAGED / "predictions.jsonl").read_text().splitlines()]
    outcomes = [json.loads(line) for line in (STAGED / "outcomes.jsonl").read_text().splitlines()]
    self.assertEqual({p["id"] for p in preds}, {o["id"] for o in outcomes})
    summary = json.loads((STAGED / "summary.json").read_text())
    for gate in ("run_all", "xgboost", "mechanism_prior"):
      self.assertIn(gate, summary["gates"])
      self.assertEqual(summary["gates"][gate]["live_recall"], 1.0)
    self.assertEqual(summary["gates"]["run_all"]["experiments_saved_vs_run_all"], 0)


class TestQKFlywheelPhase42Ablation(unittest.TestCase):
  def test_role_mechanism_prior_separates_by_role(self):
    train = [
      {"id": "t1", "role": "attn_q", "mechanism": "row_upcast", "label": "raw_accept_unconfirmed"},
      {"id": "t2", "role": "attn_q", "mechanism": "row_upcast", "label": "raw_accept_unconfirmed"},
      {"id": "t3", "role": "ffn_gate", "mechanism": "row_upcast", "label": "reject"},
    ]
    holdout = [{"id": "h1", "role": "attn_q", "mechanism": "row_upcast"},
               {"id": "h2", "role": "ffn_gate", "mechanism": "row_upcast"}]
    preds = {p["id"]: p for p in _role_mechanism_prior(train, holdout)}
    # Same mechanism, different role -> different prediction (mechanism_prior cannot do this).
    self.assertEqual(preds["h1"]["label"], "raw_accept_unconfirmed")
    self.assertEqual(preds["h2"]["label"], "reject")
    self.assertGreater(preds["h1"]["score"], preds["h2"]["score"])

  def test_committed_v2_ablation_compares_full_gate_ladder(self):
    if not (STAGED_V2 / "summary.json").exists():
      self.skipTest("shadow-staged-v2 not scored yet")
    freeze = json.loads((STAGED_V2 / "freeze.json").read_text())
    self.assertEqual(freeze["predictions_sha256"], hashlib.sha256((STAGED_V2 / "predictions.jsonl").read_bytes()).hexdigest())
    self.assertTrue(freeze["leakage_audit"]["leak_free"])
    summary = json.loads((STAGED_V2 / "summary.json").read_text())
    self.assertEqual(summary["ablation"]["ladder"], ["run_all", "mechanism_prior", "role_mechanism_prior", "xgboost"])
    self.assertIn(summary["ablation"]["simplest_sufficient_gate"],
                  ("run_all", "mechanism_prior", "role_mechanism_prior", "xgboost"))
    for gate in ("run_all", "mechanism_prior", "role_mechanism_prior", "xgboost"):
      self.assertIn(gate, summary["gates"])
      self.assertEqual(summary["gates"][gate]["live_recall"], 1.0)
    self.assertTrue(summary["per_cell"])  # per (role x mechanism) breakdown present


if __name__ == "__main__":
  unittest.main()
