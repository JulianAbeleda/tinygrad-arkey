import pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_dataset import write_dataset
from extra.qk_flywheel_triage_eval import evaluate_baselines, load_rollout_predictions, write_eval

class TestQKFlywheelTriageEval(unittest.TestCase):
  def test_baseline_eval_scores_holdout_rows(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    with TemporaryDirectory() as raw_td:
      data_out = pathlib.Path(raw_td) / "data"
      summary = write_dataset(repo, data_out)
      eval_summary = write_eval(data_out / "examples.jsonl", pathlib.Path(raw_td) / "eval")
      self.assertEqual(eval_summary["kind"], "qk_flywheel_triage_baseline_eval")
      self.assertEqual(eval_summary["examples"], summary["rows"])
      self.assertGreater(eval_summary["holdout_rows"], 0)
      self.assertIn("reject_all", eval_summary["baselines"])
      self.assertIn("mechanism_prior", eval_summary["baselines"])
      for row in eval_summary["baselines"].values():
        self.assertGreaterEqual(row["macro_f1"], 0.0)
        self.assertLessEqual(row["macro_f1"], 1.0)

  def test_load_rollout_predictions(self):
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td)
      (out / "rollouts.jsonl").write_text('{"id":"x","text":"{\\"label\\":\\"reject\\",\\"reason\\":\\"microbench_regression\\",\\"retry\\":false}"}\n')
      name, preds = load_rollout_predictions(f"toy={out}")
      self.assertEqual(name, "toy")
      self.assertEqual(preds[0]["label"], "reject")
      self.assertTrue(preds[0]["parse_ok"])

  def test_evaluate_baselines_requires_holdout(self):
    with self.assertRaisesRegex(ValueError, "train and holdout"):
      evaluate_baselines([{"id": "x", "split": "train", "label": "reject", "reason": "microbench_regression", "retry": False, "mechanism": "unknown", "family": "f"}])

if __name__ == "__main__":
  unittest.main()
