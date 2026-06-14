import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_dataset import LABELS, REASONS, build_examples, write_dataset

class TestQKFlywheelDataset(unittest.TestCase):
  def test_build_examples_from_repo_artifacts(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    rows = build_examples(repo)
    self.assertGreaterEqual(len(rows), 30)
    self.assertEqual(len(rows), len({row["id"] for row in rows}))
    self.assertIn("holdout", {row["split"] for row in rows})
    self.assertIn("train", {row["split"] for row in rows})
    labels = {row["label"] for row in rows}
    self.assertTrue(labels <= set(LABELS))
    self.assertIn("reject", labels)
    self.assertIn("accept", labels)
    for row in rows:
      self.assertIn(row["reason"], REASONS)
      self.assertIsInstance(row["pre_result_context"], dict)
      self.assertIsInstance(row["evidence"], dict)
      self.assertTrue(row["source_files"])

  def test_write_dataset_artifact(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td)
      summary = write_dataset(repo, out)
      self.assertEqual(summary["kind"], "qk_flywheel_kernel_triage_dataset")
      self.assertEqual(summary["rows"], summary["prompts"])
      examples = [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]
      prompts = [json.loads(line) for line in (out / "prompts.jsonl").read_text().splitlines()]
      train_prompts = [json.loads(line) for line in (out / "prompts-train.jsonl").read_text().splitlines()]
      holdout_prompts = [json.loads(line) for line in (out / "prompts-holdout.jsonl").read_text().splitlines()]
      self.assertEqual({row["id"] for row in examples}, {row["id"] for row in prompts})
      self.assertEqual(len(train_prompts), summary["train_rows"])
      self.assertEqual(len(holdout_prompts), summary["holdout_rows"])
      for prompt in prompts:
        self.assertTrue(prompt["prompt"].startswith("/no_think\n"))
        self.assertIn("Return only compact JSON", prompt["prompt"])
        self.assertEqual(prompt["max_tokens"], 64)
        self.assertEqual(set(prompt["expected_json"]), {"label", "reason", "retry"})

if __name__ == "__main__":
  unittest.main()
