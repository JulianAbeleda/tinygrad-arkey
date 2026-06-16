import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_adapter_json_data_v4 import CATEGORIES, build_rows, validate_dataset, validate_dataset_dir, write_dataset
from extra.llm_eval_common import read_prompt_jsonl, score_prompt

class TestLLMAdapterJsonDataV4(unittest.TestCase):
  def test_v4_dataset_writes_balanced_disjoint_rows(self):
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td)
      summary = write_dataset(out, train_per_category=3, eval_per_category=2)
      self.assertEqual(summary["kind"], "llm_adapter_json_dataset_v4")
      self.assertEqual(summary["train_rows"], 3 * len(CATEGORIES))
      self.assertEqual(summary["eval_rows"], 2 * len(CATEGORIES))
      self.assertTrue(summary["integrity"]["scorer_compatible"])
      self.assertEqual(summary["integrity"]["train_eval_answer_overlap"], 0)
      self.assertEqual(summary["integrity"]["train_eval_prompt_overlap"], 0)
      self.assertEqual(summary["integrity"]["train_eval_template_instance_overlap"], 0)
      for category in CATEGORIES:
        self.assertEqual(summary["categories"][category]["train_rows"], 3)
        self.assertEqual(summary["categories"][category]["eval_rows"], 2)

      sft_rows = [json.loads(line) for line in (out / "sft.jsonl").read_text().splitlines()]
      eval_rows = read_prompt_jsonl(out / "eval-prompts.jsonl")
      by_id = {row["id"]: row for row in sft_rows}
      self.assertEqual({row["split"] for row in sft_rows}, {"train", "eval"})
      for row in eval_rows:
        self.assertEqual(row["split"], "eval")
        self.assertEqual(json.loads(by_id[row["id"]]["completion"]), row["expected_json"])
        self.assertEqual(score_prompt(row, by_id[row["id"]]["completion"])["status"], "pass")
      self.assertEqual(validate_dataset_dir(out)["train_eval_answer_overlap"], 0)

  def test_v4_validation_rejects_answer_leakage(self):
    sft_rows, eval_rows = build_rows(train_per_category=2, eval_per_category=1)
    sft_rows[0]["normalized_answer"] = eval_rows[0]["normalized_answer"]
    with self.assertRaisesRegex(ValueError, "answer overlap"):
      validate_dataset(sft_rows, eval_rows)

  def test_v4_validation_rejects_template_instance_leakage(self):
    sft_rows, eval_rows = build_rows(train_per_category=2, eval_per_category=1)
    sft_rows[0]["template_instance"] = eval_rows[0]["template_instance"]
    with self.assertRaisesRegex(ValueError, "template_instance overlap"):
      validate_dataset(sft_rows, eval_rows)

  def test_v4_validation_rejects_malformed_eval_schema(self):
    sft_rows, eval_rows = build_rows(train_per_category=2, eval_per_category=1)
    eval_rows[0]["expected_json"] = {"answer": eval_rows[0]["normalized_answer"], "extra": "bad"}
    with self.assertRaisesRegex(ValueError, "expected_json must be exactly one answer key"):
      validate_dataset(sft_rows, eval_rows)

  def test_committed_v4_dataset_if_present(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = repo / "bench/qwen-adapter-20260613/training-data-v4"
    if not (out / "summary.json").exists():
      self.skipTest("committed bench artifact absent (gitignored post-prune); regenerate to re-lock")
    summary = json.loads((out / "summary.json").read_text())
    self.assertEqual(summary["kind"], "llm_adapter_json_dataset_v4")
    self.assertGreaterEqual(summary["eval_rows"], 198)
    self.assertEqual(summary["integrity"]["train_eval_answer_overlap"], 0)
    self.assertEqual(summary["integrity"]["train_eval_prompt_overlap"], 0)
    self.assertEqual(summary["integrity"]["train_eval_template_instance_overlap"], 0)
    self.assertEqual(validate_dataset_dir(out), summary["integrity"])

if __name__ == "__main__":
  unittest.main()
