import json, pathlib, re, unittest
from tempfile import TemporaryDirectory

from extra.llm_adapter_json_data_v4_1_compiler import DATASET_KIND, build_rows, validate_dataset, validate_dataset_dir, write_dataset
from extra.llm_eval_common import read_prompt_jsonl, score_prompt

STABLE_RE = re.compile(r"^qk_[a-z0-9_]+$")
NUMERIC_SUFFIX_RE = re.compile(r"_\d{3,}$")

class TestLLMAdapterJsonDataV41Compiler(unittest.TestCase):
  def test_v4_1_compiler_dataset_writes_stable_keys(self):
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td)
      summary = write_dataset(out, train_rows=14, eval_rows=6)
      self.assertEqual(summary["kind"], DATASET_KIND)
      self.assertEqual(summary["train_rows"], 14)
      self.assertEqual(summary["eval_rows"], 6)
      self.assertEqual(summary["categories"], {"compiler": {"train_rows": 14, "eval_rows": 6}})
      self.assertTrue(summary["integrity"]["scorer_compatible"])
      self.assertTrue(summary["integrity"]["answer_overlap_allowed"])
      self.assertGreater(summary["integrity"]["train_eval_answer_overlap"], 0)
      self.assertEqual(summary["integrity"]["train_eval_prompt_overlap"], 0)
      self.assertEqual(summary["integrity"]["train_eval_template_instance_overlap"], 0)

      sft_rows = [json.loads(line) for line in (out / "sft.jsonl").read_text().splitlines()]
      eval_rows = read_prompt_jsonl(out / "eval-prompts.jsonl")
      by_id = {row["id"]: row for row in sft_rows}
      self.assertEqual({row["category"] for row in sft_rows}, {"compiler"})
      self.assertEqual({row["split"] for row in sft_rows}, {"train", "eval"})
      for row in sft_rows:
        answer = row["expected_json"]["answer"]
        self.assertRegex(answer, STABLE_RE)
        self.assertNotRegex(answer, NUMERIC_SUFFIX_RE)
        self.assertFalse(answer.startswith(("train_", "eval_")))
        self.assertEqual(row["normalized_answer"], answer)
      for row in eval_rows:
        self.assertEqual(row["split"], "eval")
        self.assertEqual(json.loads(by_id[row["id"]]["completion"]), row["expected_json"])
        self.assertEqual(score_prompt(row, by_id[row["id"]]["completion"])["status"], "pass")
      self.assertEqual(validate_dataset_dir(out), summary["integrity"])

  def test_v4_1_validation_rejects_numeric_suffix_answers(self):
    sft_rows, eval_rows = build_rows(train_rows=2, eval_rows=1)
    sft_rows[0]["expected_json"] = {"answer": "qk_gemv_001"}
    sft_rows[0]["normalized_answer"] = "qk_gemv_001"
    sft_rows[0]["completion"] = '{"answer":"qk_gemv_001"}'
    with self.assertRaisesRegex(ValueError, "numeric suffix"):
      validate_dataset(sft_rows, eval_rows)

  def test_v4_1_validation_rejects_prompt_leakage(self):
    sft_rows, eval_rows = build_rows(train_rows=2, eval_rows=1)
    sft_rows[0]["prompt"] = eval_rows[0]["prompt"]
    with self.assertRaisesRegex(ValueError, "prompt overlap"):
      validate_dataset(sft_rows, eval_rows)

  def test_committed_v4_1_compiler_dataset_if_present(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = repo / "bench/qwen-adapter-20260613/training-data-v4_1-compiler"
    if not out.exists(): return
    summary = json.loads((out / "summary.json").read_text())
    self.assertEqual(summary["kind"], DATASET_KIND)
    self.assertGreaterEqual(summary["train_rows"], 68)
    self.assertGreaterEqual(summary["eval_rows"], 34)
    self.assertTrue(summary["integrity"]["answer_overlap_allowed"])
    self.assertEqual(summary["integrity"]["train_eval_prompt_overlap"], 0)
    self.assertEqual(summary["integrity"]["train_eval_template_instance_overlap"], 0)
    self.assertEqual(summary["integrity"]["compiler_answers_with_numeric_suffix"], 0)
    self.assertEqual(validate_dataset_dir(out), summary["integrity"])

if __name__ == "__main__":
  unittest.main()
