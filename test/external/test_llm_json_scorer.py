import argparse, json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_eval_common import quality_summary, read_prompt_jsonl, score_prompt
from extra.llm_json_scorer import score_expected_json, summarize_json_axes, wilson_interval
from extra.llm_rollout import summarize_rollouts, summary_markdown
from extra.llm_rollout_compare import build_report, report_markdown

def _json_row(row_id:str, expected:dict, text:str, *, tags=None):
  return {
    "id": row_id,
    "tags": ["json"] if tags is None else tags,
    "text": text,
    "tokens": [1],
    "generated": 1,
    "elapsed_s": 1.0,
    "tok_s": 1.0,
    "score": score_prompt({"expected_json": expected}, text),
  }

def _write_artifact(path:pathlib.Path, rows:list[dict], *, mode:str="generated"):
  path.mkdir(parents=True)
  summary = {
    "kind": "llm_rollout_summary",
    "mode": mode,
    "model": "model.gguf",
    "policy": None,
    "dataset": "dataset.jsonl",
    "storage": "shared",
    "prompt_format": "chat",
    "temperature": 0.0,
    "seed": 1,
    "prompts": len(rows),
    "generated": sum(row["generated"] for row in rows),
    "elapsed_s": sum(row["elapsed_s"] for row in rows),
    "tok_s": 1.0,
    "quality": quality_summary(rows),
  }
  (path / "summary.json").write_text(json.dumps(summary, sort_keys=True))
  (path / "rollouts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

class TestLLMJsonScorer(unittest.TestCase):
  def test_strict_json_axes_pass(self):
    score = score_expected_json(' {"answer":"42"}\n', {"answer": "42"})
    self.assertTrue(score["passed"])
    self.assertTrue(all(score["axes"].values()))

  def test_extra_text_is_separate_axis(self):
    score = score_expected_json('{"answer":"42"} trailing', {"answer": "42"})
    self.assertTrue(score["axes"]["parse_valid"])
    self.assertFalse(score["axes"]["no_extra_text"])
    self.assertTrue(score["axes"]["schema_ok"])
    self.assertFalse(score["passed"])

  def test_prefix_prose_fails_parse(self):
    score = score_expected_json('Answer: {"answer":"42"}', {"answer": "42"})
    self.assertFalse(score["axes"]["parse_valid"])
    self.assertFalse(score["axes"]["strict_pass"])

  def test_schema_type_and_value_failures_are_distinct(self):
    extra = score_expected_json('{"answer":"42","extra":"x"}', {"answer": "42"})
    self.assertFalse(extra["axes"]["schema_ok"])

    wrong_type = score_expected_json('{"answer":42}', {"answer": "42"})
    self.assertTrue(wrong_type["axes"]["schema_ok"])
    self.assertFalse(wrong_type["axes"]["type_ok"])

    wrong_value = score_expected_json('{"answer":"41"}', {"answer": "42"})
    self.assertTrue(wrong_value["axes"]["type_ok"])
    self.assertFalse(wrong_value["axes"]["value_correct"])

  def test_case_insensitive_value_match_is_explicit(self):
    self.assertFalse(score_expected_json('{"answer":"Purple"}', {"answer": "purple"})["passed"])
    self.assertTrue(score_expected_json('{"answer":"Purple"}', {"answer": "purple"}, case_insensitive=True)["passed"])

  def test_duplicate_keys_fail_schema(self):
    score = score_expected_json('{"answer":"bad","answer":"42"}', {"answer": "42"})
    self.assertTrue(score["axes"]["parse_valid"])
    self.assertFalse(score["axes"]["schema_ok"])
    self.assertFalse(score["passed"])

  def test_wilson_interval_bounds(self):
    empty = wilson_interval(0, 0)
    self.assertIsNone(empty["low"])
    mid = wilson_interval(50, 100)
    self.assertLess(mid["low"], 0.5)
    self.assertGreater(mid["high"], 0.5)
    with self.assertRaises(ValueError):
      wilson_interval(2, 1)

  def test_quality_summary_keeps_old_shape_without_json_axes(self):
    rows = [{"id": "a", "score": {"status": "pass", "passed": True, "checks": []}, "tags": ["x"]}]
    summary = quality_summary(rows)
    self.assertNotIn("ci95", summary)
    self.assertNotIn("json_axes", summary)

  def test_score_prompt_records_json_axes(self):
    score = score_prompt({"expected_json": {"answer": "42"}}, '{"answer":"42"}')
    self.assertEqual(score["status"], "pass")
    self.assertIn("json_axes", score)
    self.assertTrue(score["json_axes"]["axes"]["strict_pass"])

  def test_prompt_reader_validates_case_insensitive_bool(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "prompts.jsonl"
      path.write_text('{"id":"a","prompt":"p","expected_json":{"answer":"x"},"case_insensitive":"yes"}\n')
      with self.assertRaisesRegex(ValueError, "case_insensitive must be a boolean"):
        read_prompt_jsonl(path)

  def test_json_axis_summary_counts_each_axis(self):
    scores = [
      score_expected_json('{"answer":"42"}', {"answer": "42"}),
      score_expected_json('{"answer":"41"}', {"answer": "42"}),
    ]
    summary = summarize_json_axes(scores)
    self.assertEqual(summary["axes"]["parse_valid"]["passed"], 2)
    self.assertEqual(summary["axes"]["value_correct"]["passed"], 1)
    self.assertEqual(summary["axes"]["strict_pass"]["passed"], 1)

  def test_rollout_markdown_includes_json_axes_for_new_rows(self):
    args = argparse.Namespace(mode="generated", model="model.gguf", policy=None, dataset="dataset.jsonl",
                              storage="shared", prompt_format="chat", temperature=0.0, seed=1)
    rows = [_json_row("a", {"answer": "42"}, '{"answer":"42"}')]
    summary = summarize_rollouts(args, rows)
    self.assertIn("json_axes", summary["quality"])
    self.assertIn("JSON Quality Axes", summary_markdown(summary, rows))

  def test_compare_report_includes_json_axis_delta_for_new_artifacts(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      base, cand = root / "base", root / "candidate"
      _write_artifact(base, [_json_row("a", {"answer": "42"}, '{"answer":"41"}')])
      _write_artifact(cand, [_json_row("a", {"answer": "42"}, '{"answer":"42"}')], mode="explicit")
      report = build_report([base, cand])
      comp = report["comparisons"][0]
      self.assertEqual(comp["json_axis_delta"]["strict_pass"]["passed_delta"], 1)
      self.assertIn("JSON axis", report_markdown(report))

if __name__ == "__main__":
  unittest.main()
