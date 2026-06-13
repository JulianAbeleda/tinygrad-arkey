import pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_eval_harness import _json_from_output, _read_jsonl, score_prompt, summarize_results, summary_markdown


class TestQwenEvalHarness(unittest.TestCase):
  def test_read_jsonl_validates_ids(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "prompts.jsonl"
      path.write_text('{"id":"a","prompt":"hello"}\n{"id":"a","prompt":"again"}\n')
      with self.assertRaisesRegex(ValueError, "duplicate id"):
        _read_jsonl(path)

  def test_read_jsonl_validates_scoring_fields(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "prompts.jsonl"
      path.write_text('{"id":"a","prompt":"hello","expected_regex":"["}\n')
      with self.assertRaisesRegex(ValueError, "invalid expected_regex"):
        _read_jsonl(path)

      path.write_text('{"id":"a","prompt":"hello","expected_json":[]}\n')
      with self.assertRaisesRegex(ValueError, "expected_json must be a non-empty object"):
        _read_jsonl(path)

  def test_score_prompt_checks_contains_regex_and_exact(self):
    prompt = {
      "expected_contains": ["answer", "42"],
      "expected_regex": "forty|42",
      "expected_exact": "The answer is 42.",
    }
    score = score_prompt(prompt, "The answer is 42.")
    self.assertEqual(score["status"], "pass")

    score = score_prompt(prompt, "The answer is 41.")
    self.assertEqual(score["status"], "fail")

  def test_score_prompt_checks_strict_json(self):
    prompt = {"expected_json": {"answer": "42"}}
    self.assertEqual(score_prompt(prompt, '{"answer":"42"}')["status"], "pass")
    self.assertEqual(score_prompt(prompt, ' {"answer":"42"}\n')["status"], "pass")
    self.assertEqual(score_prompt(prompt, '{"answer":"41"}')["status"], "fail")
    self.assertEqual(score_prompt(prompt, '{"answer":"42","extra":"x"}')["status"], "fail")
    self.assertEqual(score_prompt(prompt, 'Answer: {"answer":"42"}')["status"], "fail")

  def test_summary_detects_token_match(self):
    score = {"status": "pass", "passed": True, "checks": [{"kind": "contains", "value": "a", "passed": True}]}
    explicit = {"generated": 2, "elapsed_s": 1.0, "tok_s": 2.0, "results": [
      {"id": "p1", "tokens": [1, 2], "generated": 2, "tok_s": 2.0, "text": "a", "score": score, "tags": ["t"]},
    ]}
    generated = {"generated": 2, "elapsed_s": 1.0, "tok_s": 2.0, "results": [
      {"id": "p1", "tokens": [1, 2], "generated": 2, "tok_s": 2.0, "text": "a", "score": score, "tags": ["t"]},
    ]}
    summary = summarize_results(explicit, generated)
    self.assertEqual(summary["status"], "pass")
    self.assertTrue(summary["tokens_match"])
    self.assertEqual(summary["quality"]["status"], "pass")
    self.assertIn("LLM Eval Harness", summary_markdown(summary, explicit, generated))

  def test_summary_fails_on_token_mismatch(self):
    explicit = {"generated": 2, "elapsed_s": 1.0, "tok_s": 2.0, "results": [
      {"id": "p1", "tokens": [1, 2], "generated": 2, "tok_s": 2.0, "text": "a"},
    ]}
    generated = {"generated": 2, "elapsed_s": 1.0, "tok_s": 2.0, "results": [
      {"id": "p1", "tokens": [1, 3], "generated": 2, "tok_s": 2.0, "text": "b"},
    ]}
    summary = summarize_results(explicit, generated)
    self.assertEqual(summary["status"], "fail")
    self.assertFalse(summary["tokens_match"])

  def test_prompt_id_mismatch_is_loud(self):
    explicit = {"generated": 1, "elapsed_s": 1.0, "tok_s": 1.0, "results": [
      {"id": "p1", "tokens": [1], "generated": 1, "tok_s": 1.0, "text": "a"},
    ]}
    generated = {"generated": 1, "elapsed_s": 1.0, "tok_s": 1.0, "results": [
      {"id": "p2", "tokens": [1], "generated": 1, "tok_s": 1.0, "text": "a"},
    ]}
    with self.assertRaisesRegex(ValueError, "prompt id mismatch"):
      summarize_results(explicit, generated)

  def test_child_json_parser_uses_eval_summary_schema(self):
    out = '\n'.join([
      '{"debug": true}',
      '{"elapsed_s": 1.0, "generated": 1, "mode": "explicit", "results": [], "tok_s": 1.0}',
    ])
    self.assertEqual(_json_from_output(out)["mode"], "explicit")

  def test_child_json_parser_rejects_trailing_dict_after_summary(self):
    out = '\n'.join([
      '{"elapsed_s": 1.0, "generated": 1, "mode": "explicit", "results": [], "tok_s": 1.0}',
      '{"debug": "late"}',
    ])
    with self.assertRaisesRegex(RuntimeError, "after eval summary"):
      _json_from_output(out)

  def test_child_json_parser_rejects_wrong_schema_dicts(self):
    with self.assertRaisesRegex(RuntimeError, "none matched the eval summary schema"):
      _json_from_output('{"debug": true}\n')


if __name__ == "__main__":
  unittest.main()
