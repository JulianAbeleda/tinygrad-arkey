"""Golden tests for deterministic flywheel scoring without stale bench artifacts."""
from __future__ import annotations

import json, unittest

from extra.llm_eval_common import quality_summary, score_prompt
from extra.llm_json_scorer import score_expected_json


FIXTURES = [
  {
    "id": "contains_regex_exact_json_pass",
    "tags": ["pass"],
    "text": '{"answer": "yes", "n": 3}',
    "prompt": {
      "expected_contains": "answer",
      "expected_regex": r'"n"\s*:\s*3',
      "expected_exact": '{"answer": "yes", "n": 3}',
      "expected_json": {"answer": "yes", "n": 3},
    },
  },
  {
    "id": "case_insensitive_json_pass",
    "tags": ["pass"],
    "text": '{"answer": "YES"}',
    "prompt": {
      "expected_json": {"answer": "yes"},
      "case_insensitive": True,
    },
  },
  {
    "id": "contains_fail",
    "tags": ["fail"],
    "text": "plain text",
    "prompt": {"expected_contains": "missing"},
  },
  {
    "id": "json_extra_text_fail",
    "tags": ["fail"],
    "text": '{"answer": "yes"} trailing',
    "prompt": {"expected_json": {"answer": "yes"}},
  },
]


def _scored_rows() -> list[dict]:
  rows = []
  for row in FIXTURES:
    scored = dict(row)
    scored["score"] = score_prompt(row["prompt"], row["text"])
    rows.append(scored)
  return rows


class TestFlywheelScorerGolden(unittest.TestCase):
  def test_quality_summary_inline_fixture(self):
    summary = quality_summary(_scored_rows())
    self.assertEqual(summary["status"], "fail")
    self.assertEqual(summary["scored"], 4)
    self.assertEqual(summary["passed"], 2)
    self.assertEqual(summary["pass_rate"], 0.5)
    self.assertEqual(summary["tags"]["pass"], {"scored": 2, "passed": 2, "pass_rate": 1.0})
    self.assertEqual(summary["tags"]["fail"], {"scored": 2, "passed": 0, "pass_rate": 0.0})
    self.assertEqual(summary["json_axes"]["scored"], 3)

  def test_score_expected_json_reproduces_committed_axes(self):
    for row in _scored_rows():
      axes = row["score"].get("json_axes")
      if not isinstance(axes, dict):
        continue
      recomputed = score_expected_json(
        row["text"],
        row["prompt"]["expected_json"],
        case_insensitive=bool(row["prompt"].get("case_insensitive", False)),
      )
      self.assertEqual(json.dumps(recomputed, sort_keys=True), json.dumps(axes, sort_keys=True), row["id"])

  def test_score_prompt_check_pass_flags_match_expected(self):
    expected = {
      "contains_regex_exact_json_pass": [True, True, True, True],
      "case_insensitive_json_pass": [True],
      "contains_fail": [False],
      "json_extra_text_fail": [False],
    }
    kinds_seen: set[str] = set()
    for row in _scored_rows():
      checks = row["score"]["checks"]
      self.assertEqual([check["passed"] for check in checks], expected[row["id"]])
      kinds_seen.update(check["kind"] for check in checks)
    self.assertEqual(kinds_seen, {"contains", "regex", "exact", "json"})


if __name__ == "__main__":
  unittest.main()
