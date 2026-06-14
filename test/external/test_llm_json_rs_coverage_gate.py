import unittest

from extra.llm_json_rs_coverage_gate import build_report

def _summary():
  return {
    "kind": "llm_json_rejection_sample_summary",
    "attempts": 12,
    "accepted_attempts": 4,
    "selected_train_rows": 4,
    "sft_rows": 6,
    "categories": {
      "code": {"attempts": 4, "accepted_attempts": 2, "near_miss": 1, "selected_train_rows": 2},
      "compiler": {"attempts": 4, "accepted_attempts": 0, "near_miss": 3, "selected_train_rows": 0},
      "string": {"attempts": 4, "accepted_attempts": 2, "near_miss": 0, "selected_train_rows": 2},
    },
  }

class TestLLMJsonRSCoverageGate(unittest.TestCase):
  def test_build_report_fails_sparse_category(self):
    report = build_report(_summary(), categories=["code", "compiler", "string"], min_selected=1)
    self.assertEqual(report["status"], "fail")
    self.assertEqual(report["categories"]["compiler"]["status"], "fail")
    self.assertIn("compiler: selected_train_rows 0 < 1", report["failures"])

  def test_build_report_passes_when_all_categories_clear_threshold(self):
    summary = _summary()
    summary["categories"]["compiler"]["selected_train_rows"] = 1
    report = build_report(summary, categories=["code", "compiler", "string"], min_selected=1)
    self.assertEqual(report["status"], "pass")
    self.assertEqual(report["failures"], [])

  def test_build_report_fails_missing_category(self):
    report = build_report(_summary(), categories=["compiler", "fact"], min_selected=1)
    self.assertEqual(report["status"], "fail")
    self.assertEqual(report["categories"]["fact"]["reason"], "missing category")

if __name__ == "__main__":
  unittest.main()
