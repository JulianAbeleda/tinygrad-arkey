import unittest

from extra.llm_json_nearmiss_audit import choose_intervention, classify_value_miss, strip_numeric_suffix

class TestLLMJsonNearmissAudit(unittest.TestCase):
  def test_strip_numeric_suffix(self):
    self.assertEqual(strip_numeric_suffix("train_qk_gemv_005"), "train_qk_gemv")
    self.assertEqual(strip_numeric_suffix("train_qk"), "train_qk")

  def test_classify_value_miss(self):
    self.assertEqual(classify_value_miss("train_qk_gemv_005", "train_qk_gemv"), "stem_without_index")
    self.assertEqual(classify_value_miss("train_qk_gemv_005", "train_qk"), "prefix")
    self.assertEqual(classify_value_miss("train_qk_gemv_005", ""), "empty_string")
    self.assertEqual(classify_value_miss("train_qk_gemv_005", 5), "type_mismatch_value")
    self.assertEqual(classify_value_miss("train_qk_gemv_005", "other"), "wrong_string")

  def test_choose_intervention_prefers_prompt_data_fix_for_suffix_failures(self):
    report = choose_intervention([{
      "accepted_attempts": 0,
      "near_miss": 3,
      "classification": [
        {"value": "stem_without_index", "count": 2},
        {"value": "prefix", "count": 1},
      ],
    }])
    self.assertEqual(report["choice"], "prompt_data_fix")

if __name__ == "__main__":
  unittest.main()
