import unittest

from extra.qk.amd_dynamic_tile_owner_validation import run


class TestAmdDynamicTileOwnerValidation(unittest.TestCase):
  def test_probe_records_a_bounded_result(self):
    result = run()
    self.assertEqual(result["target"], "gfx1100")
    self.assertEqual(result["tile_count"], 1)
    self.assertIn(result["classification"], ("passed", "pre_compiler_graph_failure", "INDEX_or_dynamic_store_unsupported"))
    self.assertIn("compiled_programs", result) if result["compile"] == "passed" else None
    if result["compile"] == "passed":
      self.assertGreaterEqual(result["compiled_programs"], 1)
      self.assertGreaterEqual(result["kernel_count_delta"], 1)
    if result["compile"] != "passed": self.assertTrue(result["exact_failure"])


if __name__ == "__main__": unittest.main()
