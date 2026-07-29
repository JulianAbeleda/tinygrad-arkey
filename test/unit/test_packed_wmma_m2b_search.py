import json
import tempfile
import unittest

from extra.llm_research.prefill.packed_wmma_m2b_search import (COVERED_SHAPES, EXPORT_DIR, blocked_record, check_exports,
  export, request, validate)


class TestPackedWmmaM2BSearch(unittest.TestCase):
  def test_exact_current_coverage_and_missing_space(self):
    record = request(); validate(record)
    self.assertEqual(len(record["workloads"]), 6)
    self.assertEqual(len({row["shape_key"] for row in record["workloads"]}), 6)
    self.assertEqual({row["quant"] for row in record["workloads"]}, {"Q4_K", "Q6_K"})
    self.assertEqual(record["candidate_space_status"], "MISSING")
    self.assertNotIn("finite_space", record)

  def test_blocked_record_and_checked_in_export(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = export(tmp); record = json.loads(path.read_text()); validate(record)
      self.assertEqual((record["status"], record["verdict"]), ("BLOCKED", "UNPROVEN"))
    check_exports()

  def test_no_incumbent_geometry_reference_or_catalog_promotion(self):
    record = blocked_record(); record["catalog_entry"] = True
    with self.assertRaisesRegex(ValueError, "catalog/default"): validate(record)
    source = __import__("pathlib").Path(__file__).resolve().parents[2] / "extra/llm_research/prefill/packed_wmma_m2b_search.py"
    self.assertNotIn("PACKED_WMMA_GEOM", source.read_text())
