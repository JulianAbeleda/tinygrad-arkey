import json
import tempfile
import unittest
from pathlib import Path

from extra.llm_research.prefill.direct_packed_m4_search import (EXPORT_DIR, ROUTES, TARGET, blocked_record, check_exports,
  export, request, validate)


class TestDirectPackedM4Search(unittest.TestCase):
  def test_requests_cover_exact_8b_and_14b_shape_keys_without_a_fake_space(self):
    for route_id in ROUTES:
      record = request(route_id)
      validate(record)
      self.assertEqual(len(record["workloads"]), 8)
      self.assertEqual({row["model_size"] for row in record["workloads"]}, {"8B", "14B"})
      self.assertEqual({row["role"] for row in record["workloads"]}, {"attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"})
      self.assertEqual(record["candidate_space_status"], "MISSING")
      self.assertEqual(record["target"], TARGET)
      self.assertEqual(set(record["target"]), {"backend", "architecture", "wave_size"})
      self.assertEqual(len({row["shape_key"] for row in record["workloads"]}), 8)
      self.assertEqual(record["search_system"]["revision"], "f6ee2763f47316112fbba40b91b859e0e7068a6d")
      self.assertEqual(len(record["request_sha256"]), 64)
      self.assertNotIn("finite_space", record)
      self.assertIs(record["default"], False)
      self.assertIs(record["catalog_entry"], False)


  def test_each_route_has_a_separate_deterministic_blocked_unproven_record(self):
    with tempfile.TemporaryDirectory() as tmp:
      first = export(tmp)
      contents = {path.name: path.read_bytes() for path in first}
      second = export(tmp)
      self.assertEqual([path.name for path in first], [path.name for path in second])
      self.assertEqual(contents, {path.name: path.read_bytes() for path in second})
      for path in first:
        record = json.loads(path.read_text())
        validate(record)
        self.assertEqual((record["status"], record["verdict"]), ("BLOCKED", "UNPROVEN"))
        self.assertEqual(set(record["missing"]), {"grammar", "primitive_lowerers", "ranker", "runner", "fixtures", "execution_inputs"})

  def test_checked_in_exports_are_current_and_weight_applicability_is_explicit(self):
    check_exports()
    rows = request("prefill_q6k_direct_generated")["workloads"]
    fourteen = [row for row in rows if row["model_size"] == "14B"]
    self.assertEqual({row["role"] for row in fourteen if row["weight_applicability"]["status"] == "PRESENT"}, {"attn_kv", "ffn_down"})
    self.assertTrue(all(row["weight_applicability"]["status"] == "UNVERIFIED" for row in rows if row["model_size"] == "8B"))

  def test_contract_source_has_no_incumbent_or_catalog_references(self):
    source = Path(__file__).resolve().parents[2] / "extra/llm_research/prefill/direct_packed_m4_search.py"
    text = source.read_text()
    for forbidden in ("_direct_packed_opts", "emit_q4k_packed_prefill_kernel", "emit_q6k_packed_prefill_kernel"):
      self.assertNotIn(forbidden, text)


  def test_validator_rejects_catalog_promotion_and_incumbent_option_space(self):
    record = blocked_record("prefill_q4k_direct_tile4x4_default")
    record["catalog_entry"] = True
    with self.assertRaisesRegex(ValueError, "catalog/default"):
      validate(record)
    record = request("prefill_q6k_direct_generated")
    record["finite_space"] = {"opts": ["LOCAL:0:16"]}
    with self.assertRaisesRegex(ValueError, "finite candidate"):
      validate(record)
