#!/usr/bin/env python3
"""Artifact contract-field relationships (audit C5).

There are THREE distinct field lists, deliberately kept separate because they describe different objects:

  A. extra/qk_harness_contract.py:CONTRACT_FIELDS (13) -- the abstract per-HARNESS contract (the "Harnesses Are
     Performance Primitives" 13 fields); contract_audit() scores an ab_script harness artifact against it.
  B. bench/qk-decode-eval/schema.json:required (the decode_eval_run_v1 artifact) -- the AUTHORITATIVE concrete keys
     the evaluator's own run artifact must carry.
  C. bench/qk-lifecycle-search/evaluator_contract.json:required_artifact_fields -- a human-facing CURATED SUBSET of B.

This test pins those relationships so the three cannot silently drift into disagreement:
  - C is a strict subset of B (the contract doc can't require a run field the schema doesn't);
  - A is the separate 13-field harness contract and uses a DISJOINT vocabulary from B (different concept, not a fork).
No GPU / no tinygrad import.

Run: PYTHONPATH=. python -m unittest test.unit.test_artifact_contract_fields -v
"""
from __future__ import annotations
import json
import pathlib
import unittest

from extra.qk_harness_contract import CONTRACT_FIELDS

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "bench/qk-decode-eval/schema.json"
EVAL_CONTRACT = ROOT / "bench/qk-lifecycle-search/evaluator_contract.json"


class TestArtifactContractFields(unittest.TestCase):
  def setUp(self):
    self.run_required = set(json.loads(SCHEMA.read_text())["required"])                      # B
    self.contract_required = set(json.loads(EVAL_CONTRACT.read_text())["required_artifact_fields"])  # C

  def test_evaluator_contract_is_subset_of_run_schema(self):
    extra = self.contract_required - self.run_required
    self.assertEqual(extra, set(), f"evaluator_contract.required_artifact_fields lists fields not required by the "
                                   f"decode_eval_run_v1 schema (C must be a subset of B): {sorted(extra)}")

  def test_harness_contract_is_the_distinct_13_field_list(self):
    self.assertEqual(len(CONTRACT_FIELDS), 13, "the per-harness CONTRACT_FIELDS must stay the 13-field contract")
    self.assertEqual(len(set(CONTRACT_FIELDS)), 13, "CONTRACT_FIELDS has duplicates")

  def test_harness_contract_vocabulary_is_disjoint_from_run_schema(self):
    # A and B are different concepts (abstract harness fields vs concrete run keys); they must not partially merge.
    overlap = set(CONTRACT_FIELDS) & self.run_required
    self.assertEqual(overlap, set(), f"per-harness CONTRACT_FIELDS (A) and run-schema keys (B) overlap, which blurs "
                                     f"two distinct contracts: {sorted(overlap)}")


if __name__ == "__main__":
  unittest.main()
