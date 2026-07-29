import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/"docs/task_workflow/output/p0-search-selected-ownership-ledger-20260729.json"

class TestP0SearchSelectedDocumentation(unittest.TestCase):
  def test_ledger_uses_practical_not_strict_definition(self):
    data=json.loads(LEDGER.read_text())
    self.assertEqual(data["schema"],"tinygrad.p0_search_selected_ownership_ledger.v1")
    self.assertIn("Human-authored",data["practical_definition"])
    self.assertIn("Tensor.custom_kernel does not imply fallback or manual selection.",data["documentation_invariants"])
    self.assertIn("Missing modern provenance does not imply not_machine_search.",data["documentation_invariants"])
  def test_benchmark_routes_are_protected_and_oracles_have_prerequisites(self):
    routes=json.loads(LEDGER.read_text())["route_ownership"]
    protected=[r for r in routes if r.get("benchmark_protected")]
    self.assertTrue(protected)
    self.assertTrue(all(r["disposition"]=="KEEP_MASTER_SEARCH_SELECTED" for r in protected))
    oracles=[r for r in routes if r["disposition"]=="MOVE_DEV_HAND_FALLBACK_ORACLE"]
    self.assertTrue(oracles and all(r.get("removal_prerequisite") for r in oracles))
