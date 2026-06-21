#!/usr/bin/env python3
"""Verdict SSOT enforcement — the machine-enforced half of the verdict single-source-of-truth.

extra/qk_modes.py:Verdict is the authority for decode_eval per-run verdicts. This test fails if the JSON schema enum,
the lifecycle search_policy map, the evaluator contract, or the bench README drift from it, or if qk_decode_eval.py
emits a verdict string that is not a Verdict member. No GPU / no tinygrad import.

Run: PYTHONPATH=. python -m unittest test.unit.test_verdict_ssot -v
"""
from __future__ import annotations
import ast
import json
import pathlib
import unittest

from extra.qk_modes import Verdict, VERDICTS, VERDICT_LIFECYCLE

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "bench/qk-decode-eval/schema.json"
SEARCH_POLICY = ROOT / "bench/qk-lifecycle-search/search_policy.json"
EVAL_CONTRACT = ROOT / "bench/qk-lifecycle-search/evaluator_contract.json"
README = ROOT / "bench/qk-decode-eval/README.md"
DECODE_EVAL = ROOT / "extra/qk_decode_eval.py"
DEAD = ("FAIL_WD", "FAIL_REPRODUCIBILITY", "NEEDS_BESPOKE_TEMPLATE")


class TestVerdictSSOT(unittest.TestCase):
  def test_schema_enum_equals_ssot(self):
    enum = json.loads(SCHEMA.read_text())["properties"]["verdict"]["enum"]
    self.assertEqual(set(enum), set(VERDICTS), "schema.json verdict.enum drifted from extra/qk_modes.py:Verdict")
    self.assertEqual(len(enum), len(set(enum)), "schema enum has duplicates")

  def test_search_policy_map_equals_ssot(self):
    m = json.loads(SEARCH_POLICY.read_text())["verdict_to_lifecycle_decision"]
    self.assertEqual(set(m.keys()), set(VERDICTS), "search_policy verdict_to_lifecycle_decision keys != Verdict")
    # values must match VERDICT_LIFECYCLE verbatim (catches an NFC reword of a lifecycle string).
    # note: keys are normalized via .value because str(Verdict.X) is the enum repr, not the string value.
    self.assertEqual(dict(m), {k.value: v for k, v in VERDICT_LIFECYCLE.items()},
                     "search_policy lifecycle decision values drifted from VERDICT_LIFECYCLE")

  def test_evaluator_contract_keys_equal_ssot(self):
    keys = json.loads(EVAL_CONTRACT.read_text())["verdict_interpretation"].keys()
    self.assertEqual(set(keys), set(VERDICTS), "evaluator_contract verdict_interpretation keys != Verdict")

  def test_readme_has_no_dead_verdicts(self):
    txt = README.read_text()
    for d in DEAD:
      self.assertNotIn(d, txt, f"bench/qk-decode-eval/README.md still names the dead verdict {d}")

  def test_dead_verdicts_gone_from_run_namespace(self):
    # the dead names must not appear as live decode-eval run verdicts (schema enum / policy run-map / contract)
    sp = json.loads(SEARCH_POLICY.read_text())
    run_blob = json.dumps(json.loads(SCHEMA.read_text())["properties"]["verdict"]) + \
               json.dumps(sp["verdict_to_lifecycle_decision"]) + \
               json.dumps(json.loads(EVAL_CONTRACT.read_text())["verdict_interpretation"])
    for d in DEAD:
      self.assertNotIn(d, run_blob, f"{d} still exposed as a valid decode-eval run verdict")

  def test_classify_emits_only_ssot_verdicts(self):
    """AST scan: every verdict literal in qk_decode_eval.py is a Verdict member (catches a future hardcoded typo)."""
    tree = ast.parse(DECODE_EVAL.read_text())
    offenders = []

    def check(node):
      # a verdict written as Verdict.X -> assert X is a real member
      if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "Verdict":
        if node.attr not in Verdict.__members__:
          offenders.append(f"Verdict.{node.attr} (not a member)")
      # a verdict written as a bare string constant -> assert it is a valid verdict value
      elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in VERDICTS:
        offenders.append(f'bare string "{node.value}"')

    for n in ast.walk(tree):
      # classify()'s return tuples: (verdict, reason)
      if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple) and n.value.elts:
        first = n.value.elts[0]
        if isinstance(first, ast.Attribute) or (isinstance(first, ast.Constant) and isinstance(first.value, str)
                                                and first.value.isupper()):
          check(first)
      # res["verdict"] = X  and  {"verdict": X, ...}
      if isinstance(n, ast.Assign):
        for t in n.targets:
          if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "verdict"):
            check(n.value)
      if isinstance(n, ast.Dict):
        for k, v in zip(n.keys, n.values):
          if isinstance(k, ast.Constant) and k.value == "verdict":
            check(v)
    self.assertEqual(offenders, [], f"qk_decode_eval.py emits non-SSOT verdicts: {offenders}")


if __name__ == "__main__":
  unittest.main()
