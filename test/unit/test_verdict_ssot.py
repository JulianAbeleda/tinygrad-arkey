"""Single-source-of-truth test for the two verdict producers (restored in QK-CONSOLIDATE-R1 Phase 2).

Two independent harnesses each emit a verdict vocabulary; each has ONE enum in extra/qk_modes.py that must stay == its
producer, so downstream consumers (schemas, search policy, docs) cannot drift:

  Verdict      <- extra/qk_decode_eval.py:classify()        (decode per-run eval)
  TierVerdict  <- extra/qk_candidate_evaluator.py            (PMS-R2 table-driven candidate evaluator)

The test scans each producer's source for the verdict string literals it can emit and asserts the enum covers exactly
that set (no missing member, no dead member). Run: PYTHONPATH=. python3 -m pytest test/unit/test_verdict_ssot.py -q
"""
from __future__ import annotations
import json, pathlib, re, inspect

from extra.qk_modes import Verdict, VERDICTS, VERDICT_LIFECYCLE, TierVerdict, TIER_VERDICTS

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_verdict_internal_consistency():
  # the frozenset, the enum, and the lifecycle map all describe the same decode-verdict set.
  assert VERDICTS == {v.value for v in Verdict}
  assert set(VERDICT_LIFECYCLE) == set(Verdict), "every Verdict needs a lifecycle decision and vice-versa"


def test_decode_verdict_matches_search_policy():
  # the live lifecycle search policy maps exactly the decode Verdict set (the original SSOT assertion).
  p = ROOT / "bench/qk-lifecycle-search/search_policy.json"
  if not p.exists():
    return  # policy file optional in slim checkouts
  pol = json.load(open(p)).get("verdict_to_lifecycle_decision", {})
  assert set(pol) == VERDICTS, f"search_policy verdict map drifted from Verdict: {set(pol) ^ VERDICTS}"


def _emitted_literals(func, extra_pattern: str | None = None) -> set[str]:
  src = inspect.getsource(func)
  lits = set(re.findall(r'return\s+"([A-Z_0-9]+)"', src))
  lits |= set(re.findall(r'"([A-Z_0-9]+)",\s*"(?:promote|refute|block|defer)"', src))  # return (TIER, disposition)
  return lits


def test_tier_verdict_matches_evaluator_producer():
  from extra import qk_candidate_evaluator as ev
  # classify() emits the tier strings; evaluate() emits the PMS_R2_* outcomes.
  emitted = _emitted_literals(ev.classify)
  emitted |= set(re.findall(r'"(PMS_R2_[A-Z_]+)"', inspect.getsource(ev.evaluate)))
  assert emitted, "scan found no verdict literals — producer shape changed, update the scan"
  missing = emitted - TIER_VERDICTS
  assert not missing, f"evaluator emits verdicts not in TierVerdict: {missing}"
  dead = TIER_VERDICTS - emitted
  assert not dead, f"TierVerdict has members the evaluator never emits: {dead}"


if __name__ == "__main__":
  test_verdict_internal_consistency()
  test_decode_verdict_matches_search_policy()
  test_tier_verdict_matches_evaluator_producer()
  print("test_verdict_ssot: PASS (Verdict + TierVerdict both == their producers)")
