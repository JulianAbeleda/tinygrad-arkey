from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from extra.qk.memory_adaptive_policy import select_policy
from extra.qk.mmq_attn_qo_policy_adapter import (
  attn_qo_eligibility_requirement, classify_attn_qo_production_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
COMPOSITION = ROOT / (
  "docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/evidence/"
  "qk-attn-qo-f0d7a09ce-c6-composition.json"
)
TRANSITION_SAFETY = ROOT / (
  "docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/evidence/"
  "qk-attn-qo-f0d7a09ce-transition-safety-classification.json"
)


def _proof(speed: float, **extra):
  return {
    "correctness": {"status": "PASS"},
    "resource": {"status": "PASS"},
    "gpu_health": {"status": "PASS"},
    "route_census": {"status": "PASS", "complete": True},
    "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [speed] * 3},
    **extra,
  }


def _rehash(row):
  payload = {key: value for key, value in row.items() if key != "evidence_identity"}
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  row["evidence_identity"] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_current_qo_composition_is_candidate_bound_blocked_and_cannot_win_ranking():
  composition = json.loads(COMPOSITION.read_text())
  transition = json.loads(TRANSITION_SAFETY.read_text())
  baseline = {"candidate_id": "direct", "strategy": "DIRECT_PACKED_FALLBACK"}
  qo = {
    "candidate_id": "attn_qo_staged",
    "strategy": "BOUNDED_PACKED_TILES",
    "production_eligibility_requirement": attn_qo_eligibility_requirement(composition, transition),
  }
  gate = classify_attn_qo_production_eligibility(
    candidate=qo, composition=composition, transition_safety=transition)
  assert gate["status"] == "BLOCKED"
  assert gate["promotion_eligible"] is False
  assert "cross-route numerical parity is not certified" in gate["blocker"]
  assert "transition safety classified candidate DISQUALIFIED" in gate["blocker"]
  assert gate["authority"]["transition_safety_evidence_identity"] == transition["evidence_identity"]

  result = select_policy(
    gpu_facts={"backend": "AMD"}, model_facts={"content_hash": "model"},
    workload={"prefill": 512}, candidates=(baseline, qo),
    compiler_runtime_revision={"git": "test"},
    evidence={"direct": _proof(100), "attn_qo_staged": _proof(1000, production_eligibility=gate)},
    baseline_candidate_id="direct",
  )
  assert result["selected_candidate_id"] == "direct"
  rejected = next(row for row in result["rejected_candidates"] if row["candidate_id"] == "attn_qo_staged")
  assert any("production eligibility is BLOCKED" in reason for reason in rejected["reasons"])


def test_disqualified_transition_blocks_even_if_composition_becomes_promotable():
  composition = json.loads(COMPOSITION.read_text())
  transition = json.loads(TRANSITION_SAFETY.read_text())
  composition["promotion_eligible_on_candidate_win"] = True
  composition["promotion_blocker"] = None
  _rehash(composition)
  candidate = {
    "candidate_id": "future_attn_qo",
    "strategy": "BOUNDED_PACKED_TILES",
    "production_eligibility_requirement": attn_qo_eligibility_requirement(composition, transition),
  }
  gate = classify_attn_qo_production_eligibility(
    candidate=candidate, composition=composition, transition_safety=transition)
  assert gate["status"] == "BLOCKED" and gate["promotion_eligible"] is False
  assert gate["blocker"] == "transition safety classified candidate DISQUALIFIED"


def test_qo_adapter_rejects_composition_or_candidate_identity_drift():
  composition = json.loads(COMPOSITION.read_text())
  transition = json.loads(TRANSITION_SAFETY.read_text())
  candidate = {
    "candidate_id": "attn_qo_staged",
    "strategy": "BOUNDED_PACKED_TILES",
    "production_eligibility_requirement": attn_qo_eligibility_requirement(composition, transition),
  }
  drifted = copy.deepcopy(composition)
  drifted["promotion_eligible_on_candidate_win"] = True
  with pytest.raises(ValueError, match="identity mismatch"):
    classify_attn_qo_production_eligibility(
      candidate=candidate, composition=drifted, transition_safety=transition)
  drifted_transition = copy.deepcopy(transition)
  drifted_transition["promotion_eligible"] = True
  with pytest.raises(ValueError, match="transition safety classification is invalid"):
    classify_attn_qo_production_eligibility(
      candidate=candidate, composition=composition,
      transition_safety=drifted_transition)
  forged = copy.deepcopy(candidate)
  forged["production_eligibility_requirement"]["authority"]["composition_evidence_identity"] = \
    "sha256:" + "0" * 64
  with pytest.raises(ValueError, match="does not bind"):
    classify_attn_qo_production_eligibility(
      candidate=forged, composition=composition, transition_safety=transition)
