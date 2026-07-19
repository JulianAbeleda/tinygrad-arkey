"""Bind attn_qo composition and transition safety to production selection.

This is deliberately a small adapter, not a candidate catalog.  It converts
the existing content-addressed classifications into the generic
candidate-bound PASS/BLOCKED record consumed by memory_adaptive_policy.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from extra.qk.memory_adaptive_policy import (
  build_production_eligibility, build_production_eligibility_requirement,
)
from extra.qk.mmq_attn_qo_c6_binding import COMPOSITION_SCHEMA
from extra.qk.mmq_staged_transition_safety import (
  SCHEMA as TRANSITION_SAFETY_SCHEMA,
  validate_staged_transition_safety_classification,
)


AUTHORITY_SCHEMA = "tinygrad.mmq_q4k_q8_1.attn_qo_production_authority.v1"


def _identity(value: Mapping[str, Any]) -> str:
  encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_composition(value: Any) -> dict[str, Any]:
  if not isinstance(value, Mapping) or value.get("schema") != COMPOSITION_SCHEMA or value.get("status") != "PASS":
    raise ValueError("attn_qo composition must be a mapping with exact PASS schema")
  identity = value.get("evidence_identity")
  payload = {key: item for key, item in value.items() if key != "evidence_identity"}
  try: expected_identity = _identity(payload)
  except (TypeError, ValueError) as exc:
    raise ValueError("attn_qo composition is not canonical JSON") from exc
  if identity != expected_identity:
    raise ValueError("attn_qo composition evidence identity mismatch")
  eligible, blocker = value.get("promotion_eligible_on_candidate_win"), value.get("promotion_blocker")
  if not isinstance(eligible, bool):
    raise ValueError("attn_qo composition promotion classification must be Boolean")
  if eligible and blocker is not None:
    raise ValueError("promotable attn_qo composition cannot carry a blocker")
  if not eligible and (not isinstance(blocker, str) or not blocker):
    raise ValueError("blocked attn_qo composition requires a non-empty blocker")
  return dict(value)


def _validate_transition_relation(composition: Mapping[str, Any],
                                  transition_safety: Mapping[str, Any]) -> dict[str, Any]:
  try: transition = validate_staged_transition_safety_classification(transition_safety)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"attn_qo transition safety classification is invalid: {exc}") from exc
  c6 = composition.get("c6_correctness_evidence")
  if not isinstance(c6, Mapping) or \
     transition["c6_evidence_identity"] != c6.get("evidence_identity") or \
     transition["family_identity"] != composition.get("family_identity") or \
     transition["candidate_executable_identity"] != c6.get("candidate_executable_identity"):
    raise ValueError("attn_qo transition safety classification does not bind the composition")
  return transition


def attn_qo_eligibility_requirement(composition: Mapping[str, Any],
                                    transition_safety: Mapping[str, Any]) -> dict[str, Any]:
  """Build the immutable requirement for exact composition and transition facts."""
  row = _validate_composition(composition)
  transition = _validate_transition_relation(row, transition_safety)
  return build_production_eligibility_requirement(authority={
    "schema": AUTHORITY_SCHEMA,
    "composition_schema": COMPOSITION_SCHEMA,
    "composition_evidence_identity": row["evidence_identity"],
    "transition_safety_schema": TRANSITION_SAFETY_SCHEMA,
    "transition_safety_evidence_identity": transition["evidence_identity"],
  })


def classify_attn_qo_production_eligibility(*, candidate: Mapping[str, Any],
                                            composition: Mapping[str, Any],
                                            transition_safety: Mapping[str, Any]) -> dict[str, Any]:
  """Return the conjunction of composition and transition promotion classes."""
  row = _validate_composition(composition)
  transition = _validate_transition_relation(row, transition_safety)
  requirement = attn_qo_eligibility_requirement(row, transition)
  if candidate.get("production_eligibility_requirement") != requirement:
    raise ValueError("attn_qo candidate requirement does not bind the exact safety evidence")
  eligible = row["promotion_eligible_on_candidate_win"] and transition["promotion_eligible"]
  blockers = []
  if not row["promotion_eligible_on_candidate_win"]: blockers.append(row["promotion_blocker"])
  if not transition["promotion_eligible"]:
    blockers.append(
      f"transition safety classified candidate "
      f"{transition['decision']['candidate_disposition']}")
  return build_production_eligibility(
    candidate=candidate,
    promotion_eligible=eligible,
    authority=requirement["authority"],
    blocker=None if eligible else "; ".join(blockers),
  )


__all__ = [
  "AUTHORITY_SCHEMA", "attn_qo_eligibility_requirement",
  "classify_attn_qo_production_eligibility",
]
