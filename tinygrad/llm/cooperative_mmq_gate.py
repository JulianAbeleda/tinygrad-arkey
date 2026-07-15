"""Fail-closed integration glue for a cooperative MMQ research candidate.

This module deliberately does not import an emitter, atom, or route registry.
It only admits an already-produced candidate/evidence bundle.  The caller can
use the returned decision to bind an experimental route; absent every gate,
the result is the ordinary ``direct_packed`` rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Mapping

COOPERATIVE_MMQ_SCHEMA = "tinygrad.cooperative_mmq_gate.v1"
COOPERATIVE_MMQ_ROUTE = "cooperative_mmq_research"
ROLLBACK_ROUTE = "direct_packed"


def canonical_candidate_identity(candidate: Mapping[str, Any]) -> str:
  """Return an identity bound to the complete candidate payload."""
  encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
  return "coop-mmq-" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CooperativeMMQDecision:
  status: str
  route: str
  rollback_route: str = ROLLBACK_ROUTE
  candidate_identity: str | None = None
  blockers: tuple[str, ...] = ()

  @property
  def admitted(self) -> bool:
    return self.status == "admitted"

  def to_json(self) -> dict[str, Any]:
    return {"schema": COOPERATIVE_MMQ_SCHEMA, "status": self.status, "route": self.route,
            "rollback_route": self.rollback_route, "candidate_identity": self.candidate_identity,
            "blockers": list(self.blockers)}


def _passed(evidence: Mapping[str, Any], name: str) -> bool:
  value = evidence.get(name)
  return value is True or isinstance(value, Mapping) and value.get("passed") is True


def admit_cooperative_mmq(*, candidate: Mapping[str, Any] | None,
                         evidence: Mapping[str, Any] | None,
                         enabled: bool | None = None) -> CooperativeMMQDecision:
  """Admit only a complete, identity-bound, correctness-gated candidate.

  ``enabled`` is explicit for tests; production callers default to the
  opt-in ``PREFILL_COOPERATIVE_MMQ=1`` switch.  This function never selects a
  fallback candidate on behalf of the caller.
  """
  if enabled is None:
    enabled = os.environ.get("PREFILL_COOPERATIVE_MMQ", "0").strip().lower() in {"1", "true", "yes", "on"}
  if not enabled:
    return CooperativeMMQDecision("default_off", ROLLBACK_ROUTE, blockers=("cooperative MMQ is opt-in",))
  if candidate is None:
    return CooperativeMMQDecision("blocked", ROLLBACK_ROUTE, blockers=("candidate payload unavailable",))
  if evidence is None:
    return CooperativeMMQDecision("blocked", ROLLBACK_ROUTE, blockers=("evidence unavailable",))

  identity = canonical_candidate_identity(candidate)
  blockers: list[str] = []
  if evidence.get("candidate_identity") != identity:
    blockers.append("candidate identity mismatch")
  if candidate.get("rollback_route") != ROLLBACK_ROUTE:
    blockers.append("candidate rollback route must be direct_packed")
  if candidate.get("provenance") not in ("research", "machine_authored_generated"):
    blockers.append("candidate provenance is not research/generated")
  if not _passed(evidence, "compile"):
    blockers.append("compile gate not passed")
  if not _passed(evidence, "correctness"):
    blockers.append("full-output correctness gate not passed")
  if not _passed(evidence, "guard"):
    blockers.append("guard gate not passed")
  if not _passed(evidence, "resources"):
    blockers.append("resource gate not passed")
  if evidence.get("fallback_used", False) is not False:
    blockers.append("fallback_used must be explicitly false")
  if blockers:
    return CooperativeMMQDecision("blocked", ROLLBACK_ROUTE, ROLLBACK_ROUTE, identity, tuple(blockers))
  return CooperativeMMQDecision("admitted", COOPERATIVE_MMQ_ROUTE, ROLLBACK_ROUTE, identity)


__all__ = ["COOPERATIVE_MMQ_ROUTE", "ROLLBACK_ROUTE", "CooperativeMMQDecision",
           "admit_cooperative_mmq", "canonical_candidate_identity"]
