"""Fail-closed admission for the cooperative MMQ research candidate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

COOPERATIVE_MMQ_SCHEMA = "tinygrad.cooperative_mmq_gate.v1"
COOPERATIVE_MMQ_ROUTE = "cooperative_mmq_research"
ROLLBACK_ROUTE = "direct_packed"

def canonical_candidate_identity(candidate:Mapping[str, Any]) -> str:
  encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
  return "coop-mmq-" + hashlib.sha256(encoded).hexdigest()

@dataclass(frozen=True)
class CooperativeMMQDecision:
  status: str
  route: str
  rollback_route: str = ROLLBACK_ROUTE
  candidate_identity: str|None = None
  blockers: tuple[str, ...] = ()

  @property
  def admitted(self) -> bool: return self.status == "admitted"

  def to_json(self) -> dict[str, Any]:
    return {"schema": COOPERATIVE_MMQ_SCHEMA, "status": self.status, "route": self.route,
            "rollback_route": self.rollback_route, "candidate_identity": self.candidate_identity,
            "blockers": list(self.blockers)}

def _passed(evidence:Mapping[str, Any], name:str) -> bool:
  value = evidence.get(name)
  return value is True or isinstance(value, Mapping) and value.get("passed") is True

def admit_cooperative_mmq(*, candidate:Mapping[str, Any]|None, evidence:Mapping[str, Any]|None,
                          enabled:bool=False) -> CooperativeMMQDecision:
  if not enabled:
    return CooperativeMMQDecision("default_off", ROLLBACK_ROUTE, blockers=("cooperative MMQ is opt-in",))
  if candidate is None:
    return CooperativeMMQDecision("blocked", ROLLBACK_ROUTE, blockers=("candidate payload unavailable",))
  if evidence is None:
    return CooperativeMMQDecision("blocked", ROLLBACK_ROUTE, blockers=("evidence unavailable",))

  identity, blockers = canonical_candidate_identity(candidate), []
  if evidence.get("candidate_identity") != identity: blockers.append("candidate identity mismatch")
  if candidate.get("rollback_route") != ROLLBACK_ROUTE: blockers.append("candidate rollback route must be direct_packed")
  if candidate.get("provenance") not in ("research", "machine_authored_generated"):
    blockers.append("candidate provenance is not research/generated")
  for name, label in (("compile", "compile gate"), ("correctness", "full-output correctness gate"),
                      ("dynamic_owner_compile", "dynamic-owner compile gate"),
                      ("dynamic_owner_correctness", "dynamic-owner correctness gate"),
                      ("dynamic_owner_instruction", "dynamic-owner instruction evidence"),
                      ("guard", "guard gate"), ("resources", "resource gate")):
    if not _passed(evidence, name): blockers.append(f"{label} not passed")
  if evidence.get("fallback_used", False) is not False: blockers.append("fallback_used must be explicitly false")
  if blockers: return CooperativeMMQDecision("blocked", ROLLBACK_ROUTE, ROLLBACK_ROUTE, identity, tuple(blockers))
  return CooperativeMMQDecision("admitted", COOPERATIVE_MMQ_ROUTE, ROLLBACK_ROUTE, identity)

__all__ = ["COOPERATIVE_MMQ_ROUTE", "ROLLBACK_ROUTE", "CooperativeMMQDecision",
           "admit_cooperative_mmq", "canonical_candidate_identity"]
