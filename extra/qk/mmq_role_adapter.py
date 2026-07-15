"""Fail-closed admission adapter for request-selected research MMQ candidates.

This is intentionally only an admission boundary.  It is not imported by the
live prefill route and cannot select a fallback when the candidate is absent
or not fully evidenced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any
from extra.qk.mmq_capability import MMQHardwareCapability, MMQRequest, GFX11_MMQ_CAPABILITY

MMQ_FFN_GATE_UP_ROUTE_ID = "prefill_q4k_q8_1_mmq_research"
MMQ_FFN_GATE_UP_ROLE = "ffn_gate_up"
MMQ_MILESTONES = tuple(f"M{i}" for i in range(1, 8))


class MMQRoleAdmissionError(ValueError):
  pass


def canonical_mmq_candidate_identity(payload: dict[str, Any]) -> str:
  """Hash the candidate payload, never an adapter-generated replacement."""
  try:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
  except (TypeError, ValueError) as exc:
    raise MMQRoleAdmissionError("candidate payload is not canonical JSON") from exc
  return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AdmittedMMQRoleCandidate:
  canonical_identity: str
  payload: dict[str, Any]
  role: str = MMQ_FFN_GATE_UP_ROLE
  route_id: str = MMQ_FFN_GATE_UP_ROUTE_ID


def admit_ffn_gate_up_mmq_candidate(candidate_payload: dict[str, Any],
                                    canonical_identity: str, *, _allow_requested_role: bool = False) -> AdmittedMMQRoleCandidate:
  """Admit one explicit, identity-qualified, fully evidenced research payload.

  No defaults are synthesized and no fallback is returned.  A caller must
  handle ``MMQRoleAdmissionError`` and retain its existing route itself.
  """
  if not isinstance(candidate_payload, dict):
    raise MMQRoleAdmissionError("candidate payload must be an object")
  if not isinstance(canonical_identity, str) or len(canonical_identity) != 64:
    raise MMQRoleAdmissionError("canonical identity must be a SHA-256 hex string")
  try:
    normalized = json.loads(json.dumps(candidate_payload, sort_keys=True, allow_nan=False))
  except (TypeError, ValueError) as exc:
    raise MMQRoleAdmissionError("candidate payload is not canonical JSON") from exc
  required = {"candidate_id", "route_id", "role", "quant_format", "activation_format", "evidence"}
  if set(normalized) != required:
    raise MMQRoleAdmissionError("candidate payload schema is incomplete or contains unknown fields")
  if normalized["route_id"] != MMQ_FFN_GATE_UP_ROUTE_ID:
    raise MMQRoleAdmissionError("candidate route is not the research MMQ route")
  if not _allow_requested_role and normalized["role"] != MMQ_FFN_GATE_UP_ROLE:
    raise MMQRoleAdmissionError("legacy ffn_gate_up adapter received another role")
  if normalized["quant_format"] != "Q4_K" or normalized["activation_format"] != "Q8_1":
    raise MMQRoleAdmissionError("candidate formats are unsupported")
  evidence = normalized["evidence"]
  if not isinstance(evidence, dict) or set(evidence) != set(MMQ_MILESTONES) or any(evidence[x] is not True for x in MMQ_MILESTONES):
    raise MMQRoleAdmissionError("M1-M7 evidence contract is incomplete")
  if canonical_identity != canonical_mmq_candidate_identity(normalized):
    raise MMQRoleAdmissionError("canonical identity does not match candidate payload")
  return AdmittedMMQRoleCandidate(canonical_identity, MappingProxyType(normalized))

def admit_mmq_candidate(candidate_payload: dict[str, Any], canonical_identity: str, *,
                        request: MMQRequest, capability: MMQHardwareCapability = GFX11_MMQ_CAPABILITY) -> AdmittedMMQRoleCandidate:
  request.validate(); capability.validate()
  if candidate_payload.get("role") != request.role: raise MMQRoleAdmissionError("candidate role does not match request")
  return admit_ffn_gate_up_mmq_candidate(candidate_payload, canonical_identity, _allow_requested_role=True)
