"""Deterministic, research-only policy artifact for the exact 14B mixed-QK rows."""
from __future__ import annotations

import hashlib, json
from collections.abc import Mapping, Sequence
from typing import Any

from extra.qk.prefill.q4k_q8_five_buffer_role_gate import SCHEMA as Q4_EVIDENCE_SCHEMA, admitted_q4k_non_fitting_roles
from extra.qk.route_manifest import canonical_policy_rows

SCHEMA = "tinygrad.qk_exact_six_row_research_policy.v1"
Q6_EVIDENCE_SCHEMA = "tinygrad.q6_direct_packed_qualification.v1"
TARGET = {"backend":"AMD", "arch":"gfx1100", "wave_size":32}
DIRECT_PACKED_ROUTE = "direct_packed"


class MissingQualificationEvidence(ValueError):
  def __init__(self, missing: Sequence[str]):
    self.missing = tuple(missing)
    super().__init__("missing canonical qualification evidence: " + ", ".join(self.missing))


def _semantic(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _identity(namespace: str, value: Any) -> str:
  return f"{namespace}:sha256:{hashlib.sha256(_semantic(value)).hexdigest()}"


def _workload(row: Mapping[str, Any]) -> dict[str, Any]:
  return {"phase":"prefill", "role":row["role"], "quant_format":row["quant_format"],
          "shape":dict(row["shape"]), "target":dict(row["target"])}


def _inventory_rows(inventory: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
  rows = inventory.get("inventory", {}).get("rows")
  if not isinstance(rows, list): raise ValueError("canonical candidate inventory has no inventory rows")
  selected = tuple(sorted(({**row, "target":dict(TARGET)} for row in rows if row.get("quant_format") in ("Q4_K", "Q6_K")),
                          key=lambda row:(row["quant_format"], row["role"], *(row["shape"][x] for x in ("m", "n", "k")))))
  counts = {quant:sum(row["quant_format"] == quant for row in selected) for quant in ("Q4_K", "Q6_K")}
  if counts != {"Q4_K":4, "Q6_K":2}: raise ValueError(f"exact six-row inventory requires four Q4_K and two Q6_K rows, got {counts!r}")
  return selected


def missing_qualification_evidence(inventory: Mapping[str, Any], *, q4_evidence: Mapping[str, Any] | None = None,
                                   q6_evidence: Sequence[Mapping[str, Any]] = ()) -> tuple[str, ...]:
  """Return every absent exact evidence identity in deterministic policy-row order."""
  rows = _inventory_rows(inventory)
  q4_expected = {admission.canonical_identity:(entry, admission) for entry, admission in admitted_q4k_non_fitting_roles(inventory)}
  q4_rows = q4_evidence.get("rows", ()) if isinstance(q4_evidence, Mapping) and \
    q4_evidence.get("schema") == Q4_EVIDENCE_SCHEMA and q4_evidence.get("passed") is True else ()
  q4_present = {row.get("canonical_identity") for row in q4_rows if isinstance(row, Mapping) and row.get("compile_status") == "pass"}
  binding_ids = {(binding["inventory_key"]["role"], binding["inventory_key"]["quant_format"]):binding["canonical_identity"]
                 for binding in inventory.get("bindings", ())}
  expected_q6 = {binding_ids[(row["role"], "Q6_K")]:_workload(row) for row in rows if row["quant_format"] == "Q6_K"}
  q6_present = {row.get("canonical_identity") for row in q6_evidence if isinstance(row, Mapping) and
    row.get("schema") == Q6_EVIDENCE_SCHEMA and row.get("status") == "qualified" and
    row.get("route_id") == DIRECT_PACKED_ROUTE and row.get("workload") == expected_q6.get(row.get("canonical_identity"))}
  missing = []
  for row in rows:
    if row["quant_format"] == "Q4_K":
      identity = next(admission.canonical_identity for _, admission in q4_expected.values()
                      if admission.normalized_payload["workload"]["role"] == row["role"])
      if identity not in q4_present: missing.append(f"Q4_K:{row['role']}:{identity}:five_buffer_role_gate")
    else:
      identity = binding_ids[(row["role"], "Q6_K")]
      if identity not in q6_present: missing.append(f"Q6_K:{row['role']}:{identity}:direct_packed_qualification")
  return tuple(missing)


def build_six_row_policy_artifact(inventory: Mapping[str, Any], *, q4_evidence: Mapping[str, Any] | None = None,
                                  q6_evidence: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
  """Build the immutable semantic artifact, or fail without partially binding rows."""
  missing = missing_qualification_evidence(inventory, q4_evidence=q4_evidence, q6_evidence=q6_evidence)
  if missing: raise MissingQualificationEvidence(missing)
  rows = _inventory_rows(inventory)
  q4_by_role = {admission.normalized_payload["workload"]["role"]:entry
                for entry, admission in admitted_q4k_non_fitting_roles(inventory)}
  q6_by_id = {row["canonical_identity"]:dict(row) for row in q6_evidence}
  bindings = {(row["inventory_key"]["role"], row["inventory_key"]["quant_format"]):row["canonical_identity"]
              for row in inventory["bindings"]}
  entries = [{"canonical_identity":q4_by_role[row["role"]].canonical_identity, "payload":q4_by_role[row["role"]].payload}
             for row in rows if row["quant_format"] == "Q4_K"]
  fallbacks = []
  for row in rows:
    if row["quant_format"] != "Q6_K": continue
    evidence = q6_by_id[bindings[(row["role"], "Q6_K")]]
    evidence_identity = _identity("fallback_evidence", evidence)
    content = {"workload":_workload(row), "route_id":DIRECT_PACKED_ROUTE, "evidence_identity":evidence_identity}
    fallbacks.append({**content, "evidence":evidence, "fallback_identity":_identity("fallback", content)})
  candidate_set = {"schema":"boltbeam.full_kernel_candidate_set.v1", "entries":entries, "fallbacks":fallbacks}
  capability = {"target":dict(TARGET), "phases":["prefill"], "quant_formats":["Q4_K", "Q6_K"]}
  exact_inventory = {**inventory["inventory"], "target":dict(TARGET)}
  policy_rows = canonical_policy_rows(exact_inventory, capability, candidate_set,
                                      route_id="q4k_q8_five_buffer_research")
  artifact = {"schema":SCHEMA, "status":"research_only", "production_promotion":False,
              "inventory_identity":inventory["inventory_identity"], "candidate_set":candidate_set,
              "policy_rows":list(policy_rows)}
  return {**artifact, "artifact_identity":_identity("qk_exact_six_row_policy", artifact)}


__all__ = ["SCHEMA", "Q6_EVIDENCE_SCHEMA", "MissingQualificationEvidence", "missing_qualification_evidence",
           "build_six_row_policy_artifact"]
