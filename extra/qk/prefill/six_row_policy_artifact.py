"""Deterministic, research-only policy artifact for the exact 14B mixed-QK rows."""
from __future__ import annotations

import argparse, hashlib, json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from extra.qk.prefill.q4k_q8_five_buffer_role_gate import admitted_q4k_non_fitting_roles
from extra.qk.route_manifest import _identity as _route_identity, canonical_policy_rows

SCHEMA = "tinygrad.qk_exact_six_row_research_policy.v2"
Q4_JOINED_EVIDENCE_SCHEMA = "q4k-q8-1-mmq-machine-search.v1"
Q6_EVIDENCE_SCHEMA = "tinygrad.q6_direct_packed_qualification.v1"
TARGET = {"backend":"AMD", "arch":"gfx1100", "wave_size":32}
DIRECT_PACKED_ROUTE = "direct_packed"
Q4_CANDIDATE_ROLE = "ffn_gate_up"


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


def _expected_evidence(inventory: Mapping[str, Any]):
  rows = _inventory_rows(inventory)
  q4 = {admission.normalized_payload["workload"]["role"]:(row, entry, admission)
        for entry, admission in admitted_q4k_non_fitting_roles(inventory)
        for row in rows if row["quant_format"] == "Q4_K" and
        admission.normalized_payload["workload"]["role"] == row["role"]}
  bindings = inventory.get("bindings")
  if not isinstance(bindings, list): raise ValueError("canonical candidate inventory has no bindings")
  binding_ids = {}
  for binding in bindings:
    try: key, identity = binding["inventory_key"], binding["canonical_identity"]
    except (KeyError, TypeError): raise ValueError("canonical candidate inventory has a malformed binding") from None
    pair = (key.get("role"), key.get("quant_format"))
    if pair in binding_ids: raise ValueError(f"duplicate canonical inventory binding for {pair!r}")
    binding_ids[pair] = identity
  q6 = {binding_ids[(row["role"], "Q6_K")]:row for row in rows if row["quant_format"] == "Q6_K"}
  if len(q4) != 4 or len(q6) != 2 or Q4_CANDIDATE_ROLE not in q4:
    raise ValueError("canonical evidence obligations are not exactly four Q4_K plus two Q6_K rows")
  return rows, q4, q6


def _validated_q4_joined_evidence(evidence: Mapping[str, Any] | None,
                                  expected: Mapping[str, tuple[Any, Any, Any]]) -> dict[str, Any] | None:
  """Re-run the retained R6/R7 validators; compile-only role evidence is insufficient."""
  if not isinstance(evidence, Mapping) or evidence.get("schema") != Q4_JOINED_EVIDENCE_SCHEMA: return None
  try:
    from extra.qk.mmq_machine_search import (CANDIDATE_ROUTE_ID, FULL_GPU_PROBE_CANDIDATE_ID, FULL_GRID_BACKEND_ID,
      build_r6_route_gate_status, build_r7_reduction_status)
    recomputed_r6 = build_r6_route_gate_status(
      evidence["r5_geometry_search"], target_evidence=evidence["target_role_probe"],
      independent_epoch_evidence=evidence["independent_epoch_evidence"])
    recomputed_r7 = build_r7_reduction_status(evidence["target_role_probe"])
    candidate_rows, retained_r6, retained_r7 = evidence["role_candidates"], evidence["r6_route_gate_status"], evidence["r7_reduction_status"]
  except Exception: return None
  if retained_r6 != recomputed_r6 or retained_r7 != recomputed_r7 or \
      recomputed_r6.get("status") != "READY_FOR_ONE_ROLE_OPT_IN" or \
      recomputed_r7.get("status") != "PASS_TARGET_ROLE_REDUCTION": return None
  if evidence.get("promotion_verdict") != "ONE_ROLE_EVIDENCE_READY_PRODUCTION_PROMOTION_BLOCKED" or \
      evidence.get("production_promotion_verdict") != "BLOCKED" or \
      evidence.get("production_dispatch_changed") is not False or evidence.get("default_route") != DIRECT_PACKED_ROUTE or \
      evidence.get("candidate_route_id") != CANDIDATE_ROUTE_ID:
    return None
  if not isinstance(candidate_rows, list) or len(candidate_rows) != 1 or not isinstance(candidate_rows[0], Mapping): return None
  candidate = candidate_rows[0]
  row = expected[Q4_CANDIDATE_ROLE][0]
  expected_shape = {"role":Q4_CANDIDATE_ROLE, **{key.upper():row["shape"][key] for key in ("m", "n", "k")}}
  if candidate.get("candidate_id") != FULL_GPU_PROBE_CANDIDATE_ID or candidate.get("backend") != FULL_GRID_BACKEND_ID or \
      candidate.get("role") != Q4_CANDIDATE_ROLE or candidate.get("shape") != expected_shape or \
      candidate.get("status") != "one_role_evidence_ready" or \
      candidate.get("research_opt_in_implementation_eligible") is not True or \
      candidate.get("one_role_opt_in_eligible") is not False or candidate.get("route_binding_implemented") is not False or \
      candidate.get("live_route_census_performed") is not False or candidate.get("promotion_eligible") is not False or \
      candidate.get("production_dispatch_changed") is not False or candidate.get("default_route") != DIRECT_PACKED_ROUTE or \
      candidate.get("evidence") != evidence.get("target_role_probe"):
    return None
  negative = recomputed_r6.get("negative_role_fallback_smoke")
  rejected = sorted(role for role in expected if role != Q4_CANDIDATE_ROLE)
  if not isinstance(negative, Mapping) or negative.get("status") != "PASS_STATIC_DESCRIPTOR" or \
      negative.get("accepted_roles") != [Q4_CANDIDATE_ROLE] or \
      sorted(negative.get("rejected_roles", ())) != rejected or negative.get("rollback_route") != DIRECT_PACKED_ROUTE or \
      negative.get("ffn_gate_up_only") is not True or negative.get("static_negative_role_scope") is not True or \
      negative.get("static_direct_packed_rollback") is not True or negative.get("research_descriptor_unbound") is not True or \
      negative.get("production_dispatch_changed") is not False:
    return None
  return {
    "schema":"tinygrad.q4k_q8_one_role_joined_qualification.v1", "status":"qualified",
    "source_schema":Q4_JOINED_EVIDENCE_SCHEMA, "source_evidence_identity":_identity("q4_one_role_joined_evidence", evidence),
    "candidate_role":Q4_CANDIDATE_ROLE, "negative_roles":rejected, "fallback_route":DIRECT_PACKED_ROUTE,
    "negative_role_evidence_scope":"static_route_manifest_descriptor", "production_promotion":False,
  }


def _validated_q6_evidence(evidence: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any] | None:
  identity = evidence.get("qualification_identity")
  unsigned = {key:value for key, value in evidence.items() if key != "qualification_identity"}
  computed = "q6_direct_packed:sha256:" + hashlib.sha256(_semantic(unsigned)).hexdigest()
  if evidence.get("schema") != Q6_EVIDENCE_SCHEMA or evidence.get("status") != "qualified" or \
      evidence.get("route_id") != DIRECT_PACKED_ROUTE or evidence.get("canonical_identity") != expected["canonical_identity"] or \
      evidence.get("workload") != _workload(expected["row"]) or identity != computed:
    return None
  return dict(evidence)


def _validated_evidence(inventory: Mapping[str, Any], q4_evidence: Mapping[str, Any] | None,
                        q6_evidence: Sequence[Mapping[str, Any]]):
  rows, q4_expected, q6_expected = _expected_evidence(inventory)
  q4_qualification = _validated_q4_joined_evidence(q4_evidence, q4_expected)
  q4_present = set(q4_expected) if q4_qualification is not None else set()
  q6_present, q6_by_id = set(), {}
  for evidence in q6_evidence if len(q6_evidence) == 2 else ():
    if not isinstance(evidence, Mapping): continue
    identity = evidence.get("canonical_identity")
    row = q6_expected.get(identity)
    if row is None or identity in q6_present: continue
    validated = _validated_q6_evidence(evidence, {"canonical_identity":identity, "row":row})
    if validated is not None: q6_present.add(identity); q6_by_id[identity] = validated
  return rows, q4_expected, q6_expected, q4_present, q6_present, q4_qualification, q6_by_id


def missing_qualification_evidence(inventory: Mapping[str, Any], *, q4_evidence: Mapping[str, Any] | None = None,
                                   q6_evidence: Sequence[Mapping[str, Any]] = ()) -> tuple[str, ...]:
  """Return every absent exact evidence identity in deterministic policy-row order."""
  rows, q4_expected, q6_expected, q4_present, q6_present, _, _ = _validated_evidence(inventory, q4_evidence, q6_evidence)
  missing = []
  for row in rows:
    if row["quant_format"] == "Q4_K":
      identity = q4_expected[row["role"]][1].canonical_identity
      kind = "one_role_joined_candidate" if row["role"] == Q4_CANDIDATE_ROLE else "one_role_joined_direct_packed_negative"
      if row["role"] not in q4_present: missing.append(f"Q4_K:{row['role']}:{identity}:{kind}")
    else:
      identity = next(identity for identity, expected in q6_expected.items() if expected["role"] == row["role"])
      if identity not in q6_present: missing.append(f"Q6_K:{row['role']}:{identity}:direct_packed_qualification")
  return tuple(missing)


def build_six_row_policy_artifact(inventory: Mapping[str, Any], *, q4_evidence: Mapping[str, Any] | None = None,
                                  q6_evidence: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
  """Build the immutable semantic artifact, or fail without partially binding rows."""
  missing = missing_qualification_evidence(inventory, q4_evidence=q4_evidence, q6_evidence=q6_evidence)
  if missing: raise MissingQualificationEvidence(missing)
  rows, q4_expected, _, _, _, q4_qualification, q6_by_id = _validated_evidence(inventory, q4_evidence, q6_evidence)
  assert q4_qualification is not None
  q4_by_role = {admission.normalized_payload["workload"]["role"]:entry
                for entry, admission in admitted_q4k_non_fitting_roles(inventory)}
  bindings = {(row["inventory_key"]["role"], row["inventory_key"]["quant_format"]):row["canonical_identity"]
              for row in inventory["bindings"]}
  entry = q4_by_role[Q4_CANDIDATE_ROLE]
  entries = [{"canonical_identity":entry.canonical_identity, "payload":entry.payload}]
  fallbacks = []
  for row in rows:
    if row["quant_format"] == "Q4_K":
      if row["role"] == Q4_CANDIDATE_ROLE: continue
      evidence = {
        "schema":"tinygrad.q4k_direct_packed_negative_role_qualification.v1", "status":"qualified",
        "source_evidence_identity":q4_qualification["source_evidence_identity"],
        "canonical_identity":q4_expected[row["role"]][1].canonical_identity,
        "workload":_workload(row), "candidate_rejected":True, "route_id":DIRECT_PACKED_ROUTE,
        "evidence_scope":"static_route_manifest_descriptor", "live_negative_role_tested":False,
        "production_promotion":False,
      }
    else:
      evidence = q6_by_id[bindings[(row["role"], "Q6_K")]]
    evidence_identity = _route_identity("fallback_evidence", evidence)
    content = {"workload":_workload(row), "route_id":DIRECT_PACKED_ROUTE, "evidence_identity":evidence_identity}
    fallbacks.append({**content, "evidence":evidence, "fallback_identity":_route_identity("fallback", content)})
  candidate_set = {"schema":"boltbeam.full_kernel_candidate_set.v1", "entries":entries, "fallbacks":fallbacks}
  capability = {"target":dict(TARGET), "phases":["prefill"], "quant_formats":["Q4_K", "Q6_K"]}
  exact_inventory = {**inventory["inventory"], "target":dict(TARGET)}
  policy_rows = canonical_policy_rows(exact_inventory, capability, candidate_set,
                                      route_id="q4k_q8_five_buffer_research")
  artifact = {"schema":SCHEMA, "status":"research_only", "production_promotion":False,
              "inventory_identity":inventory["inventory_identity"], "candidate_set":candidate_set,
              "q4_one_role_qualification":q4_qualification,
              "policy_rows":list(policy_rows)}
  return {**artifact, "artifact_identity":_identity("qk_exact_six_row_policy", artifact)}


def load_explicit_evidence(q4_path: str | Path, q6_paths: Sequence[str | Path]):
  """Load explicitly named canonical evidence paths. No directory discovery is performed."""
  if len(q6_paths) != 2: raise ValueError(f"exact six-row policy requires exactly two Q6 evidence paths, got {len(q6_paths)}")
  return json.loads(Path(q4_path).read_text()), tuple(json.loads(Path(path).read_text()) for path in q6_paths)


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("inventory"); parser.add_argument("--q4-evidence", required=True)
  parser.add_argument("--q6-evidence", nargs=2, required=True, metavar=("ATTN_KV", "FFN_DOWN"))
  parser.add_argument("--output")
  args = parser.parse_args(argv)
  inventory = json.loads(Path(args.inventory).read_text())
  q4, q6 = load_explicit_evidence(args.q4_evidence, args.q6_evidence)
  encoded = json.dumps(build_six_row_policy_artifact(inventory, q4_evidence=q4, q6_evidence=q6), sort_keys=True)
  if args.output: Path(args.output).write_text(encoded + "\n")
  else: print(encoded)


__all__ = ["SCHEMA", "Q4_JOINED_EVIDENCE_SCHEMA", "Q6_EVIDENCE_SCHEMA", "MissingQualificationEvidence", "missing_qualification_evidence",
           "build_six_row_policy_artifact", "load_explicit_evidence", "main"]


if __name__ == "__main__": main()
