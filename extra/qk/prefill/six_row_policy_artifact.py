"""Deterministic, research-only policy artifact for the exact 14B mixed-QK rows."""
from __future__ import annotations

import argparse, hashlib, json
from collections.abc import Mapping, Sequence
from pathlib import Path
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


def _expected_evidence(inventory: Mapping[str, Any]):
  rows = _inventory_rows(inventory)
  q4 = {admission.canonical_identity:(row, entry, admission) for row in rows if row["quant_format"] == "Q4_K"
        for entry, admission in admitted_q4k_non_fitting_roles(inventory)
        if admission.normalized_payload["workload"]["role"] == row["role"]}
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
  if len(q4) != 4 or len(q6) != 2: raise ValueError("canonical evidence obligations are not exactly four Q4_K plus two Q6_K rows")
  return rows, q4, q6


def _validated_evidence(inventory: Mapping[str, Any], q4_evidence: Mapping[str, Any] | None,
                        q6_evidence: Sequence[Mapping[str, Any]]):
  rows, q4_expected, q6_expected = _expected_evidence(inventory)
  q4_rows = q4_evidence.get("rows") if isinstance(q4_evidence, Mapping) else None
  if not isinstance(q4_rows, list): q4_rows = []
  q4_present = set()
  if len(q4_rows) == 4 and q4_evidence and q4_evidence.get("schema") == Q4_EVIDENCE_SCHEMA and q4_evidence.get("passed") is True and \
      q4_evidence.get("status") == "pass" and q4_evidence.get("role_count") == 4 and q4_evidence.get("blockers") == []:
    for evidence in q4_rows:
      if not isinstance(evidence, Mapping): continue
      identity = evidence.get("canonical_identity")
      expected = q4_expected.get(identity)
      if expected is None or identity in q4_present: continue
      row = expected[0]
      if evidence.get("compile_status") == "pass" and \
          (evidence.get("role"), evidence.get("M"), evidence.get("N"), evidence.get("K")) == \
          (row["role"], row["shape"]["m"], row["shape"]["n"], row["shape"]["k"]): q4_present.add(identity)
  q6_present, q6_by_id = set(), {}
  for evidence in q6_evidence if len(q6_evidence) == 2 else ():
    if not isinstance(evidence, Mapping): continue
    identity = evidence.get("canonical_identity")
    expected = q6_expected.get(identity)
    if expected is None or identity in q6_present: continue
    if evidence.get("schema") == Q6_EVIDENCE_SCHEMA and evidence.get("status") == "qualified" and \
        evidence.get("route_id") == DIRECT_PACKED_ROUTE and isinstance(evidence.get("qualification_identity"), str) and \
        evidence.get("qualification_identity") and evidence.get("workload") == _workload(expected):
      q6_present.add(identity); q6_by_id[identity] = dict(evidence)
  return rows, q4_expected, q6_expected, q4_present, q6_present, q6_by_id


def missing_qualification_evidence(inventory: Mapping[str, Any], *, q4_evidence: Mapping[str, Any] | None = None,
                                   q6_evidence: Sequence[Mapping[str, Any]] = ()) -> tuple[str, ...]:
  """Return every absent exact evidence identity in deterministic policy-row order."""
  rows, q4_expected, q6_expected, q4_present, q6_present, _ = _validated_evidence(inventory, q4_evidence, q6_evidence)
  missing = []
  for row in rows:
    if row["quant_format"] == "Q4_K":
      identity = next(identity for identity, (expected, _, _) in q4_expected.items() if expected["role"] == row["role"])
      if identity not in q4_present: missing.append(f"Q4_K:{row['role']}:{identity}:five_buffer_role_gate")
    else:
      identity = next(identity for identity, expected in q6_expected.items() if expected["role"] == row["role"])
      if identity not in q6_present: missing.append(f"Q6_K:{row['role']}:{identity}:direct_packed_qualification")
  return tuple(missing)


def build_six_row_policy_artifact(inventory: Mapping[str, Any], *, q4_evidence: Mapping[str, Any] | None = None,
                                  q6_evidence: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
  """Build the immutable semantic artifact, or fail without partially binding rows."""
  missing = missing_qualification_evidence(inventory, q4_evidence=q4_evidence, q6_evidence=q6_evidence)
  if missing: raise MissingQualificationEvidence(missing)
  rows, _, _, _, _, q6_by_id = _validated_evidence(inventory, q4_evidence, q6_evidence)
  q4_by_role = {admission.normalized_payload["workload"]["role"]:entry
                for entry, admission in admitted_q4k_non_fitting_roles(inventory)}
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


__all__ = ["SCHEMA", "Q6_EVIDENCE_SCHEMA", "MissingQualificationEvidence", "missing_qualification_evidence",
           "build_six_row_policy_artifact", "load_explicit_evidence", "main"]


if __name__ == "__main__": main()
