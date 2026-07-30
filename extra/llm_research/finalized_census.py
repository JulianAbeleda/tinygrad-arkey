"""Fail-closed validation for serialized MR1/MR5 evidence consumers."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import re
from typing import Any

from tinygrad.engine.jit import GraphAdmissionDecision, GraphAdmissionReason
from tinygrad.engine.metadata import PROGRAM_IDENTITY_FIELDS
from tinygrad.llm.model_facts import ProgramIdentityMetadata

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _semantic_identity(identity: Any) -> bool:
  if not isinstance(identity, Mapping) or set(identity) != set(PROGRAM_IDENTITY_FIELDS): return False
  try: typed = ProgramIdentityMetadata(name=identity["role"], caller="", **identity)
  except (TypeError, ValueError): return False
  return all(getattr(typed, key) == identity[key] for key in PROGRAM_IDENTITY_FIELDS) and typed.accumulator_dtype is not None


def _exact_mapping(actual:Any, expected:Mapping, message:str) -> None:
  if not isinstance(actual, Mapping) or dict(actual) != dict(expected): raise ValueError(message)


def validate_finalized_census(census: Mapping[str, Any]) -> None:
  """Require complete, reconciled serialized evidence without changing generic capture."""
  if not isinstance(census, Mapping) or census.get("schema") != "tinygrad.graph_admission_census.v1":
    raise ValueError("finalized MR5 census is required")
  counts, records = census.get("counts"), census.get("records")
  if not isinstance(counts, Mapping) or not isinstance(records, list): raise ValueError("finalized census counts/records are unavailable")
  required_counts = ("logical_calls", "graph_members", "direct_calls", "ignored_slice_nodes", "graph_batches",
                     "constructor_failures", "semantic_calls", "generic_calls")
  if any(not isinstance(counts.get(key), int) or isinstance(counts[key], bool) or counts[key] < 0 for key in required_counts):
    raise ValueError("finalized census has invalid counts")
  logical = counts["logical_calls"]
  failures = census.get("constructor_failures")
  if counts["constructor_failures"] or failures != []:
    raise ValueError("finalized census contains graph-constructor failures")
  if logical != len(records) or logical != counts["graph_members"] + counts["direct_calls"] + counts["ignored_slice_nodes"]:
    raise ValueError("finalized census does not reconcile")
  if [row.get("call_index") for row in records if isinstance(row, Mapping)] != list(range(logical)):
    raise ValueError("finalized census call indexes do not reconcile")
  allowed_decisions = {item.value for item in GraphAdmissionDecision}
  allowed_reasons = {item.value for item in GraphAdmissionReason} - {GraphAdmissionReason.UNKNOWN.value}
  batches:dict[int, list[int]] = defaultdict(list)
  batch_sizes:dict[int, int] = {}
  direct_indexes:list[int] = []
  semantic_calls = 0
  role_histogram:Counter[str] = Counter()
  reason_histogram:Counter[str] = Counter()
  admission_histogram:Counter[str] = Counter()
  boundary_histogram:Counter[str] = Counter()
  assignments = Counter()
  for row in records:
    if not isinstance(row, Mapping): raise ValueError("finalized census contains an invalid record")
    decision, reason, admission_reason = row.get("decision"), row.get("reason"), row.get("admission_reason")
    boundary_reason = row.get("batch_boundary_reason")
    if decision not in allowed_decisions or reason not in allowed_reasons or admission_reason not in allowed_reasons or \
       boundary_reason is not None and boundary_reason not in allowed_reasons:
      raise ValueError("finalized census contains unknown decisions")
    assignment = row.get("assignment")
    if assignment not in ("graph", "direct", "ignored"): raise ValueError("finalized census has an invalid assignment")
    assignments[assignment] += 1
    graph_coordinates = (row.get("batch_index"), row.get("batch_member_index"), row.get("batch_size"))
    direct_index = row.get("direct_call_index")
    if assignment == "graph":
      if decision != GraphAdmissionDecision.ADMITTED.value or reason != GraphAdmissionReason.ADMITTED.value or \
         admission_reason != GraphAdmissionReason.ADMITTED.value or direct_index is not None or \
         any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in graph_coordinates[:2]) or \
         not isinstance(graph_coordinates[2], int) or isinstance(graph_coordinates[2], bool) or graph_coordinates[2] < 1:
        raise ValueError("finalized graph assignment is inconsistent")
      batch_index, member_index, batch_size = graph_coordinates
      if batch_index in batch_sizes and batch_sizes[batch_index] != batch_size:
        raise ValueError("finalized graph batch sizes are inconsistent")
      batch_sizes[batch_index] = batch_size
      batches[batch_index].append(member_index)
    elif assignment == "direct":
      if any(value is not None for value in graph_coordinates) or not isinstance(direct_index, int) or isinstance(direct_index, bool) or direct_index < 0:
        raise ValueError("finalized direct assignment is inconsistent")
      if decision == GraphAdmissionDecision.ADMITTED.value:
        consistent = reason == GraphAdmissionReason.SINGLETON_GRAPH_ELIDED.value and admission_reason == GraphAdmissionReason.ADMITTED.value
      elif decision == GraphAdmissionDecision.REJECTED.value:
        consistent = reason == admission_reason and reason != GraphAdmissionReason.ADMITTED.value
      else:
        consistent = decision == GraphAdmissionDecision.BATCH_BOUNDARY.value and \
          reason == admission_reason == GraphAdmissionReason.EXPLICIT_GRAPH_BARRIER.value
      # An admitted singleton can retain the reason it could not extend the
      # preceding graph batch. That boundary is orthogonal to its final direct
      # assignment and is emitted by the canonical serializer.
      if not consistent or decision != GraphAdmissionDecision.ADMITTED.value and boundary_reason is not None:
        raise ValueError("finalized direct assignment is inconsistent")
      direct_indexes.append(direct_index)
    else:
      if decision != GraphAdmissionDecision.IGNORED.value or reason != GraphAdmissionReason.IGNORED_SLICE_NODE.value or \
         admission_reason != GraphAdmissionReason.IGNORED_SLICE_NODE.value or boundary_reason is not None or \
         any(value is not None for value in (*graph_coordinates, direct_index)):
        raise ValueError("finalized ignored assignment is inconsistent")
    if assignment != "ignored" and any(_SHA256.fullmatch(str(row.get(key, ""))) is None
                                        for key in ("program_hash", "source_sha256", "binary_sha256")):
      raise ValueError("finalized census lacks material program identity")
    status, identities = row.get("metadata_status"), row.get("semantic_identities")
    if status == "semantic":
      if not isinstance(identities, list) or not identities or not all(_semantic_identity(identity) for identity in identities):
        raise ValueError("finalized census has incomplete semantic identity")
      roles = list(dict.fromkeys(identity["role"] for identity in identities))
      if row.get("workload_roles") != roles: raise ValueError("semantic census record has ambiguous identity")
      semantic_calls += 1
    elif status in ("generic", "unavailable"):
      if identities not in ([], ()) or row.get("workload_roles") != ["generic"]:
        raise ValueError("generic census record has ambiguous identity")
    else: raise ValueError("material census record lacks semantic or generic identity")
    reason_histogram[reason] += 1
    admission_histogram[admission_reason] += 1
    if boundary_reason is not None: boundary_histogram[boundary_reason] += 1
    role_histogram.update(row["workload_roles"])

  if sorted(batch_sizes) != list(range(len(batch_sizes))) or any(sorted(batches[index]) != list(range(size))
      for index,size in batch_sizes.items()): raise ValueError("finalized graph batch coordinates do not reconcile")
  if sorted(direct_indexes) != list(range(len(direct_indexes))): raise ValueError("finalized direct-call indexes do not reconcile")
  expected_counts = {"logical_calls":logical, "graph_members":assignments["graph"], "direct_calls":assignments["direct"],
    "ignored_slice_nodes":assignments["ignored"], "graph_batches":len(batch_sizes), "constructor_failures":0,
    "semantic_calls":semantic_calls, "generic_calls":logical-semantic_calls}
  _exact_mapping(counts, expected_counts, "finalized census derived counts do not reconcile")
  expected_batches = [{"batch_index":index, "size":batch_sizes[index]} for index in sorted(batch_sizes)]
  if census.get("batches") != expected_batches: raise ValueError("finalized census batches do not reconcile")
  _exact_mapping(census.get("reason_histogram"), dict(sorted(reason_histogram.items())), "finalized reason histogram does not reconcile")
  _exact_mapping(census.get("admission_reason_histogram"), dict(sorted(admission_histogram.items())),
                 "finalized admission histogram does not reconcile")
  _exact_mapping(census.get("batch_boundary_histogram"), dict(sorted(boundary_histogram.items())),
                 "finalized boundary histogram does not reconcile")
  _exact_mapping(census.get("workload_role_histogram"), dict(sorted(role_histogram.items())),
                 "finalized workload-role histogram does not reconcile")


__all__ = ["validate_finalized_census"]
