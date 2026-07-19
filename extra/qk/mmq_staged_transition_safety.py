"""CPU-only safety classification for staged/direct route transitions.

This module does not execute or time either route.  It joins already-retained
individual-route authorities with guarded ``direct_packed -> staged_candidate``
diagnostics and selects the passing direct route only when both PM4 and AQL
show the same candidate-at-epoch-zero SQ type-2 failure after direct returned
PASS.  A successful classification is deliberately not a C8 timing result or
a production-promotion claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from extra.qk.direct_packed_executable_attestor import (
  QUALIFICATION_SCHEMA, DirectPackedAttestationBindings,
  validate_direct_packed_qualification_artifact,
)
from extra.qk.mmq_attn_qo_c6_binding import C6_BINDING_SCHEMA
from extra.qk.mmq_frozen_staged_c8_sessions import (
  ROUTE_SEQUENCE_SCHEMA, _validated_queue_attestation,
)
from extra.qk.mmq_frozen_staged_family import (
  QUEUE_MODES, FrozenStagedFamily,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_transition_safety_classification.v1"
CANDIDATE_AUTHORITY_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.staged_c7_queue_capture_isolation.v1"
GUARDED_ROUTE_SEQUENCE_SCHEMA = f"{ROUTE_SEQUENCE_SCHEMA}.guarded"
DIRECT_RECEIPT_SCHEMA = \
  "tinygrad.direct_packed.complete_role_timing_receipt.v1"
STAGED_FAILURE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_candidate_failure.v1"
DEFAULT_DIRECT_PROGRAM_PREFIX = "q4k_gen_prefill_"
_HEX = frozenset("0123456789abcdef")
_SQ_TYPE_2 = re.compile(r"\bsq_intr:.*\btype 2\b", re.IGNORECASE)


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(
      f"{label} fields differ: expected {sorted(expected)!r}, "
      f"got {sorted(value)!r}")


def _digest(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value.startswith("sha256:") or \
     len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _lower_digest(value: Any, label: str) -> str:
  if not isinstance(value, str) or len(value) != 64 or \
     any(char not in _HEX for char in value):
    raise ValueError(f"{label} must be a lowercase SHA-256 digest")
  return value


def _c6(
    value: Any, *, family: FrozenStagedFamily, software_identity: str,
    ) -> dict[str, Any]:
  row = dict(_mapping(value, "C6 correctness evidence"))
  _exact_keys(row, {
    "schema", "status", "family_identity", "candidate_executable_identity",
    "candidate_binary_sha256", "workload_identity", "input_identity",
    "device_identity", "software_identity", "queue_correctness",
    "queue_comparators", "evidence_identity",
  }, "C6 correctness evidence")
  payload = {key: item for key, item in row.items() if key != "evidence_identity"}
  if row["schema"] != C6_BINDING_SCHEMA or row["status"] != "PASS" or \
     row["family_identity"] != family.family_identity or \
     row["candidate_binary_sha256"] != family.binding.binary_sha256 or \
     row["software_identity"] != software_identity or \
     row["evidence_identity"] != _identity(payload):
    raise ValueError("C6 family, binary, software, or content identity differs")
  _digest(row["candidate_executable_identity"], "C6 candidate executable identity")
  for field in ("workload_identity", "input_identity", "device_identity",
                "software_identity"):
    _digest(row[field], f"C6 {field}")
  for field in ("queue_correctness", "queue_comparators"):
    values = _mapping(row[field], f"C6 {field}")
    if set(values) != set(QUEUE_MODES):
      raise ValueError(f"C6 {field} must contain exactly {QUEUE_MODES!r}")
    for queue in QUEUE_MODES:
      _digest(values[queue], f"C6 {field}.{queue}")
  return row


def _candidate_authority(
    value: Any, *, family: FrozenStagedFamily, queue_mode: str,
    expected_identity: str,
    ) -> dict[str, Any]:
  row = dict(_mapping(value, f"{queue_mode} staged candidate authority"))
  expected_identity = _digest(
    expected_identity, f"{queue_mode} expected candidate authority identity")
  validation = _mapping(
    row.get("probe_validation"), f"{queue_mode} candidate probe validation")
  raw = _mapping(row.get("raw_probe"), f"{queue_mode} candidate raw probe")
  correctness = _mapping(
    raw.get("correctness"), f"{queue_mode} candidate correctness")
  comparison = _mapping(
    correctness.get("comparison"), f"{queue_mode} candidate comparison")
  runtime = _mapping(
    raw.get("runtime_evidence"), f"{queue_mode} candidate runtime")
  if row.get("schema") != CANDIDATE_AUTHORITY_SCHEMA or \
     row.get("status") != "PASS" or row.get("queue_mode") != queue_mode or \
     row.get("family_identity") != family.family_identity or \
     row.get("binary_sha256") != family.binding.binary_sha256 or \
     row.get("program_key") != family.binding.program_key or \
     row.get("health_before") is not True or row.get("health_after") is not True or \
     row.get("kernel_faults") != [] or row.get("no_fallback") is not True or \
     row.get("production_dispatch_changed") is not False or \
     row.get("timed_out") is not False or row.get("child_status") != "passed" or \
     validation.get("all_checks_pass") is not True or \
     raw.get("status") != "PASS" or correctness.get("status") != "PASS" or \
     comparison.get("status") != "pass" or comparison.get("mismatch_count") != 0 or \
     runtime.get("queue_mode") != queue_mode or \
     runtime.get("binary_sha256") != family.binding.binary_sha256:
    raise ValueError(f"{queue_mode} individual staged candidate authority did not pass exactly")
  observed_identity = _identity(row)
  if observed_identity != expected_identity:
    raise ValueError(
      f"{queue_mode} staged candidate authority differs from composition identity")
  return {
    "authority_identity": observed_identity,
    "schema": row["schema"], "status": row["status"],
    "program_key": row["program_key"], "binary_sha256": row["binary_sha256"],
  }


def _transition(
    value: Any, *, family: FrozenStagedFamily, queue_mode: str,
    ) -> tuple[dict[str, Any], str]:
  row = dict(_mapping(value, f"{queue_mode} guarded transition"))
  _exact_keys(row, {
    "schema", "status", "exact_blocker", "queue_mode", "sequence",
    "health_before", "health_after", "kernel_faults",
    "kernel_fault_evidence", "timed_out", "error", "elapsed_seconds",
    "child", "spawn_count", "no_retry", "no_queue_fallback",
    "evidence_identity",
  }, f"{queue_mode} guarded transition")
  payload = {key: item for key, item in row.items() if key != "evidence_identity"}
  child = dict(_mapping(row["child"], f"{queue_mode} guarded transition child"))
  child_payload = {
    key: item for key, item in child.items() if key != "evidence_identity"}
  _exact_keys(child, {
    "schema", "status", "exact_blocker", "queue_mode", "family_identity",
    "session_identity", "clock_identity", "effective_queue_attestation",
    "sequence", "completed_positions", "invocation_counts", "breadcrumbs",
    "invocation_failure", "persistent_child_session", "no_retry",
    "no_queue_fallback", "production_dispatch_changed", "evidence_identity",
  }, f"{queue_mode} guarded transition child")
  if row["evidence_identity"] != _identity(payload) or \
     child["evidence_identity"] != _identity(child_payload):
    raise ValueError(f"{queue_mode} guarded transition content identity differs")
  if row["schema"] != GUARDED_ROUTE_SEQUENCE_SCHEMA or row["status"] != "BLOCKED" or \
     row["queue_mode"] != queue_mode or \
     row["sequence"] != ["direct_packed", "staged_candidate"] or \
     row["health_before"] is not True or row["health_after"] is not True or \
     row["timed_out"] is not False or row["error"] is not None or \
     row["spawn_count"] != 1 or row["no_retry"] is not True or \
     row["no_queue_fallback"] is not True:
    raise ValueError(f"{queue_mode} guarded transition envelope differs")
  if child["schema"] != ROUTE_SEQUENCE_SCHEMA or child["status"] != "BLOCKED" or \
     child["queue_mode"] != queue_mode or \
     child["family_identity"] != family.family_identity or \
     child["sequence"] != row["sequence"] or child["completed_positions"] != 1 or \
     child["invocation_counts"] != {"staged_candidate": 0, "direct_packed": 1} or \
     child["persistent_child_session"] is not True or \
     child["no_retry"] is not True or child["no_queue_fallback"] is not True or \
     child["production_dispatch_changed"] is not False or \
     child["exact_blocker"] != row["exact_blocker"]:
    raise ValueError(f"{queue_mode} guarded transition child contract differs")
  _validated_queue_attestation(child["effective_queue_attestation"], queue_mode)

  breadcrumbs = child["breadcrumbs"]
  if not isinstance(breadcrumbs, list) or len(breadcrumbs) != 1:
    raise ValueError(f"{queue_mode} transition must retain exactly one completed route")
  direct = _mapping(breadcrumbs[0], f"{queue_mode} direct transition breadcrumb")
  _exact_keys(direct, {
    "position", "route", "invocation_index", "receipt_schema",
    "receipt_status", "complete_role_ms", "persistent_session_lifecycle",
  }, f"{queue_mode} direct transition breadcrumb")
  elapsed = direct["complete_role_ms"]
  if direct["position"] != 0 or direct["route"] != "direct_packed" or \
     direct["invocation_index"] != 0 or \
     direct["receipt_schema"] != DIRECT_RECEIPT_SCHEMA or \
     direct["receipt_status"] != "PASS" or \
     not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or \
     not math.isfinite(elapsed) or elapsed <= 0 or \
     direct["persistent_session_lifecycle"] is not None:
    raise ValueError(f"{queue_mode} direct route did not PASS before the transition")

  failure = _mapping(
    child["invocation_failure"], f"{queue_mode} candidate transition failure")
  nested = _mapping(
    failure.get("nested_failure"), f"{queue_mode} nested candidate failure")
  raw = _mapping(
    nested.get("raw_probe_failure"), f"{queue_mode} raw candidate failure")
  if failure.get("position") != 1 or failure.get("route") != "staged_candidate" or \
     failure.get("invocation_index") != 0 or \
     failure.get("exception") != "StagedCandidateExecutionError" or \
     nested.get("schema") != STAGED_FAILURE_SCHEMA or \
     nested.get("status") != "BLOCKED" or \
     raw.get("completed_epochs") != 0:
    raise ValueError(
      f"{queue_mode} candidate failure was not the first transition epoch")

  faults = row["kernel_faults"]
  if not isinstance(faults, list) or not faults or \
     any(not isinstance(line, str) for line in faults):
    raise ValueError(f"{queue_mode} transition lacks kernel fault lines")
  sq_lines = [line for line in faults if "sq_intr:" in line.lower()]
  if not sq_lines or any(_SQ_TYPE_2.search(line) is None for line in sq_lines):
    raise ValueError(f"{queue_mode} transition SQ interrupt lines are not all type-2")
  fault_evidence = _mapping(
    row["kernel_fault_evidence"], f"{queue_mode} kernel fault evidence")
  if fault_evidence.get("schema") != "tinygrad.amd_kernel_fault_evidence.v1" or \
     fault_evidence.get("status") != "FAULTS":
    raise ValueError(f"{queue_mode} kernel fault evidence did not attest faults")
  return {
    "transition_evidence_identity": row["evidence_identity"],
    "child_evidence_identity": child["evidence_identity"],
    "session_identity": _digest(
      child["session_identity"], f"{queue_mode} transition session identity"),
    "clock_identity": child["clock_identity"],
    "direct_completed_before_candidate": True,
    "direct_receipt_status": "PASS",
    "candidate_failed_invocation_index": 0,
    "candidate_completed_epochs": 0,
    "sq_type_2_line_count": len(sq_lines),
    "health_recovered": True,
    "no_retry": True, "no_queue_fallback": True,
  }, child["clock_identity"]


def classify_staged_transition_safety(
    *, family: FrozenStagedFamily,
    c6_correctness_evidence: Mapping[str, Any],
    software_identity: str,
    candidate_authorities_by_queue: Mapping[str, Mapping[str, Any]],
    candidate_authority_identities_by_queue: Mapping[str, str],
    direct_authorities_by_queue: Mapping[str, Mapping[str, Any]],
    guarded_transitions_by_queue: Mapping[str, Mapping[str, Any]],
    required_direct_program_prefix: str = DEFAULT_DIRECT_PROGRAM_PREFIX,
    ) -> dict[str, Any]:
  """Classify an exact staged family as unsafe after dual-queue transitions."""
  if not isinstance(family, FrozenStagedFamily):
    raise TypeError("family must be a loader-validated FrozenStagedFamily")
  software_identity = _digest(software_identity, "software_identity")
  if not isinstance(required_direct_program_prefix, str) or \
     not required_direct_program_prefix:
    raise ValueError("required_direct_program_prefix must be non-empty")
  collections = {
    "candidate_authorities_by_queue": candidate_authorities_by_queue,
    "candidate_authority_identities_by_queue":
      candidate_authority_identities_by_queue,
    "direct_authorities_by_queue": direct_authorities_by_queue,
    "guarded_transitions_by_queue": guarded_transitions_by_queue,
  }
  for label, values in collections.items():
    if not isinstance(values, Mapping) or set(values) != set(QUEUE_MODES):
      raise ValueError(f"{label} must contain exactly {QUEUE_MODES!r}")
  c6 = _c6(
    c6_correctness_evidence, family=family,
    software_identity=software_identity)

  queues, sessions, clocks = {}, set(), set()
  for queue in QUEUE_MODES:
    candidate = _candidate_authority(
      candidate_authorities_by_queue[queue], family=family, queue_mode=queue,
      expected_identity=candidate_authority_identities_by_queue[queue])
    transition, clock_identity = _transition(
      guarded_transitions_by_queue[queue], family=family, queue_mode=queue)
    bindings = DirectPackedAttestationBindings(
      queue_mode=queue, workload_identity=c6["workload_identity"],
      input_identity=c6["input_identity"], device_identity=c6["device_identity"],
      software_identity=software_identity,
      comparator_identity=c6["queue_comparators"][queue],
      clock_identity=clock_identity,
      required_program_prefix=required_direct_program_prefix,
    ).validate()
    direct = validate_direct_packed_qualification_artifact(
      direct_authorities_by_queue[queue], bindings)
    sessions.add(transition["session_identity"])
    clocks.add(clock_identity)
    queues[queue] = {
      "queue_mode": queue,
      "candidate_individual_authority": candidate,
      "direct_individual_authority": {
        "schema": direct["schema"], "status": direct["status"],
        "qualification_identity": direct["qualification_identity"],
        "fallback_evidence_identity":
          direct["fallback_evidence"]["evidence_identity"],
      },
      "transition": transition,
      "candidate_disposition": "DISQUALIFIED",
      "selected_route": "direct_packed",
    }
  if len(sessions) != len(QUEUE_MODES):
    raise ValueError("PM4 and AQL transition session identities alias")
  if len(clocks) != 1:
    raise ValueError("PM4 and AQL transition clock identities differ")

  payload = {
    "schema": SCHEMA, "status": "PASS",
    "classification_scope": "transition_safety_only",
    "family_identity": family.family_identity,
    "candidate_executable_identity": c6["candidate_executable_identity"],
    "c6_evidence_identity": c6["evidence_identity"],
    "software_identity": software_identity,
    "clock_identity": next(iter(clocks)),
    "queues": queues,
    "decision": {
      "rule": (
        "safety-first: disqualify the exact staged candidate when both PM4 "
        "and AQL direct-to-candidate transitions fault at candidate epoch 0 "
        "after direct returned PASS"),
      "candidate_route": "staged_candidate",
      "candidate_disposition": "DISQUALIFIED",
      "selected_route": "direct_packed",
      "selection_kind": "SAFETY_FALLBACK",
      "c8_status": "BLOCKED_AT_C8",
      "timing_c8_status": "NOT_EVALUATED",
      "timing_c8_win": False,
      "promotion_eligible": False,
    },
    "promotion_eligible": False,
    "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def validate_staged_transition_safety_classification(
    value: Any,
    ) -> dict[str, Any]:
  """Validate the immutable output envelope without re-reading authorities."""
  row = dict(_mapping(value, "transition safety classification"))
  _exact_keys(row, {
    "schema", "status", "classification_scope", "family_identity",
    "candidate_executable_identity", "c6_evidence_identity",
    "software_identity", "clock_identity", "queues", "decision",
    "promotion_eligible", "production_dispatch_changed", "evidence_identity",
  }, "transition safety classification")
  identity = row.pop("evidence_identity", None)
  _digest(identity, "transition safety evidence identity")
  for field in (
      "family_identity", "candidate_executable_identity",
      "c6_evidence_identity", "software_identity"):
    _digest(row[field], f"transition safety {field}")
  if not isinstance(row["clock_identity"], str) or not row["clock_identity"]:
    raise ValueError("transition safety clock_identity must be non-empty")
  decision = _mapping(row["decision"], "transition safety decision")
  _exact_keys(decision, {
    "rule", "candidate_route", "candidate_disposition", "selected_route",
    "selection_kind", "c8_status", "timing_c8_status", "timing_c8_win",
    "promotion_eligible",
  }, "transition safety decision")
  expected_rule = (
    "safety-first: disqualify the exact staged candidate when both PM4 "
    "and AQL direct-to-candidate transitions fault at candidate epoch 0 "
    "after direct returned PASS")
  if identity != _identity(row) or row["schema"] != SCHEMA or \
     row["status"] != "PASS" or \
     row["classification_scope"] != "transition_safety_only" or \
     row["promotion_eligible"] is not False or \
     row["production_dispatch_changed"] is not False or \
     decision != {
       "rule": expected_rule, "candidate_route": "staged_candidate",
       "candidate_disposition": "DISQUALIFIED",
       "selected_route": "direct_packed",
       "selection_kind": "SAFETY_FALLBACK",
       "c8_status": "BLOCKED_AT_C8",
       "timing_c8_status": "NOT_EVALUATED", "timing_c8_win": False,
       "promotion_eligible": False,
     }:
    raise ValueError("transition safety classification identity or scope differs")
  queues = _mapping(row["queues"], "transition safety queues")
  if set(queues) != set(QUEUE_MODES):
    raise ValueError(f"transition safety queues must contain exactly {QUEUE_MODES!r}")
  sessions = set()
  for queue in QUEUE_MODES:
    queue_row = _mapping(queues[queue], f"{queue} transition safety row")
    _exact_keys(queue_row, {
      "queue_mode", "candidate_individual_authority",
      "direct_individual_authority", "transition",
      "candidate_disposition", "selected_route",
    }, f"{queue} transition safety row")
    candidate = _mapping(
      queue_row["candidate_individual_authority"],
      f"{queue} candidate individual authority")
    _exact_keys(candidate, {
      "authority_identity", "schema", "status", "program_key",
      "binary_sha256",
    }, f"{queue} candidate individual authority")
    _digest(candidate["authority_identity"], f"{queue} candidate authority identity")
    _lower_digest(candidate["program_key"], f"{queue} candidate program key")
    _lower_digest(candidate["binary_sha256"], f"{queue} candidate binary")
    direct = _mapping(
      queue_row["direct_individual_authority"],
      f"{queue} direct individual authority")
    _exact_keys(direct, {
      "schema", "status", "qualification_identity",
      "fallback_evidence_identity",
    }, f"{queue} direct individual authority")
    _digest(direct["qualification_identity"], f"{queue} direct qualification identity")
    _digest(
      direct["fallback_evidence_identity"],
      f"{queue} direct fallback evidence identity")
    transition = _mapping(queue_row["transition"], f"{queue} transition")
    _exact_keys(transition, {
      "transition_evidence_identity", "child_evidence_identity",
      "session_identity", "clock_identity",
      "direct_completed_before_candidate", "direct_receipt_status",
      "candidate_failed_invocation_index", "candidate_completed_epochs",
      "sq_type_2_line_count", "health_recovered", "no_retry",
      "no_queue_fallback",
    }, f"{queue} transition")
    for field in (
        "transition_evidence_identity", "child_evidence_identity",
        "session_identity"):
      _digest(transition[field], f"{queue} transition {field}")
    if queue_row["queue_mode"] != queue or \
       candidate["schema"] != CANDIDATE_AUTHORITY_SCHEMA or \
       candidate["status"] != "PASS" or \
       direct["schema"] != QUALIFICATION_SCHEMA or direct["status"] != "PASS" or \
       queue_row["candidate_disposition"] != "DISQUALIFIED" or \
       queue_row["selected_route"] != "direct_packed" or \
       transition["clock_identity"] != row["clock_identity"] or \
       transition["direct_completed_before_candidate"] is not True or \
       transition["direct_receipt_status"] != "PASS" or \
       transition["candidate_failed_invocation_index"] != 0 or \
       transition["candidate_completed_epochs"] != 0 or \
       not isinstance(transition["sq_type_2_line_count"], int) or \
       isinstance(transition["sq_type_2_line_count"], bool) or \
       transition["sq_type_2_line_count"] <= 0 or \
       transition["health_recovered"] is not True or \
       transition["no_retry"] is not True or \
       transition["no_queue_fallback"] is not True:
      raise ValueError(f"{queue} transition safety classification differs")
    sessions.add(transition["session_identity"])
  if len(sessions) != len(QUEUE_MODES):
    raise ValueError("transition safety session identities alias")
  return dict(value)


__all__ = [
  "DEFAULT_DIRECT_PROGRAM_PREFIX", "SCHEMA",
  "classify_staged_transition_safety",
  "validate_staged_transition_safety_classification",
]
