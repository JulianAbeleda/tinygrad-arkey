from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tinygrad import Tensor

from extra.qk.direct_packed_executable_attestor import (
  DirectPackedAttestationBindings, DirectPackedLinearExecutionCapture,
  build_direct_packed_qualification_artifact,
)
from extra.qk.mmq_frozen_epoch_runtime_preconstruction_canary import QUEUE_CLASSES
from extra.qk.mmq_frozen_staged_c8_sessions import (
  QUEUE_ATTESTATION_SCHEMA, ROUTE_SEQUENCE_SCHEMA,
)
from extra.qk.mmq_frozen_staged_family import (
  FrozenStagedFamily, load_frozen_staged_family_manifest,
)
from extra.qk.mmq_staged_transition_safety import (
  SCHEMA, classify_staged_transition_safety,
  validate_staged_transition_safety_classification,
)
from test.unit.test_mmq_frozen_staged_family import _loader, _produce


QUEUES = ("PM4", "AQL")
SOFTWARE_IDENTITY = "sha256:" + "5" * 64
CLOCK_IDENTITY = "clock-policy-0"


def _canonical(value):
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value):
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@pytest.fixture
def family(tmp_path) -> FrozenStagedFamily:
  role_spec, binding, output, _ = _produce(tmp_path)
  return load_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle",
    binding_loader=_loader(binding))


def _c6(family):
  payload = {
    "schema": "tinygrad.mmq_q4k_q8_1.staged_c6_correctness_binding.v1",
    "status": "PASS", "family_identity": family.family_identity,
    "candidate_executable_identity": "sha256:" + "1" * 64,
    "candidate_binary_sha256": family.binding.binary_sha256,
    "workload_identity": "sha256:" + "6" * 64,
    "input_identity": "sha256:" + "2" * 64,
    "device_identity": "sha256:" + "7" * 64,
    "software_identity": SOFTWARE_IDENTITY,
    "queue_correctness": {
      "PM4": "sha256:" + "3" * 64, "AQL": "sha256:" + "4" * 64},
    "queue_comparators": {
      "PM4": "sha256:" + "9" * 64, "AQL": "sha256:" + "a" * 64},
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _bindings(c6, queue):
  return DirectPackedAttestationBindings(
    queue_mode=queue, workload_identity=c6["workload_identity"],
    input_identity=c6["input_identity"],
    device_identity=c6["device_identity"],
    software_identity=c6["software_identity"],
    comparator_identity=c6["queue_comparators"][queue],
    clock_identity=CLOCK_IDENTITY, required_program_prefix="E_")


def _direct_authorities(c6, monkeypatch):
  bindings = {queue: _bindings(c6, queue) for queue in QUEUES}
  capture = DirectPackedLinearExecutionCapture(bindings_by_queue=bindings)
  result = {}
  for queue, aql in (("PM4", "0"), ("AQL", "1")):
    monkeypatch.setenv("AMD_AQL", aql)
    value = Tensor([1, 2, 3, 4], device="CPU")
    output = (value * value + 1).contiguous()
    capture.realize_output(output)
    observation = capture.observation_post_sync(output, queue)
    result[queue] = build_direct_packed_qualification_artifact(
      observation, bindings[queue])
  return result


def _candidate_authority(family, queue):
  return {
    "schema": "tinygrad.mmq_q4k_q8_1.staged_c7_queue_capture_isolation.v1",
    "status": "PASS", "queue_mode": queue,
    "family_identity": family.family_identity,
    "binary_sha256": family.binding.binary_sha256,
    "program_key": family.binding.program_key,
    "health_before": True, "health_after": True, "kernel_faults": [],
    "no_fallback": True, "production_dispatch_changed": False,
    "timed_out": False, "child_status": "passed",
    "probe_validation": {"all_checks_pass": True},
    "raw_probe": {
      "status": "PASS",
      "correctness": {
        "status": "PASS",
        "comparison": {"status": "pass", "mismatch_count": 0}},
      "runtime_evidence": {
        "queue_mode": queue, "binary_sha256": family.binding.binary_sha256},
    },
  }


def _attestation(queue):
  aql = "1" if queue == "AQL" else "0"
  return {
    "schema": QUEUE_ATTESTATION_SCHEMA,
    "authority": "instantiated_device_state", "device": "AMD",
    "requested_queue_mode": queue, "effective_queue_mode": queue,
    "effective_queue_class": QUEUE_CLASSES[queue],
    "expected_queue_class": QUEUE_CLASSES[queue],
    "environment_amd_aql": aql,
    "checks": {
      "environment_matches_requested": True,
      "requested_matches_effective": True,
      "queue_class_matches_effective": True,
    },
    "all_checks_pass": True,
  }


def _transition(family, queue):
  blocker = f"{queue} candidate failed at transition epoch zero"
  child_payload = {
    "schema": ROUTE_SEQUENCE_SCHEMA, "status": "BLOCKED",
    "exact_blocker": blocker, "queue_mode": queue,
    "family_identity": family.family_identity,
    "session_identity": "sha256:" + ("b" if queue == "PM4" else "c") * 64,
    "clock_identity": CLOCK_IDENTITY,
    "effective_queue_attestation": _attestation(queue),
    "sequence": ["direct_packed", "staged_candidate"],
    "completed_positions": 1,
    "invocation_counts": {"staged_candidate": 0, "direct_packed": 1},
    "breadcrumbs": [{
      "position": 0, "route": "direct_packed", "invocation_index": 0,
      "receipt_schema":
        "tinygrad.direct_packed.complete_role_timing_receipt.v1",
      "receipt_status": "PASS", "complete_role_ms": 10.0,
      "persistent_session_lifecycle": None,
    }],
    "invocation_failure": {
      "position": 1, "route": "staged_candidate", "invocation_index": 0,
      "exception": "StagedCandidateExecutionError", "error": "failed",
      "nested_failure": {
        "schema":
          "tinygrad.mmq_q4k_q8_1.frozen_staged_candidate_failure.v1",
        "status": "BLOCKED", "exact_blocker": "target failed",
        "exception": "ValueError", "error": "probe failed",
        "persistent_session_lifecycle": None,
        "raw_probe_failure": {
          "completed_epochs": 0, "error": "",
          "exact_blocker": "target-role GPU dispatch failed or timed out",
          "exception": "RuntimeError",
        },
      },
    },
    "persistent_child_session": True, "no_retry": True,
    "no_queue_fallback": True, "production_dispatch_changed": False,
  }
  child = {**child_payload, "evidence_identity": _identity(child_payload)}
  fault = (
    "[1.0] amdgpu: sq_intr: error, detail 0x00000000, "
    "type 2, sh 0, priv 1")
  payload = {
    "schema": f"{ROUTE_SEQUENCE_SCHEMA}.guarded", "status": "BLOCKED",
    "exact_blocker": blocker, "queue_mode": queue,
    "sequence": ["direct_packed", "staged_candidate"],
    "health_before": True, "health_after": True,
    "kernel_faults": [fault],
    "kernel_fault_evidence": {
      "schema": "tinygrad.amd_kernel_fault_evidence.v1",
      "status": "FAULTS"},
    "timed_out": False, "error": None, "elapsed_seconds": 1.0,
    "child": child, "spawn_count": 1, "no_retry": True,
    "no_queue_fallback": True,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _rehash_transition(row):
  child_payload = {
    key: value for key, value in row["child"].items()
    if key != "evidence_identity"}
  row["child"]["evidence_identity"] = _identity(child_payload)
  payload = {key: value for key, value in row.items()
             if key != "evidence_identity"}
  row["evidence_identity"] = _identity(payload)


def _inputs(family, monkeypatch):
  c6 = _c6(family)
  candidate_authorities = {
    queue: _candidate_authority(family, queue) for queue in QUEUES}
  return {
    "family": family, "c6_correctness_evidence": c6,
    "software_identity": SOFTWARE_IDENTITY,
    "candidate_authorities_by_queue": candidate_authorities,
    "candidate_authority_identities_by_queue": {
      queue: _identity(candidate_authorities[queue]) for queue in QUEUES},
    "direct_authorities_by_queue": _direct_authorities(c6, monkeypatch),
    "guarded_transitions_by_queue": {
      queue: _transition(family, queue) for queue in QUEUES},
    "required_direct_program_prefix": "E_",
  }


def test_dual_queue_epoch_zero_sq_type_2_selects_safety_fallback(
    family, monkeypatch):
  result = classify_staged_transition_safety(**_inputs(family, monkeypatch))
  assert result["schema"] == SCHEMA and result["status"] == "PASS"
  assert result["decision"] == {
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
  }
  assert result["promotion_eligible"] is False
  assert result["production_dispatch_changed"] is False
  assert validate_staged_transition_safety_classification(result) == result
  for queue in QUEUES:
    row = result["queues"][queue]
    assert row["candidate_disposition"] == "DISQUALIFIED"
    assert row["selected_route"] == "direct_packed"
    assert row["transition"]["direct_completed_before_candidate"] is True
    assert row["transition"]["candidate_completed_epochs"] == 0
    assert row["transition"]["sq_type_2_line_count"] == 1
    assert row["transition"]["health_recovered"] is True


@pytest.mark.parametrize(("mutation", "message"), [
  ("direct_not_pass", "direct route did not PASS"),
  ("candidate_epoch_one", "first transition epoch"),
  ("sq_type_one", "SQ interrupt lines are not all type-2"),
  ("unhealthy_after", "guarded transition envelope differs"),
  ("retry", "guarded transition envelope differs"),
])
def test_transition_semantics_fail_closed(
    family, monkeypatch, mutation, message):
  inputs = _inputs(family, monkeypatch)
  row = inputs["guarded_transitions_by_queue"]["PM4"]
  if mutation == "direct_not_pass":
    row["child"]["breadcrumbs"][0]["receipt_status"] = "BLOCKED"
  elif mutation == "candidate_epoch_one":
    row["child"]["invocation_failure"]["nested_failure"][
      "raw_probe_failure"]["completed_epochs"] = 1
  elif mutation == "sq_type_one":
    row["kernel_faults"][0] = row["kernel_faults"][0].replace("type 2", "type 1")
  elif mutation == "unhealthy_after":
    row["health_after"] = False
  elif mutation == "retry":
    row["no_retry"] = False
  _rehash_transition(row)
  with pytest.raises(ValueError, match=message):
    classify_staged_transition_safety(**inputs)


def test_individual_authority_and_identity_drift_fail_closed(
    family, monkeypatch):
  inputs = _inputs(family, monkeypatch)
  inputs["candidate_authorities_by_queue"]["AQL"]["health_after"] = False
  with pytest.raises(ValueError, match="individual staged candidate authority"):
    classify_staged_transition_safety(**inputs)

  inputs = _inputs(family, monkeypatch)
  inputs["candidate_authority_identities_by_queue"]["PM4"] = \
    "sha256:" + "0" * 64
  with pytest.raises(ValueError, match="differs from composition identity"):
    classify_staged_transition_safety(**inputs)

  inputs = _inputs(family, monkeypatch)
  inputs["direct_authorities_by_queue"]["PM4"]["fallback_evidence"][
    "software_identity"] = "sha256:" + "0" * 64
  with pytest.raises(ValueError, match="identity or scope differs|evidence"):
    classify_staged_transition_safety(**inputs)

  inputs = _inputs(family, monkeypatch)
  inputs["c6_correctness_evidence"]["software_identity"] = \
    "sha256:" + "0" * 64
  with pytest.raises(ValueError, match="C6 family, binary, software"):
    classify_staged_transition_safety(**inputs)


def test_requires_distinct_sessions_and_common_clock_policy(
    family, monkeypatch):
  inputs = _inputs(family, monkeypatch)
  pm4 = inputs["guarded_transitions_by_queue"]["PM4"]
  aql = inputs["guarded_transitions_by_queue"]["AQL"]
  aql["child"]["session_identity"] = pm4["child"]["session_identity"]
  _rehash_transition(aql)
  with pytest.raises(ValueError, match="session identities alias"):
    classify_staged_transition_safety(**inputs)

  inputs = _inputs(family, monkeypatch)
  aql = inputs["guarded_transitions_by_queue"]["AQL"]
  aql["child"]["clock_identity"] = "other-clock-policy"
  _rehash_transition(aql)
  # The direct authority is still bound to clock-policy-0, so drift is
  # rejected before it could become a cross-queue classification.
  with pytest.raises(ValueError, match="identity or scope differs|clock"):
    classify_staged_transition_safety(**inputs)


def test_output_validator_rejects_promotion_or_timing_claim(
    family, monkeypatch):
  result = classify_staged_transition_safety(**_inputs(family, monkeypatch))
  promoted = copy.deepcopy(result)
  promoted["decision"]["promotion_eligible"] = True
  payload = {key: value for key, value in promoted.items()
             if key != "evidence_identity"}
  promoted["evidence_identity"] = _identity(payload)
  with pytest.raises(ValueError, match="identity or scope differs"):
    validate_staged_transition_safety_classification(promoted)

  timing = copy.deepcopy(result)
  timing["decision"]["timing_c8_win"] = True
  payload = {key: value for key, value in timing.items()
             if key != "evidence_identity"}
  timing["evidence_identity"] = _identity(payload)
  with pytest.raises(ValueError, match="identity or scope differs"):
    validate_staged_transition_safety_classification(timing)

  extra = copy.deepcopy(result)
  extra["queues"]["PM4"]["transition"]["untrusted_extra"] = True
  payload = {key: value for key, value in extra.items()
             if key != "evidence_identity"}
  extra["evidence_identity"] = _identity(payload)
  with pytest.raises(ValueError, match="fields differ"):
    validate_staged_transition_safety_classification(extra)

  malformed = copy.deepcopy(result)
  malformed["queues"]["AQL"]["transition"]["session_identity"] = \
    "sha256:not-a-digest"
  payload = {key: value for key, value in malformed.items()
             if key != "evidence_identity"}
  malformed["evidence_identity"] = _identity(payload)
  with pytest.raises(ValueError, match="sha256 content identity"):
    validate_staged_transition_safety_classification(malformed)
