"""CPU-only matched complete-role timing authority for generated ``ffn_gate_up``.

This contract closes the descriptor gap between a frozen staged candidate and
the production direct-packed fallback.  It does not collect timings or import
device/runtime code.  Callers inject immutable C6, C7, transition, candidate,
and fallback identities; the builder binds them to one exact workload and one
outer synchronized-wall boundary.  The validator rebuilds that exact
authority and fails closed on legacy, missing, or mismatched descriptors.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any


SCHEMA = "tinygrad.mmq_q4k_q8_1.ffn_gate_up_matched_complete_role_timing.v1"
STATUS = "READY"
QUEUE_MODES = ("PM4", "AQL")
CANDIDATE_ROUTE = "staged_candidate"
DIRECT_ROUTE = "direct_packed"

ROLE = "ffn_gate_up"
M, N, K = 512, 17408, 5120
K_PER_LAUNCH = 256
K_LAUNCHES = 20
OUTPUT_ELEMENTS = M * N

_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_FIELDS = {
  "schema", "status", "workload", "common_inputs", "timing_boundary",
  "allocation_residency_policy", "candidate", "direct_packed",
  "joint_session_c7", "queue_preconditions", "legacy_descriptors_accepted",
  "production_dispatch_changed", "evidence_identity",
}
_CANDIDATE_BINDING_FIELDS = {
  "family_identity", "candidate_executable_identity", "program_key",
  "binary_sha256",
}
_DIRECT_BINDING_FIELDS = {
  "qualification_identity", "executable_identity", "binary_sha256",
}
_C6_FIELDS = {
  "evidence_identity", "candidate_correctness_identity", "comparator_identity",
  "workload_identity", "input_identity",
}
_TRANSITION_FIELDS = {
  "candidate_candidate", "direct_direct", "direct_candidate_prefix1",
  "direct_candidate_full_role", "candidate_direct_candidate",
}


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


def _content_identity(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value.startswith("sha256:") or \
     len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _lower_digest(value: Any, label: str) -> str:
  if not isinstance(value, str) or len(value) != 64 or \
     any(char not in _HEX for char in value):
    raise ValueError(f"{label} must be a lowercase SHA-256 digest")
  return value


def _candidate_binding(value: Any) -> dict[str, str]:
  row = _mapping(value, "candidate binding")
  _exact_keys(row, _CANDIDATE_BINDING_FIELDS, "candidate binding")
  return {
    "family_identity":
      _content_identity(row["family_identity"], "candidate family identity"),
    "candidate_executable_identity": _content_identity(
      row["candidate_executable_identity"], "candidate executable identity"),
    "program_key": _lower_digest(row["program_key"], "candidate program key"),
    "binary_sha256":
      _lower_digest(row["binary_sha256"], "candidate binary SHA-256"),
  }


def _direct_bindings(value: Any) -> dict[str, dict[str, str]]:
  rows = _mapping(value, "direct bindings by queue")
  if set(rows) != set(QUEUE_MODES):
    raise ValueError(f"direct bindings must contain exactly {QUEUE_MODES!r}")
  normalized: dict[str, dict[str, str]] = {}
  for queue in QUEUE_MODES:
    row = _mapping(rows[queue], f"{queue} direct binding")
    _exact_keys(row, _DIRECT_BINDING_FIELDS, f"{queue} direct binding")
    normalized[queue] = {
      "qualification_identity": _content_identity(
        row["qualification_identity"], f"{queue} direct qualification identity"),
      "executable_identity": _content_identity(
        row["executable_identity"], f"{queue} direct executable identity"),
      "binary_sha256": _lower_digest(
        row["binary_sha256"], f"{queue} direct binary SHA-256"),
    }
  if normalized["PM4"]["qualification_identity"] == \
     normalized["AQL"]["qualification_identity"]:
    raise ValueError("PM4 and AQL direct qualification identities must be distinct")
  return normalized


def _c6_bindings(
    value: Any, *, workload_identity: str, input_identity: str,
    ) -> dict[str, dict[str, str]]:
  rows = _mapping(value, "C6 bindings by queue")
  if set(rows) != set(QUEUE_MODES):
    raise ValueError(f"C6 bindings must contain exactly {QUEUE_MODES!r}")
  normalized: dict[str, dict[str, str]] = {}
  for queue in QUEUE_MODES:
    row = _mapping(rows[queue], f"{queue} C6 binding")
    _exact_keys(row, _C6_FIELDS, f"{queue} C6 binding")
    normalized[queue] = {
      field: _content_identity(row[field], f"{queue} C6 {field}")
      for field in _C6_FIELDS
    }
    if normalized[queue]["workload_identity"] != workload_identity:
      raise ValueError(f"{queue} C6 workload identity differs from the exact role")
    if normalized[queue]["input_identity"] != input_identity:
      raise ValueError(f"{queue} C6 input identity differs from common inputs")
  if normalized["PM4"]["evidence_identity"] == \
     normalized["AQL"]["evidence_identity"]:
    raise ValueError("PM4 and AQL C6 evidence identities must be distinct")
  return normalized


def _transition_bindings(value: Any) -> dict[str, dict[str, str]]:
  rows = _mapping(value, "transition preflight bindings by queue")
  if set(rows) != set(QUEUE_MODES):
    raise ValueError(
      f"transition preflight bindings must contain exactly {QUEUE_MODES!r}")
  normalized: dict[str, dict[str, str]] = {}
  all_identities: list[str] = []
  for queue in QUEUE_MODES:
    row = _mapping(rows[queue], f"{queue} transition preflight binding")
    _exact_keys(row, _TRANSITION_FIELDS, f"{queue} transition preflight binding")
    normalized[queue] = {
      field: _content_identity(
        row[field], f"{queue} transition preflight {field}")
      for field in _TRANSITION_FIELDS
    }
    all_identities.extend(normalized[queue].values())
  if len(set(all_identities)) != len(all_identities):
    raise ValueError("transition preflight identities must be distinct")
  return normalized


def _timing_boundary() -> dict[str, Any]:
  payload = {
    "measurement_source": "outer_synchronized_wall",
    "clock_source": "dependency_injected_monotonic_ns",
    "start": {
      "precondition": "common_inputs_and_declared_resident_allocations_realized",
      "synchronization": "device_synchronize_completed",
      "point": "immediately_before_route_invocation",
    },
    "end": {
      "output_state": "complete_fp32_role_output_realized",
      "synchronization": "device_synchronize_completed",
      "point": "immediately_after_post_route_synchronize_before_attestation_or_readback",
    },
    "route_boundary_bindings": {
      CANDIDATE_ROUTE: "identical_outer_start_and_end",
      DIRECT_ROUTE: "identical_outer_start_and_end",
    },
    "candidate_q8_producer_included": True,
    "readback": {
      CANDIDATE_ROUTE: "excluded",
      DIRECT_ROUTE: "excluded",
      "between_paired_rounds": "forbidden",
    },
    "complete_role_only": True,
  }
  return {**payload, "boundary_identity": _identity(payload)}


def _allocation_residency_policy() -> dict[str, Any]:
  payload = {
    "joint_session_co_residency": True,
    "dense_fp16_weight_materialization": False,
    "all_live_allocations_counted_in_joint_c7": True,
    "common": {
      "logical_q4_packed": {
        "residency": "resident_before_timed_wall",
        "timing": "excluded_static_input_preparation",
        "memory": "counted_in_joint_c7",
      },
      "fp16_activation": {
        "residency": "resident_before_timed_wall",
        "timing": "excluded_common_input_realization",
        "memory": "counted_in_joint_c7",
      },
      "code_and_runtime": {
        "residency": "prequalified_and_resident_before_timed_wall",
        "timing": "excluded",
        "memory": "counted_in_joint_c7",
      },
    },
    CANDIDATE_ROUTE: {
      "epoch_major_q4_repack": {
        "residency": "model_load_static_resident",
        "timing": "excluded",
        "memory": "counted_in_joint_c7_distinct_from_direct_q4",
      },
      "q8_values_scales_sums": {
        "residency": "per_invocation_from_common_fp16_activation",
        "timing": "producer_and_realization_included",
        "memory": "counted_in_joint_c7_peak",
      },
      "compact_stages": {
        "residency": "persistent_preallocated",
        "timing": "allocation_excluded_transfers_included",
        "memory": "counted_in_joint_c7",
      },
      "output": {
        "residency": "persistent_preallocated",
        "timing": "allocation_excluded_zero_initialization_included",
        "memory": "counted_in_joint_c7",
      },
    },
    DIRECT_ROUTE: {
      "q4_packed": {
        "residency": "common_logical_q4_resident",
        "timing": "input_realization_excluded",
        "memory": "counted_in_joint_c7",
      },
      "output": {
        "residency": "production_per_invocation",
        "timing": "allocation_and_realization_included",
        "memory": "counted_in_joint_c7_peak",
      },
    },
  }
  return {**payload, "policy_identity": _identity(payload)}


def _candidate_transform() -> dict[str, Any]:
  return {
    "logical_q4": "q4_k_n_major_to_epoch_major_model_load_repack",
    "resident_fp16_activation":
      "fp16_to_fp32_then_physical_ds4_q8_1_per_invocation",
    "q8_producer": {
      "producer_id":
        "extra.qk.q4k_q8_activation_producer.produce_physical_ds4_q8_1_tensor",
      "outputs": ["q8_values", "q8_scales", "q8_original_sums"],
      "included_in_outer_synchronized_wall": True,
      "source": "common_resident_fp16_activation",
    },
    "complete_role": "20_ordered_k256_in_place_fp32_accumulation_dispatches",
    "output": "fp32_tokens_rows",
  }


def _direct_transform() -> dict[str, Any]:
  return {
    "logical_q4": "production_direct_packed_q4_k",
    "resident_fp16_activation":
      "production_direct_packed_fp16_contiguous_activation",
    "q8_producer": "not_applicable",
    "complete_role": "one_production_direct_packed_full_role_invocation",
    "output": "fp32_tokens_rows",
  }


def _transition_rows(identities: Mapping[str, str]) -> dict[str, Any]:
  return {
    "candidate_candidate": {
      "sequence": [CANDIDATE_ROUTE, CANDIDATE_ROUTE],
      "candidate_prefix_epochs": [K_LAUNCHES, K_LAUNCHES],
      "evidence_identity": identities["candidate_candidate"],
    },
    "direct_direct": {
      "sequence": [DIRECT_ROUTE, DIRECT_ROUTE],
      "candidate_prefix_epochs": [],
      "evidence_identity": identities["direct_direct"],
    },
    "direct_candidate_prefix1": {
      "sequence": [DIRECT_ROUTE, CANDIDATE_ROUTE],
      "candidate_prefix_epochs": [1],
      "evidence_identity": identities["direct_candidate_prefix1"],
    },
    "direct_candidate_full_role": {
      "sequence": [DIRECT_ROUTE, CANDIDATE_ROUTE],
      "candidate_prefix_epochs": [K_LAUNCHES],
      "evidence_identity": identities["direct_candidate_full_role"],
    },
    "candidate_direct_candidate": {
      "sequence": [CANDIDATE_ROUTE, DIRECT_ROUTE, CANDIDATE_ROUTE],
      "candidate_prefix_epochs": [K_LAUNCHES, K_LAUNCHES],
      "evidence_identity": identities["candidate_direct_candidate"],
    },
  }


def build_ffn_gate_up_matched_complete_role_timing_contract(
    *, workload_identity: str, input_identity: str,
    logical_q4_identity: str, resident_fp16_activation_identity: str,
    candidate_binding: Mapping[str, Any],
    direct_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    joint_session_c7_identity: str,
    c6_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    transition_preflight_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
  """Build the exact CPU-only authority required before matched GPU timing."""
  workload_identity = _content_identity(workload_identity, "workload identity")
  input_identity = _content_identity(input_identity, "input identity")
  logical_q4_identity = _content_identity(
    logical_q4_identity, "logical Q4 identity")
  resident_fp16_activation_identity = _content_identity(
    resident_fp16_activation_identity, "resident FP16 activation identity")
  candidate = _candidate_binding(candidate_binding)
  direct = _direct_bindings(direct_bindings_by_queue)
  c6 = _c6_bindings(
    c6_bindings_by_queue, workload_identity=workload_identity,
    input_identity=input_identity)
  transitions = _transition_bindings(
    transition_preflight_bindings_by_queue)
  joint_session_c7_identity = _content_identity(
    joint_session_c7_identity, "joint-session C7 identity")
  timing_boundary = _timing_boundary()
  allocation_policy = _allocation_residency_policy()

  payload = {
    "schema": SCHEMA,
    "status": STATUS,
    "workload": {
      "identity": workload_identity,
      "role": ROLE, "M": M, "N": N, "K": K,
      "k_per_launch": K_PER_LAUNCH, "k_launches": K_LAUNCHES,
      "complete_role": True, "output_elements": OUTPUT_ELEMENTS,
    },
    "common_inputs": {
      "identity": input_identity,
      "logical_q4_identity": logical_q4_identity,
      "logical_q4_format": "Q4_K",
      "resident_fp16_activation_identity":
        resident_fp16_activation_identity,
      "resident_fp16_activation_shape": [1, M, K],
      "resident_fp16_activation_dtype": "float16",
      "same_logical_inputs_for_both_routes": True,
    },
    "timing_boundary": timing_boundary,
    "allocation_residency_policy": allocation_policy,
    "candidate": {
      "route_id": CANDIDATE_ROUTE,
      **candidate,
      "transform": _candidate_transform(),
      "timing_boundary_identity": timing_boundary["boundary_identity"],
      "allocation_residency_policy_identity":
        allocation_policy["policy_identity"],
    },
    "direct_packed": {
      "route_id": DIRECT_ROUTE,
      "transform": _direct_transform(),
      "timing_boundary_identity": timing_boundary["boundary_identity"],
      "allocation_residency_policy_identity":
        allocation_policy["policy_identity"],
      "queue_qualifications": {
        queue: {
          "queue_mode": queue, "status": "PASS", **direct[queue],
        } for queue in QUEUE_MODES
      },
    },
    "joint_session_c7": {
      "status": "PASS",
      "evidence_identity": joint_session_c7_identity,
      "covers_routes": [CANDIDATE_ROUTE, DIRECT_ROUTE],
      "co_resident_peak_measured": True,
      "allocation_residency_policy_identity":
        allocation_policy["policy_identity"],
    },
    "queue_preconditions": {
      queue: {
        "queue_mode": queue,
        "c6": {
          "status": "PASS", **c6[queue],
        },
        "transition_preflights": {
          "status": "PASS",
          "candidate_family_identity": candidate["family_identity"],
          "direct_executable_identity": direct[queue]["executable_identity"],
          "sequences": _transition_rows(transitions[queue]),
        },
      } for queue in QUEUE_MODES
    },
    "legacy_descriptors_accepted": False,
    "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _assert_exact_descriptor(
    observed: Any, expected: Any, *, path: str = "contract",
    ) -> None:
  if isinstance(expected, Mapping):
    if not isinstance(observed, Mapping):
      raise ValueError(
        f"legacy, missing, or mismatched descriptor at {path}: expected mapping")
    if set(observed) != set(expected):
      missing = sorted(set(expected) - set(observed))
      extra = sorted(set(observed) - set(expected))
      raise ValueError(
        f"legacy, missing, or mismatched descriptor at {path}: "
        f"missing={missing!r}, extra={extra!r}")
    for key in expected:
      _assert_exact_descriptor(
        observed[key], expected[key], path=f"{path}.{key}")
    return
  if isinstance(expected, list):
    if not isinstance(observed, list) or len(observed) != len(expected):
      raise ValueError(
        f"legacy, missing, or mismatched descriptor at {path}: list differs")
    for index, (observed_item, expected_item) in enumerate(
        zip(observed, expected)):
      _assert_exact_descriptor(
        observed_item, expected_item, path=f"{path}[{index}]")
    return
  if observed != expected or type(observed) is not type(expected):
    raise ValueError(
      f"legacy, missing, or mismatched descriptor at {path}: "
      f"expected {expected!r}, got {observed!r}")


def validate_ffn_gate_up_matched_complete_role_timing_contract(
    value: Any, *, workload_identity: str, input_identity: str,
    logical_q4_identity: str, resident_fp16_activation_identity: str,
    candidate_binding: Mapping[str, Any],
    direct_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    joint_session_c7_identity: str,
    c6_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    transition_preflight_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
  """Validate one contract against dependency-injected exact authorities."""
  row = _mapping(value, "matched complete-role timing contract")
  if row.get("schema") != SCHEMA or set(row) != _TOP_LEVEL_FIELDS:
    raise ValueError(
      "legacy or missing matched complete-role timing descriptor is rejected")
  expected = build_ffn_gate_up_matched_complete_role_timing_contract(
    workload_identity=workload_identity, input_identity=input_identity,
    logical_q4_identity=logical_q4_identity,
    resident_fp16_activation_identity=resident_fp16_activation_identity,
    candidate_binding=candidate_binding,
    direct_bindings_by_queue=direct_bindings_by_queue,
    joint_session_c7_identity=joint_session_c7_identity,
    c6_bindings_by_queue=c6_bindings_by_queue,
    transition_preflight_bindings_by_queue=
      transition_preflight_bindings_by_queue)
  _assert_exact_descriptor(row, expected)
  payload = {key: item for key, item in row.items() if key != "evidence_identity"}
  if row["evidence_identity"] != _identity(payload):
    raise ValueError("matched complete-role timing content identity differs")
  return dict(row)


__all__ = [
  "CANDIDATE_ROUTE", "DIRECT_ROUTE", "K", "K_LAUNCHES", "K_PER_LAUNCH",
  "M", "N", "OUTPUT_ELEMENTS", "QUEUE_MODES", "ROLE", "SCHEMA", "STATUS",
  "build_ffn_gate_up_matched_complete_role_timing_contract",
  "validate_ffn_gate_up_matched_complete_role_timing_contract",
]
