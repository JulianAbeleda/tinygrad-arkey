from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  K, K_LAUNCHES, K_PER_LAUNCH, M, N, OUTPUT_ELEMENTS, SCHEMA,
  build_ffn_gate_up_matched_complete_role_timing_contract,
  validate_ffn_gate_up_matched_complete_role_timing_contract,
)


def _sid(label: str) -> str:
  return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _raw(label: str) -> str:
  return hashlib.sha256(label.encode()).hexdigest()


def _authority() -> dict:
  workload_identity, input_identity = _sid("workload"), _sid("inputs")
  return {
    "workload_identity": workload_identity,
    "input_identity": input_identity,
    "logical_q4_identity": _sid("logical-q4"),
    "resident_fp16_activation_identity": _sid("resident-fp16-activation"),
    "candidate_binding": {
      "family_identity": _sid("candidate-family"),
      "candidate_executable_identity": _sid("candidate-executable"),
      "program_key": _raw("candidate-program"),
      "binary_sha256": _raw("candidate-binary"),
    },
    "direct_bindings_by_queue": {
      queue: {
        "qualification_identity": _sid(f"{queue}-direct-qualification"),
        "executable_identity": _sid(f"{queue}-direct-executable"),
        "binary_sha256": _raw(f"{queue}-direct-binary"),
      } for queue in ("PM4", "AQL")
    },
    "joint_session_c7_identity": _sid("joint-session-c7"),
    "c6_bindings_by_queue": {
      queue: {
        "evidence_identity": _sid(f"{queue}-c6"),
        "candidate_correctness_identity":
          _sid(f"{queue}-candidate-correctness"),
        "comparator_identity": _sid(f"{queue}-comparator"),
        "workload_identity": workload_identity,
        "input_identity": input_identity,
      } for queue in ("PM4", "AQL")
    },
    "transition_preflight_bindings_by_queue": {
      queue: {
        name: _sid(f"{queue}-{name}") for name in (
          "candidate_candidate", "direct_direct",
          "direct_candidate_prefix1", "direct_candidate_full_role",
          "candidate_direct_candidate",
        )
      } for queue in ("PM4", "AQL")
    },
  }


def _build(authority: dict | None = None) -> dict:
  return build_ffn_gate_up_matched_complete_role_timing_contract(
    **(_authority() if authority is None else authority))


def _reidentify(contract: dict) -> None:
  payload = {key: value for key, value in contract.items()
             if key != "evidence_identity"}
  encoded = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  contract["evidence_identity"] = \
    "sha256:" + hashlib.sha256(encoded).hexdigest()


def _set_path(value: dict, path: tuple[str, ...], replacement) -> None:
  cursor = value
  for key in path[:-1]: cursor = cursor[key]
  cursor[path[-1]] = replacement


def test_build_and_validate_exact_matched_complete_role_contract():
  authority = _authority()
  contract = _build(authority)
  assert validate_ffn_gate_up_matched_complete_role_timing_contract(
    contract, **authority) == contract
  assert contract["schema"] == SCHEMA
  assert contract["status"] == "READY"
  assert contract["workload"] == {
    "identity": authority["workload_identity"],
    "role": "ffn_gate_up", "M": M, "N": N, "K": K,
    "k_per_launch": K_PER_LAUNCH, "k_launches": K_LAUNCHES,
    "complete_role": True, "output_elements": OUTPUT_ELEMENTS,
  }
  assert K_PER_LAUNCH * K_LAUNCHES == K
  assert OUTPUT_ELEMENTS == M * N


def test_outer_wall_q8_readback_and_residency_are_explicit():
  contract = _build()
  boundary = contract["timing_boundary"]
  assert boundary["measurement_source"] == "outer_synchronized_wall"
  assert boundary["route_boundary_bindings"] == {
    "staged_candidate": "identical_outer_start_and_end",
    "direct_packed": "identical_outer_start_and_end",
  }
  assert boundary["candidate_q8_producer_included"] is True
  assert boundary["readback"] == {
    "staged_candidate": "excluded", "direct_packed": "excluded",
    "between_paired_rounds": "forbidden",
  }
  assert contract["candidate"]["transform"]["q8_producer"][
    "included_in_outer_synchronized_wall"] is True
  policy = contract["allocation_residency_policy"]
  assert policy["joint_session_co_residency"] is True
  assert policy["all_live_allocations_counted_in_joint_c7"] is True
  assert policy["dense_fp16_weight_materialization"] is False
  assert policy["staged_candidate"]["epoch_major_q4_repack"]["memory"] == \
    "counted_in_joint_c7_distinct_from_direct_q4"
  assert policy["staged_candidate"]["q8_values_scales_sums"]["timing"] == \
    "producer_and_realization_included"


def test_exact_candidate_direct_c7_c6_and_transition_joins():
  authority = _authority()
  contract = _build(authority)
  assert {
    key: contract["candidate"][key]
    for key in authority["candidate_binding"]
  } == authority["candidate_binding"]
  assert contract["joint_session_c7"]["evidence_identity"] == \
    authority["joint_session_c7_identity"]
  for queue in ("PM4", "AQL"):
    direct = contract["direct_packed"]["queue_qualifications"][queue]
    assert direct["status"] == "PASS"
    assert {key: direct[key] for key in
            authority["direct_bindings_by_queue"][queue]} == \
      authority["direct_bindings_by_queue"][queue]
    c6 = contract["queue_preconditions"][queue]["c6"]
    assert c6["status"] == "PASS"
    assert {key: c6[key] for key in
            authority["c6_bindings_by_queue"][queue]} == \
      authority["c6_bindings_by_queue"][queue]
    preflights = contract["queue_preconditions"][queue][
      "transition_preflights"]
    assert preflights["status"] == "PASS"
    assert preflights["candidate_family_identity"] == \
      authority["candidate_binding"]["family_identity"]
    assert preflights["direct_executable_identity"] == \
      authority["direct_bindings_by_queue"][queue]["executable_identity"]
    assert preflights["sequences"]["direct_candidate_prefix1"][
      "candidate_prefix_epochs"] == [1]
    assert preflights["sequences"]["direct_candidate_full_role"][
      "candidate_prefix_epochs"] == [K_LAUNCHES]


@pytest.mark.parametrize(("path", "replacement"), (
  (("workload", "role"), "attn_qo"),
  (("workload", "M"), 128),
  (("workload", "N"), 128),
  (("workload", "K"), 256),
  (("workload", "k_launches"), 1),
  (("workload", "output_elements"), 16384),
  (("common_inputs", "logical_q4_identity"), _sid("other-q4")),
  (("common_inputs", "resident_fp16_activation_identity"),
   _sid("other-activation")),
  (("candidate", "family_identity"), _sid("other-family")),
  (("candidate", "program_key"), _raw("other-program")),
  (("candidate", "binary_sha256"), _raw("other-binary")),
  (("candidate", "transform", "resident_fp16_activation"),
   "legacy_prebuilt_q8"),
  (("timing_boundary", "start", "point"), "after_candidate_preparation"),
  (("timing_boundary", "end", "point"), "before_output_realization"),
  (("timing_boundary", "candidate_q8_producer_included"), False),
  (("timing_boundary", "readback", "staged_candidate"), "included"),
  (("timing_boundary", "readback", "direct_packed"), "included"),
  (("allocation_residency_policy", "joint_session_co_residency"), False),
  (("allocation_residency_policy", "dense_fp16_weight_materialization"), True),
  (("direct_packed", "queue_qualifications", "PM4",
    "qualification_identity"), _sid("other-direct-qualification")),
  (("direct_packed", "queue_qualifications", "AQL",
    "executable_identity"), _sid("other-direct-executable")),
  (("joint_session_c7", "evidence_identity"), _sid("other-c7")),
  (("queue_preconditions", "PM4", "c6", "evidence_identity"),
   _sid("other-c6")),
  (("queue_preconditions", "AQL", "transition_preflights", "sequences",
    "direct_candidate_full_role", "evidence_identity"),
   _sid("other-transition")),
))
def test_validator_fails_closed_on_reidentified_descriptor_mismatch(
    path, replacement):
  authority = _authority()
  contract = _build(authority)
  _set_path(contract, path, replacement)
  _reidentify(contract)
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    validate_ffn_gate_up_matched_complete_role_timing_contract(
      contract, **authority)


def test_validator_explicitly_rejects_legacy_and_missing_descriptors():
  authority = _authority()
  legacy = _build(authority)
  legacy["schema"] = \
    "q4k-q8-1-mmq-r5-geometry-search.v1"
  _reidentify(legacy)
  with pytest.raises(ValueError, match="legacy or missing"):
    validate_ffn_gate_up_matched_complete_role_timing_contract(
      legacy, **authority)

  missing_top = _build(authority)
  del missing_top["timing_boundary"]
  _reidentify(missing_top)
  with pytest.raises(ValueError, match="legacy or missing"):
    validate_ffn_gate_up_matched_complete_role_timing_contract(
      missing_top, **authority)

  missing_nested = _build(authority)
  del missing_nested["timing_boundary"]["readback"]
  _reidentify(missing_nested)
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    validate_ffn_gate_up_matched_complete_role_timing_contract(
      missing_nested, **authority)


def test_validator_rejects_changed_injected_authority():
  authority = _authority()
  contract = _build(authority)
  changed = deepcopy(authority)
  changed["logical_q4_identity"] = _sid("changed-authoritative-q4")
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    validate_ffn_gate_up_matched_complete_role_timing_contract(
      contract, **changed)


def test_builder_rejects_c6_input_or_workload_aliasing():
  authority = _authority()
  authority["c6_bindings_by_queue"]["PM4"]["input_identity"] = \
    _sid("wrong-input")
  with pytest.raises(ValueError, match="C6 input identity differs"):
    _build(authority)

  authority = _authority()
  authority["c6_bindings_by_queue"]["AQL"]["workload_identity"] = \
    _sid("wrong-workload")
  with pytest.raises(ValueError, match="C6 workload identity differs"):
    _build(authority)


def test_builder_rejects_missing_queue_and_legacy_transition_set():
  authority = _authority()
  del authority["direct_bindings_by_queue"]["AQL"]
  with pytest.raises(ValueError, match="direct bindings must contain exactly"):
    _build(authority)

  authority = _authority()
  del authority["transition_preflight_bindings_by_queue"]["PM4"][
    "direct_candidate_prefix1"]
  with pytest.raises(ValueError, match="transition preflight binding fields differ"):
    _build(authority)


def test_builder_rejects_aliased_queue_and_transition_evidence():
  authority = _authority()
  authority["direct_bindings_by_queue"]["AQL"]["qualification_identity"] = \
    authority["direct_bindings_by_queue"]["PM4"]["qualification_identity"]
  with pytest.raises(ValueError, match="qualification identities must be distinct"):
    _build(authority)

  authority = _authority()
  authority["transition_preflight_bindings_by_queue"]["AQL"][
    "candidate_candidate"] = \
    authority["transition_preflight_bindings_by_queue"]["PM4"][
      "candidate_candidate"]
  with pytest.raises(ValueError, match="preflight identities must be distinct"):
    _build(authority)


def test_module_has_no_device_or_runtime_imports():
  source = Path(
    "extra/qk/mmq_ffn_gate_up_matched_timing_contract.py").read_text()
  tree = ast.parse(source)
  imports = [
    node.module for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) and node.module is not None
  ] + [
    alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
    for alias in node.names
  ]
  assert not any(name.startswith(("tinygrad", "extra.qk")) for name in imports)
