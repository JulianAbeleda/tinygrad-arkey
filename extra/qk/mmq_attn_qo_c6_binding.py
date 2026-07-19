"""Exact dual-queue C6 binding and deterministic attn_qo fixture authority.

This module converts the retained PM4/AQL full-role C6 execution artifacts
into the small content-addressed C6 schema consumed by C8.  It also rebuilds
the exact Q4/source/Q8 bytes from the seeds retained by C6 and refuses any
hash, role, shape, queue, family, program, or memory-authority drift.

No Device is imported or initialized.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from extra.qk.mmq_exact_role_spec import ExactRoleSpec
from extra.qk.mmq_frozen_staged_family import QUEUE_MODES, FrozenStagedFamily
from extra.qk.mmq_staged_c7_authority import validate_staged_c7_authority_snapshot
from extra.qk.mmq_staged_c7_c8_contract import validate_staged_c7_memory_ledger


C6_BINDING_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c6_correctness_binding.v1"
COMPOSITION_SCHEMA = "tinygrad.mmq_q4k_q8_1.attn_qo_c6_composition.v1"
INPUT_IDENTITY_SCHEMA = "tinygrad.mmq_q4k_q8_1.attn_qo_exact_input.v1"
ACTIVATION_RELATION_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.attn_qo_fp32_source_to_fp16_production_activation.v1"
WORKLOAD_IDENTITY_SCHEMA = "tinygrad.mmq_q4k_q8_1.attn_qo_workload.v1"
CANDIDATE_EXECUTABLE_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_candidate_executable.v1"
CORRECTNESS_IDENTITY_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c6_queue_correctness.v1"
COMPARATOR_IDENTITY_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c6_comparator.v1"
EXECUTION_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_family_execution.v1"
PROBE_SCHEMA = "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1"
C7_CAPTURE_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c7_queue_capture_isolation.v1"
FIXTURE_SCHEMA = "tinygrad.mmq_q4k_q8_1_target_fixture.v1"
QO_ROLE = "attn_qo"
QO_SEEDS = {"q4": 20260721, "q8_source": 20260722}


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _bytes_sha256(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _digest(value: Any, label: str, *, prefixed: bool = True) -> str:
  value = _nonempty(value, label)
  offset = 7 if prefixed else 0
  if prefixed and not value.startswith("sha256:"):
    raise ValueError(f"{label} must use the sha256: prefix")
  if len(value) != offset + 64 or any(char not in "0123456789abcdef" for char in value[offset:]):
    raise ValueError(f"{label} must be a lowercase SHA-256 digest")
  return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping): raise ValueError(f"{label} must be a mapping")
  return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(
      f"{label} fields differ: expected {sorted(expected)!r}, got {sorted(value)!r}")


@dataclass(frozen=True)
class AttnQoExactFixture:
  """Rebuilt host bytes used by both retained C6 queue runs."""

  role_spec: ExactRoleSpec
  execution_fixture: Mapping[str, Any]
  words: np.ndarray
  source: np.ndarray
  production_activation: np.ndarray
  q8_values: np.ndarray
  q8_scales: np.ndarray
  q8_sums: np.ndarray
  q4_epoch_major: np.ndarray
  fixture_identity: str
  activation_relation: Mapping[str, Any]
  input_identity: str


def _validate_fixture_manifest(value: Any, role_spec: ExactRoleSpec) -> dict[str, Any]:
  value = dict(_mapping(value, "C6 execution fixture"))
  _exact_keys(value, {
    "schema", "role", "shape", "total_epochs", "seeds", "repack", "source_sha256",
  }, "C6 execution fixture")
  if role_spec.role != QO_ROLE or value["schema"] != FIXTURE_SCHEMA or \
     value["role"] != QO_ROLE or value["shape"] != list(role_spec.shape) or \
     value["total_epochs"] != role_spec.epochs or value["seeds"] != QO_SEEDS:
    raise ValueError("C6 execution fixture differs from exact attn_qo role/seeds")
  repack = _mapping(value["repack"], "C6 execution fixture repack")
  _exact_keys(repack, {
    "q4_sha256", "q4_layout", "q8_values_sha256", "q8_scales_sha256",
    "q8_sums_sha256", "q8_layout", "q4_epoch_major_sha256",
    "q4_epoch_major_layout", "q4_epoch_major_dtype", "q4_epoch_major_elements",
  }, "C6 execution fixture repack")
  expected_literals = {
    "q4_layout": "q4_k_bytes[n, k_epoch, 144]",
    "q8_layout": "q8_ds4[epoch, m, groups]",
    "q4_epoch_major_layout": "q4_k_bytes[k_epoch, n, 144]",
    "q4_epoch_major_dtype": "uint32",
  }
  if any(repack[key] != expected for key, expected in expected_literals.items()):
    raise ValueError("C6 execution fixture layout differs")
  for field in (
      "q4_sha256", "q8_values_sha256", "q8_scales_sha256",
      "q8_sums_sha256", "q4_epoch_major_sha256"):
    _digest(repack[field], f"C6 execution fixture {field}", prefixed=False)
  _digest(value["source_sha256"], "C6 execution fixture source_sha256", prefixed=False)
  expected_elements = role_spec.shape[1] * role_spec.epochs * 36
  if repack["q4_epoch_major_elements"] != expected_elements:
    raise ValueError("C6 q4 epoch-major element count differs")
  return value


def rebuild_attn_qo_exact_fixture(
    role_spec: ExactRoleSpec, execution_fixture: Mapping[str, Any],
    ) -> AttnQoExactFixture:
  """Regenerate and byte-check the exact retained C6 Qo fixture."""
  fixture = _validate_fixture_manifest(execution_fixture, role_spec)
  from extra.qk.mmq_llama_five_buffer_gpu_harness import \
    _pack_q4_epochs_contiguous, _random_q4_words
  from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference

  m, n, k = role_spec.shape
  words = _random_q4_words(n, k, fixture["seeds"]["q4"])
  source = np.random.default_rng(
    fixture["seeds"]["q8_source"]).standard_normal((m, k), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  blocks = words.view(np.uint8).reshape(n, role_spec.epochs, 144)
  epoch_major = _pack_q4_epochs_contiguous(blocks)
  repack = fixture["repack"]
  observed = {
    "q4_sha256": _bytes_sha256(blocks),
    "q8_values_sha256": _bytes_sha256(values),
    "q8_scales_sha256": _bytes_sha256(scales),
    "q8_sums_sha256": _bytes_sha256(sums),
    "q4_epoch_major_sha256": _bytes_sha256(epoch_major),
    "source_sha256": _bytes_sha256(source),
  }
  expected = {key: fixture["source_sha256"] if key == "source_sha256" else repack[key]
              for key in observed}
  if observed != expected:
    drifted = sorted(key for key in observed if observed[key] != expected[key])
    raise ValueError(f"rebuilt attn_qo fixture bytes differ: {drifted!r}")
  fixture_identity = _identity(fixture)
  production_activation = source.astype(np.float16)
  activation_relation = {
    "schema": ACTIVATION_RELATION_SCHEMA,
    "logical_source_dtype": "float32",
    "logical_source_shape": [m, k],
    "logical_source_sha256": observed["source_sha256"],
    "production_direct_packed_transform": "numpy_astype_float16_rne",
    "production_activation_dtype": "float16",
    "production_activation_shape": [m, k],
    "production_activation_sha256": _bytes_sha256(production_activation),
    "staged_activation_transform":
      "q8_1_mmq_ds4_quantize_reference_from_logical_fp32_source",
    "shared_execution_bytes_claimed": False,
  }
  input_identity = _identity({
    "schema": INPUT_IDENTITY_SCHEMA, "fixture_identity": fixture_identity,
    "role": role_spec.role, "shape": list(role_spec.shape),
    "activation_relation": activation_relation,
    "q4_sha256": observed["q4_sha256"],
  })
  return AttnQoExactFixture(
    role_spec, fixture, words, source, production_activation,
    values, scales, sums, epoch_major, fixture_identity,
    activation_relation, input_identity)


def _validate_raw_queue_c6(
    value: Any, *, family: FrozenStagedFamily, queue_mode: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  row = dict(_mapping(value, f"{queue_mode} raw C6"))
  role = family.binding.role_spec
  if row.get("schema") != EXECUTION_SCHEMA or row.get("status") != "PASS" or \
     row.get("gate") != "C6" or row.get("queue_mode") != queue_mode or \
     row.get("queue_qualification_is_separate") is not True or \
     row.get("role") != QO_ROLE or row.get("shape") != list(role.shape) or \
     row.get("prefix_epochs") != role.epochs or \
     row.get("family_identity") != family.family_identity or \
     row.get("program_key") != family.binding.program_key or \
     row.get("binary_sha256") != family.binding.binary_sha256 or \
     row.get("compile_performed") is not False or \
     row.get("requires_recompile") is not False or \
     row.get("hip_used") is not False or row.get("no_fallback") is not True or \
     row.get("exact_blocker") is not None:
    raise ValueError(f"{queue_mode} raw C6 top-level contract differs")
  raw = _mapping(row.get("raw_probe"), f"{queue_mode} raw C6 probe")
  validation = _mapping(row.get("validation"), f"{queue_mode} raw C6 validation")
  runtime = _mapping(raw.get("runtime_evidence"), f"{queue_mode} C6 runtime")
  correctness = _mapping(raw.get("correctness"), f"{queue_mode} C6 correctness")
  comparison = _mapping(correctness.get("comparison"), f"{queue_mode} C6 comparison")
  if raw.get("schema") != PROBE_SCHEMA or raw.get("status") != "PASS" or \
     raw.get("role") != QO_ROLE or raw.get("shape") != list(role.shape) or \
     raw.get("compile_performed") is not False or raw.get("requires_recompile") is not False or \
     raw.get("no_fallback") is not True or raw.get("production_dispatch_changed") is not False or \
     correctness.get("status") != "PASS" or \
     correctness.get("authority") != "same_session_fp16_rounded_ds4_reference" or \
     comparison.get("status") != "pass" or comparison.get("mismatch_count") != 0 or \
     runtime.get("queue_mode") != queue_mode or \
     runtime.get("amd_aql_effective") is not (queue_mode == "AQL") or \
     runtime.get("binary_sha256") != family.binding.binary_sha256 or \
     runtime.get("launch_count") != role.epochs or \
     validation.get("all_checks_pass") is not True:
    raise ValueError(f"{queue_mode} raw C6 probe/correctness/runtime contract differs")
  fixture = _validate_fixture_manifest(raw.get("execution_fixture"), role)
  frozen = _mapping(
    _mapping(raw.get("artifacts"), f"{queue_mode} C6 artifacts").get("frozen_bundle"),
    f"{queue_mode} C6 frozen bundle")
  canonical_sha = hashlib.sha256(_canonical(fixture)).hexdigest()
  if frozen.get("execution_fixture_canonical_sha256") != canonical_sha or \
     frozen.get("program_key") != family.binding.program_key:
    raise ValueError(f"{queue_mode} C6 fixture/program content identity differs")
  canary = _mapping(row.get("c4_runtime_canary"), f"{queue_mode} C4 runtime canary")
  if canary.get("status") != "PASS" or canary.get("all_checks_pass") is not True or \
     canary.get("queue_mode") != queue_mode or \
     canary.get("family_identity") != family.family_identity:
    raise ValueError(f"{queue_mode} embedded C4 runtime canary differs")
  return row, fixture, dict(canary)


def compose_attn_qo_c6_binding(
    *, family: FrozenStagedFamily, raw_c6_by_queue: Mapping[str, Any],
    c7_memory_ledger: Mapping[str, Any],
    c7_authority_snapshot: Mapping[str, Any],
    c7_captures_by_queue: Mapping[str, Any],
    ) -> dict[str, Any]:
  """Compose strict queue C6 artifacts with their shared passing C7 authority."""
  if not isinstance(family, FrozenStagedFamily):
    raise TypeError("family must be a loader-validated FrozenStagedFamily")
  if family.binding.role_spec.role != QO_ROLE:
    raise ValueError("C6 composer is exact to attn_qo")
  if not isinstance(raw_c6_by_queue, Mapping) or set(raw_c6_by_queue) != set(QUEUE_MODES):
    raise ValueError(f"raw_c6_by_queue must contain exactly {QUEUE_MODES!r}")
  if not isinstance(c7_captures_by_queue, Mapping) or \
     set(c7_captures_by_queue) != set(QUEUE_MODES):
    raise ValueError(f"c7_captures_by_queue must contain exactly {QUEUE_MODES!r}")
  c7 = validate_staged_c7_memory_ledger(c7_memory_ledger, family=family)
  if c7["status"] != "PASS" or c7["dense_fp16_weight_materialization"] is not False:
    raise ValueError("attn_qo C6 composition requires passing staged C7")
  authority = dict(c7["budget"]["authority"])
  authority_snapshot = validate_staged_c7_authority_snapshot(
    c7_authority_snapshot, verify_current_software=False)
  if authority_snapshot["memory_authority"] != authority:
    raise ValueError("C7 ledger and dual-queue authority snapshot differ")

  rows, fixtures, canaries, current_captures = {}, {}, {}, {}
  for queue in QUEUE_MODES:
    rows[queue], fixtures[queue], canaries[queue] = _validate_raw_queue_c6(
      raw_c6_by_queue[queue], family=family, queue_mode=queue)
  if fixtures["PM4"] != fixtures["AQL"]:
    raise ValueError("PM4/AQL C6 execution fixtures differ")
  fixture = fixtures["PM4"]
  rebuilt = rebuild_attn_qo_exact_fixture(family.binding.role_spec, fixture)
  fixture_identity, input_identity = rebuilt.fixture_identity, rebuilt.input_identity
  workload_identity = _identity({
    "schema": WORKLOAD_IDENTITY_SCHEMA, "role": QO_ROLE,
    "shape": list(family.binding.role_spec.shape),
    "epoch_count": family.binding.role_spec.epochs,
  })
  candidate_executable_identity = _identity({
    "schema": CANDIDATE_EXECUTABLE_SCHEMA,
    "family_identity": family.family_identity,
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
  })
  correctness_identities, comparator_identities = {}, {}
  raw_identities = {}
  for queue in QUEUE_MODES:
    capture = dict(_mapping(c7_captures_by_queue[queue], f"{queue} C7 capture"))
    current_raw = _mapping(capture.get("raw_probe"), f"{queue} C7 raw probe")
    current_validation = _mapping(
      capture.get("probe_validation"), f"{queue} C7 probe validation")
    current_runtime = _mapping(
      current_raw.get("runtime_evidence"), f"{queue} C7 runtime")
    current_correctness = _mapping(
      current_raw.get("correctness"), f"{queue} C7 correctness")
    current_comparison = _mapping(
      current_correctness.get("comparison"), f"{queue} C7 comparison")
    isolation = _mapping(
      capture.get("c4_runtime_canary_isolation"), f"{queue} C7 C4 isolation")
    current_canary = _mapping(
      isolation.get("runtime_canary"), f"{queue} C7 runtime canary")
    if capture.get("schema") != C7_CAPTURE_SCHEMA or capture.get("status") != "PASS" or \
       capture.get("queue_mode") != queue or \
       capture.get("family_identity") != family.family_identity or \
       capture.get("program_key") != family.binding.program_key or \
       capture.get("binary_sha256") != family.binding.binary_sha256 or \
       capture.get("health_before") is not True or capture.get("health_after") is not True or \
       capture.get("kernel_faults") != [] or capture.get("timed_out") is not False or \
       capture.get("target_dispatch_attempted") is not True or \
       capture.get("target_dispatch_attempted_authority") != "passing_child_queue_probe" or \
       capture.get("compile_performed") is not False or \
       capture.get("requires_recompile") is not False or \
       capture.get("no_fallback") is not True or \
       current_raw.get("status") != "PASS" or \
       current_raw.get("execution_fixture") != fixture or \
       current_runtime.get("queue_mode") != queue or \
       current_runtime.get("binary_sha256") != family.binding.binary_sha256 or \
       current_runtime.get("launch_count") != family.binding.role_spec.epochs or \
       current_correctness.get("status") != "PASS" or \
       current_comparison.get("status") != "pass" or \
       current_comparison.get("mismatch_count") != 0 or \
       current_validation.get("all_checks_pass") is not True or \
       current_canary.get("status") != "PASS" or \
       current_canary.get("queue_mode") != queue or \
       current_canary.get("family_identity") != family.family_identity:
      raise ValueError(f"{queue} current C7 full-role correctness capture differs")
    historical_correctness = rows[queue]["raw_probe"]["correctness"]
    for field in ("authority",):
      if current_correctness.get(field) != historical_correctness.get(field):
        raise ValueError(f"{queue} C7/C6 comparator authority differs")
    historical_comparison = historical_correctness["comparison"]
    for field in ("reference_shape", "reference_size", "rtol", "atol"):
      if current_comparison.get(field) != historical_comparison.get(field):
        raise ValueError(f"{queue} C7/C6 comparator facts differ")
    current_captures[queue] = capture
    canaries[queue] = dict(current_canary)
    raw_identities[queue] = _identity(rows[queue])
    correctness_identities[queue] = _identity({
      "schema": CORRECTNESS_IDENTITY_SCHEMA, "queue_mode": queue,
      "current_c7_capture_identity": _identity(capture),
      "historical_raw_c6_identity": raw_identities[queue],
      "family_identity": family.family_identity,
      "input_identity": input_identity,
    })
    comparator_identities[queue] = _identity({
      "schema": COMPARATOR_IDENTITY_SCHEMA, "queue_mode": queue,
      "authority": current_correctness["authority"],
      "input_identity": input_identity,
      "reference_shape": current_comparison["reference_shape"],
      "reference_size": current_comparison["reference_size"],
      "rtol": current_comparison["rtol"], "atol": current_comparison["atol"],
    })
  c6_payload = {
    "schema": C6_BINDING_SCHEMA, "status": "PASS",
    "family_identity": family.family_identity,
    "candidate_executable_identity": candidate_executable_identity,
    "candidate_binary_sha256": family.binding.binary_sha256,
    "workload_identity": workload_identity, "input_identity": input_identity,
    "device_identity": authority["device_identity"],
    "software_identity": authority["software_identity"],
    "queue_correctness": correctness_identities,
    "queue_comparators": comparator_identities,
  }
  c6 = {**c6_payload, "evidence_identity": _identity(c6_payload)}
  payload = {
    "schema": COMPOSITION_SCHEMA, "status": "PASS",
    "family_identity": family.family_identity,
    "c6_correctness_evidence": c6,
    "execution_fixture": fixture,
    "fixture_identity": fixture_identity,
    "activation_relation": rebuilt.activation_relation,
    "raw_c6_identities": raw_identities,
    "current_c7_capture_identities": {
      queue: _identity(current_captures[queue]) for queue in QUEUE_MODES},
    "runtime_canary_by_queue": canaries,
    "c7_evidence_identity": c7["evidence_identity"],
    "c7_authority_snapshot_identity": authority_snapshot["snapshot_identity"],
    "c7_memory_authority": authority,
    "promotion_eligible_on_candidate_win": False,
    "promotion_blocker":
      "staged and production direct_packed derive from one seeded fp32 source "
      "but consume different activation bytes; cross-route numerical parity is not certified",
    "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def read_json(path: str | Path, label: str) -> dict[str, Any]:
  value = json.loads(Path(path).read_text())
  if not isinstance(value, dict): raise ValueError(f"{label} must contain one JSON object")
  return value


__all__ = [
  "AttnQoExactFixture", "C6_BINDING_SCHEMA", "COMPOSITION_SCHEMA",
  "ACTIVATION_RELATION_SCHEMA", "INPUT_IDENTITY_SCHEMA", "QO_ROLE", "QO_SEEDS",
  "compose_attn_qo_c6_binding", "read_json", "rebuild_attn_qo_exact_fixture",
]
