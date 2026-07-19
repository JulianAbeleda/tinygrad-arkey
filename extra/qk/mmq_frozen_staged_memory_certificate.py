"""CPU-only final-native C3 certificate for one compact staged PROGRAM.

The staged family reuses one K256 PROGRAM for every full-role epoch.  Its
native addresses therefore belong to the compact K256 ABI, not to the sparse
full-role epoch-offset ABI.  This adapter validates the immutable
``FrozenStagedFamily`` identity, certifies the selected native UOp graph over
the complete launch grid, and binds the result to the retained source,
binary, serialized PROGRAM, and family hashes.  It creates no Device,
runtime, allocator, or queue.
"""
from __future__ import annotations

import hashlib
import json
from math import prod
from typing import Any, Mapping

from tinygrad.uop.ops import Ops

from extra.qk.mmq_exact_role_spec import EPOCH_K, ExactRoleSpec
from extra.qk.mmq_frozen_epoch_memory_certificate import (
  ABI_NAMES as MEMORY_ABI_NAMES, _counter_row, _expected_source_counters,
  certify_native_program_memory, certify_source_sink_memory,
)
from extra.qk.mmq_frozen_staged_family import (
  SCHEMA as STAGED_FAMILY_SCHEMA, STATE as STAGED_FAMILY_STATE,
  FrozenStagedFamily, _manifest_payload, _validate_provenance,
)
from extra.qk.mmq_frozen_target_artifact import LEGACY_SCHEMA as LEGACY_TARGET_SCHEMA, SCHEMA as TARGET_SCHEMA
from extra.qk.prefill.frozen_exact_role_runtime import ABI_DTYPES, ABI_NAMES, FrozenExactRoleBinding


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_memory_certificate.v1"
C3A_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_source_memory_certificate.v1"
FULL_C3_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_full_memory_certificate.v1"
_MANIFEST_KEYS = {
  "schema", "state", "family_identity", "role", "artifact", "program",
  "staging", "queue_modes", "provenance",
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
  return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def _validated_family(family: FrozenStagedFamily) -> tuple[Mapping[str, Any], FrozenExactRoleBinding]:
  if not isinstance(family, FrozenStagedFamily) or not isinstance(family.binding, FrozenExactRoleBinding):
    raise TypeError("C3 staged certificate requires a loader-validated FrozenStagedFamily")
  manifest, binding = family.manifest, family.binding
  if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
    raise ValueError("C3 staged-family manifest fields differ")
  if manifest.get("schema") != STAGED_FAMILY_SCHEMA or manifest.get("state") != STAGED_FAMILY_STATE:
    raise ValueError("C3 staged-family schema or frozen state differs")
  provenance = _validate_provenance(manifest["provenance"], binding)
  expected = _manifest_payload(binding.role_spec, binding, provenance)
  observed = {key: value for key, value in manifest.items() if key != "family_identity"}
  expected_identity = f"sha256:{_sha256(_canonical_bytes(expected))}"
  if observed != expected or family.family_identity != expected_identity or \
     manifest.get("family_identity") != expected_identity:
    raise ValueError("C3 staged-family content differs from its exact binding or family identity")
  return manifest, binding


def _program_payload_identity(manifest: Mapping[str, Any], binding: FrozenExactRoleBinding) -> dict[str, Any]:
  artifact, program = binding.artifact, binding.artifact.program
  sources = [node.arg for node in program.src if node.op is Ops.SOURCE]
  binaries = [node.arg for node in program.src if node.op is Ops.BINARY]
  if sources != [artifact.source] or binaries != [artifact.binary]:
    raise ValueError("C3 staged PROGRAM native source or binary payload differs from its binding")
  source_sha256, binary_sha256 = _sha256(artifact.source.encode()), _sha256(artifact.binary)
  retained = manifest["program"]
  if source_sha256 != binding.source_sha256 or binary_sha256 != binding.binary_sha256 or \
     retained.get("source_sha256") != source_sha256 or retained.get("binary_sha256") != binary_sha256 or \
     retained.get("key") != program.key.hex() or binding.program_key != program.key.hex():
    raise ValueError("C3 staged PROGRAM key/source/binary identity differs")
  serialized_sha256 = retained.get("serialized_program_sha256")
  if not isinstance(serialized_sha256, str) or len(serialized_sha256) != 64 or \
     any(char not in "0123456789abcdef" for char in serialized_sha256):
    raise ValueError("C3 staged serialized PROGRAM identity is malformed")
  return {
    "program_key": program.key.hex(),
    "source_sha256": source_sha256,
    "binary_sha256": binary_sha256,
    "serialized_program_sha256": serialized_sha256,
    "native_source_and_binary_match_frozen_program": True,
  }


def _abi_certificate(role_spec: ExactRoleSpec, manifest: Mapping[str, Any]) -> dict[str, Any]:
  program, retained = role_spec.program, manifest["program"]
  expected = [{
    "slot": slot, "name": name, "dtype": f"{dtype}.ptr({elements})", "elements": elements,
    "nbytes": elements * dtype.itemsize, "direction": "inout" if slot == 0 else "in",
  } for slot, (name, dtype, elements) in enumerate(zip(ABI_NAMES, ABI_DTYPES, program.abi_elements))]
  if retained.get("abi") != expected:
    raise ValueError("C3 staged PROGRAM ABI slots, dtypes, extents, or directions differ")
  if any(row["nbytes"] > 0xffffffff for row in expected):
    raise ValueError("C3 staged PROGRAM ABI allocation extent exceeds uint32 addressability")
  return {
    "kernarg_bytes": 5 * 8,
    "globals": list(range(5)),
    "effects": {"outs": [0], "ins": list(range(5))},
    "slots": expected,
    "exact_five_kernarg_slots_dtypes_extents_and_effects": True,
    "all_allocation_extents_fit_uint32_without_overflow": True,
  }


def _pre_lowering_sink_scope(manifest: Mapping[str, Any], binding: FrozenExactRoleBinding) -> dict[str, Any]:
  """Describe retained source authority without evaluating or claiming C3a."""
  artifact, sink = binding.artifact, binding.artifact.sink
  artifact_schema = artifact.manifest.get("schema")
  if artifact_schema == LEGACY_TARGET_SCHEMA:
    if sink is not None:
      raise ValueError("C3b legacy v1 artifact cannot retain an unmanifested pre-lowering SINK")
    return {
      "artifact_schema": artifact_schema,
      "retention": "ABSENT_LEGACY_V1_ARTIFACT",
      "retained": False, "identity_bound": False,
      "evaluated_by_this_c3b_certificate": False, "c3a_claimed": False,
    }
  if artifact_schema != TARGET_SCHEMA or sink is None or sink.op is not Ops.SINK:
    raise ValueError("C3b staged v2 artifact lacks its retained pre-lowering SINK")
  sink_identity = sink.key.hex()
  if manifest["program"].get("sink_identity") != sink_identity:
    raise ValueError("C3b retained pre-lowering SINK differs from staged-family identity")
  source_sink = artifact.manifest.get("source_sink")
  if not isinstance(source_sink, Mapping) or \
     source_sink.get("authority") != "same_session_pre_lowering_sink_passed_to_compiler" or \
     source_sink.get("key") != sink_identity:
    raise ValueError("C3b retained pre-lowering SINK differs from target-artifact identity")
  serialized_sha256 = source_sink.get("serialized_sha256")
  if not isinstance(serialized_sha256, str) or len(serialized_sha256) != 64 or \
     any(char not in "0123456789abcdef" for char in serialized_sha256):
    raise ValueError("C3b retained pre-lowering SINK serialized identity is malformed")
  return {
    "artifact_schema": artifact_schema,
    "retention": "RETAINED_AND_IDENTITY_BOUND",
    "retained": True, "identity_bound": True, "sink_key": sink_identity,
    "serialized_sink_sha256": serialized_sha256,
    "evaluated_by_this_c3b_certificate": False, "c3a_claimed": False,
  }


def _expected_c3a_source_rows(role_spec: ExactRoleSpec) -> list[dict[str, Any]]:
  expected, rows = _expected_source_counters(role_spec, 0), []
  for kind, slot in sorted(expected):
    elements = role_spec.program.abi_elements[slot]
    if slot == 0:
      digest = hashlib.sha256(b"tinygrad.c3.exact_once.v1\0" + bytes([1]) * elements).hexdigest()
      row = {
        "accesses": elements, "unique_elements": elements,
        "min_element": 0, "max_element": elements-1, "allocation_elements": elements,
        "coverage_sha256": digest, "unique_coverage_sha256": digest,
      }
    else:
      counter = expected[(kind, slot)]
      if counter is None: raise AssertionError("C3a internal expected input counter is absent")
      row = _counter_row(counter, elements=elements)
    rows.append({"kind": kind, "slot": slot, "name": MEMORY_ABI_NAMES[slot], **row})
  return rows


def _validate_c3a_source_certificate(role_spec: ExactRoleSpec, sink: Any,
                                     certificate: Mapping[str, Any]) -> None:
  expected_keys = {
    "authority", "epoch", "sink_key", "full_grid", "local_size", "rows",
    "output_read_modify_write_complete_once",
  }
  if not isinstance(certificate, Mapping) or set(certificate) != expected_keys:
    raise ValueError("C3a source-SINK certificate fields differ")
  if certificate.get("authority") != "retained_pre_lowering_sink" or certificate.get("epoch") != 0 or \
     certificate.get("sink_key") != sink.key.hex():
    raise ValueError("C3a source-SINK authority, epoch, or retained identity differs")
  if certificate.get("full_grid") != list(role_spec.program.grid) or \
     certificate.get("local_size") != [256, 1, 1]:
    raise ValueError("C3a source-SINK grid or local size differs")
  if certificate.get("rows") != _expected_c3a_source_rows(role_spec):
    raise ValueError("C3a source-SINK exact six-row coverage census differs")
  if certificate.get("output_read_modify_write_complete_once") is not True:
    raise ValueError("C3a source-SINK output read/modify/write completeness differs")


def certify_frozen_staged_memory(family: FrozenStagedFamily) -> dict[str, Any]:
  """Exhaustively certify the one final-native compact PROGRAM over its full grid."""
  manifest, binding = _validated_family(family)
  role_spec, program = binding.role_spec, binding.artifact.program
  compact_spec = ExactRoleSpec(
    role_spec.role, role_spec.m, role_spec.n, EPOCH_K,
    role_spec.candidate_canonical_identity)
  if tuple(program.arg.outs) != (0,) or tuple(program.arg.ins) != tuple(range(5)):
    raise ValueError("C3 staged PROGRAM lost its exact in-place five-buffer effects")
  if manifest["program"].get("dispatch_count") != role_spec.epochs or \
     manifest["program"].get("program_count") != 1:
    raise ValueError("C3 staged family does not reuse exactly one PROGRAM for every epoch")

  identity = _program_payload_identity(manifest, binding)
  abi = _abi_certificate(compact_spec, manifest)
  native = certify_native_program_memory(compact_spec, program, 0)
  arithmetic = native.get("native_address_arithmetic")
  if not isinstance(arithmetic, Mapping) or \
     arithmetic.get("all_intermediates_within_uint32_without_overflow_or_wrap") is not True:
    raise ValueError("C3 staged native address arithmetic lacks a no-overflow/no-wrap proof")
  if native.get("all_native_effective_addresses_within_declared_allocations") is not True or \
     native.get("all_native_global_bases_resolve_to_five_buffer_kernarg_slots") is not True or \
     native.get("output_read_modify_write_complete_once") is not True:
    raise ValueError("C3 staged native address certificate did not pass")

  body = {
    "schema": SCHEMA, "state": "PASS", "gate": "C3b_final_native", "cpu_only": True,
    "family_identity": family.family_identity,
    "role": {
      "name": role_spec.role, "shape": list(role_spec.shape), "epochs": role_spec.epochs,
      "candidate_identity": role_spec.candidate_canonical_identity,
    },
    "compact_program": {
      "shape": list(compact_spec.shape), "grid": list(compact_spec.program.grid),
      "local_size": [256, 1, 1], "program_count": 1,
      "dispatch_count": role_spec.epochs,
      "workgroups": prod(compact_spec.program.grid),
      "workitems_exhaustively_evaluated": prod(compact_spec.program.grid) * 256,
      **identity,
    },
    "abi": abi,
    "final_native": native,
    "pre_lowering_source_sink_scope": _pre_lowering_sink_scope(manifest, binding),
    "proofs": {
      "complete_declared_grid_exhaustively_evaluated": True,
      "all_input_load_and_output_rmw_addresses_in_bounds": True,
      "all_native_address_arithmetic_without_overflow_or_wrap": True,
      "frozen_family_and_program_payload_identity_bound": True,
    },
  }
  return {**body, "certificate_sha256": _sha256(_canonical_bytes(body))}


def certify_frozen_staged_source_memory(family: FrozenStagedFamily) -> dict[str, Any]:
  """Exhaustively certify the retained compact pre-lowering SINK (C3a only)."""
  manifest, binding = _validated_family(family)
  role_spec, sink = binding.role_spec, binding.artifact.sink
  compact_spec = ExactRoleSpec(
    role_spec.role, role_spec.m, role_spec.n, EPOCH_K,
    role_spec.candidate_canonical_identity)
  retained = _pre_lowering_sink_scope(manifest, binding)
  if retained["retention"] != "RETAINED_AND_IDENTITY_BOUND":
    raise ValueError("C3a requires a v2 retained and identity-bound pre-lowering SINK")
  source = certify_source_sink_memory(compact_spec, sink, 0)
  _validate_c3a_source_certificate(compact_spec, sink, source)
  body = {
    "schema": C3A_SCHEMA, "state": "PASS", "gate": "C3a_source_sink", "cpu_only": True,
    "family_identity": family.family_identity,
    "role": {
      "name": role_spec.role, "shape": list(role_spec.shape), "epochs": role_spec.epochs,
      "candidate_identity": role_spec.candidate_canonical_identity,
    },
    "compact_program_shape": list(compact_spec.shape),
    "retained_source_sink": {
      **{key: value for key, value in retained.items()
         if key not in ("evaluated_by_this_c3b_certificate", "c3a_claimed")},
      "evaluated_by_this_c3a_certificate": True,
      "c3a_claimed": True,
    },
    "source_sink": source,
    "proofs": {
      "complete_declared_grid_exhaustively_evaluated": True,
      "all_logical_input_load_and_output_rmw_addresses_in_bounds": True,
      "output_read_modify_write_complete_once": True,
      "retained_sink_identity_bound_to_frozen_family": True,
    },
  }
  return {**body, "certificate_sha256": _sha256(_canonical_bytes(body))}


def _validate_child_certificate(certificate: Mapping[str, Any], *, schema: str, gate: str,
                                family_identity: str) -> None:
  if not isinstance(certificate, Mapping) or certificate.get("schema") != schema or \
     certificate.get("state") != "PASS" or certificate.get("gate") != gate or \
     certificate.get("cpu_only") is not True or certificate.get("family_identity") != family_identity:
    raise ValueError(f"full C3 {gate} child certificate identity or PASS state differs")
  digest = certificate.get("certificate_sha256")
  body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
  if not isinstance(digest, str) or digest != _sha256(_canonical_bytes(body)):
    raise ValueError(f"full C3 {gate} child certificate content digest differs")


def certify_frozen_staged_full_memory(family: FrozenStagedFamily) -> dict[str, Any]:
  """Compose independently evaluated C3a source and C3b final-native proofs."""
  c3a = certify_frozen_staged_source_memory(family)
  c3b = certify_frozen_staged_memory(family)
  _validate_child_certificate(
    c3a, schema=C3A_SCHEMA, gate="C3a_source_sink", family_identity=family.family_identity)
  _validate_child_certificate(
    c3b, schema=SCHEMA, gate="C3b_final_native", family_identity=family.family_identity)
  source_identity, native_scope = c3a["retained_source_sink"], c3b["pre_lowering_source_sink_scope"]
  if source_identity.get("sink_key") != native_scope.get("sink_key") or \
     source_identity.get("serialized_sink_sha256") != native_scope.get("serialized_sink_sha256") or \
     native_scope.get("evaluated_by_this_c3b_certificate") is not False or \
     native_scope.get("c3a_claimed") is not False:
    raise ValueError("full C3 retained source-SINK identity differs between C3a and C3b")
  body = {
    "schema": FULL_C3_SCHEMA, "state": "PASS", "gate": "C3_full", "cpu_only": True,
    "family_identity": family.family_identity,
    "gates": {"C3a": "PASS", "C3b": "PASS", "C3": "PASS"},
    "identity_binding": {
      "sink_key": source_identity["sink_key"],
      "serialized_sink_sha256": source_identity["serialized_sink_sha256"],
      "program_key": c3b["compact_program"]["program_key"],
      "source_sha256": c3b["compact_program"]["source_sha256"],
      "binary_sha256": c3b["compact_program"]["binary_sha256"],
      "serialized_program_sha256": c3b["compact_program"]["serialized_program_sha256"],
    },
    "c3a": c3a, "c3b": c3b,
  }
  return {**body, "certificate_sha256": _sha256(_canonical_bytes(body))}


__all__ = [
  "C3A_SCHEMA", "FULL_C3_SCHEMA", "SCHEMA",
  "certify_frozen_staged_full_memory", "certify_frozen_staged_memory",
  "certify_frozen_staged_source_memory",
]
