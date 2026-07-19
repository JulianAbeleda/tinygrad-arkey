from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json

import pytest

from extra.qk.mmq_frozen_staged_family import FrozenStagedFamily, load_frozen_staged_family_manifest
from extra.qk import mmq_frozen_staged_memory_certificate as staged_c3
from test.unit.test_mmq_frozen_staged_family import _loader, _produce


@pytest.fixture
def family(tmp_path) -> FrozenStagedFamily:
  role_spec, binding, output, _ = _produce(tmp_path)
  loaded = load_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle",
    binding_loader=_loader(binding))
  sink = loaded.binding.artifact.sink
  artifact_manifest = {
    **loaded.binding.artifact.manifest,
    "source_sink": {
      "authority": "same_session_pre_lowering_sink_passed_to_compiler",
      "key": sink.key.hex(), "serialized_sha256": "e" * 64, "serialized_nbytes": 1,
    },
  }
  artifact = replace(loaded.binding.artifact, manifest=artifact_manifest)
  rebound = replace(loaded.binding, artifact=artifact)
  payload = staged_c3._manifest_payload(role_spec, rebound, loaded.manifest["provenance"])
  identity = f"sha256:{hashlib.sha256(staged_c3._canonical_bytes(payload)).hexdigest()}"
  return FrozenStagedFamily({**payload, "family_identity": identity}, rebound, identity)


def _native_pass(role_spec, program) -> dict:
  return {
    "authority": "retained_final_selected_native_uop_graph",
    "epoch": 0,
    "program_key": program.key.hex(),
    "global_memory_instruction_lanes": 6,
    "full_grid": list(role_spec.program.grid),
    "local_size": [256, 1, 1],
    "exhaustive_launch_coordinate_count": 40 * 4 * 256,
    "rows": [
      {"kind": "load", "slot": slot, "name": name}
      for slot, name in enumerate(("output", "q4", "q8_values", "q8_scales", "q8_original_sums"))
    ] + [{"kind": "store", "slot": 0, "name": "output"}],
    "all_native_global_bases_resolve_to_five_buffer_kernarg_slots": True,
    "all_native_effective_addresses_within_declared_allocations": True,
    "native_address_arithmetic": {
      "evaluated_binary_operations": 1,
      "minimum_binary_result": 0,
      "maximum_binary_result": 1,
      "memoized_node_reuses": 0,
      "projected_address_evaluations": 1,
      "projected_address_evaluations_reused_from_family_cache": 0,
      "count_semantics": "unit fixture",
      "uint32_minimum": 0,
      "uint32_maximum": 0xffffffff,
      "all_leaves_passthroughs_operands_and_results_range_checked_before_memoization": True,
      "all_intermediates_within_uint32_without_overflow_or_wrap": True,
    },
    "output_read_modify_write_complete_once": True,
  }


def _source_pass(role_spec, sink) -> dict:
  return {
    "authority": "retained_pre_lowering_sink", "epoch": 0,
    "sink_key": sink.key.hex(), "full_grid": list(role_spec.program.grid),
    "local_size": [256, 1, 1], "rows": staged_c3._expected_c3a_source_rows(role_spec),
    "output_read_modify_write_complete_once": True,
  }


def test_staged_c3_binds_compact_full_grid_abi_and_payload_identity(monkeypatch, family):
  seen = []

  def certify(role_spec, program, epoch):
    seen.append((role_spec, program, epoch))
    return _native_pass(role_spec, program)

  monkeypatch.setattr(staged_c3, "certify_native_program_memory", certify)
  first = staged_c3.certify_frozen_staged_memory(family)
  second = staged_c3.certify_frozen_staged_memory(family)

  assert first == second
  assert first["schema"] == staged_c3.SCHEMA and first["state"] == "PASS"
  assert first["gate"] == "C3b_final_native"
  assert first["cpu_only"] is True and first["family_identity"] == family.family_identity
  assert first["role"]["shape"] == [512, 5120, 5120] and first["role"]["epochs"] == 20
  assert first["compact_program"]["shape"] == [512, 5120, 256]
  assert first["compact_program"]["grid"] == [40, 4, 1]
  assert first["compact_program"]["workgroups"] == 160
  assert first["compact_program"]["workitems_exhaustively_evaluated"] == 40 * 4 * 256
  assert first["compact_program"]["program_key"] == family.binding.program_key
  assert first["compact_program"]["source_sha256"] == family.binding.source_sha256
  assert first["compact_program"]["binary_sha256"] == family.binding.binary_sha256
  assert first["compact_program"]["serialized_program_sha256"] == \
    family.manifest["program"]["serialized_program_sha256"]
  assert first["abi"]["kernarg_bytes"] == 40
  assert [row["slot"] for row in first["abi"]["slots"]] == list(range(5))
  assert first["abi"]["effects"] == {"outs": [0], "ins": list(range(5))}
  assert first["pre_lowering_source_sink_scope"] == {
    "artifact_schema": staged_c3.TARGET_SCHEMA,
    "retention": "RETAINED_AND_IDENTITY_BOUND",
    "retained": True, "identity_bound": True,
    "sink_key": family.binding.artifact.sink.key.hex(),
    "serialized_sink_sha256": "e" * 64,
    "evaluated_by_this_c3b_certificate": False, "c3a_claimed": False,
  }
  assert first["proofs"] == {
    "complete_declared_grid_exhaustively_evaluated": True,
    "all_input_load_and_output_rmw_addresses_in_bounds": True,
    "all_native_address_arithmetic_without_overflow_or_wrap": True,
    "frozen_family_and_program_payload_identity_bound": True,
  }
  assert all(row[0].k == 256 and row[0].epochs == 1 and row[2] == 0 for row in seen)
  body = {key: value for key, value in first.items() if key != "certificate_sha256"}
  assert first["certificate_sha256"] == hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_staged_c3_rejects_family_identity_and_native_payload_drift(monkeypatch, family):
  monkeypatch.setattr(
    staged_c3, "certify_native_program_memory",
    lambda role_spec, program, epoch: _native_pass(role_spec, program))
  forged_identity = FrozenStagedFamily(
    family.manifest, family.binding, "sha256:" + "0" * 64)
  with pytest.raises(ValueError, match="family identity"):
    staged_c3.certify_frozen_staged_memory(forged_identity)

  artifact = replace(family.binding.artifact, source="different native source")
  forged_binding = replace(family.binding, artifact=artifact)
  forged_payload = FrozenStagedFamily(family.manifest, forged_binding, family.family_identity)
  with pytest.raises(ValueError, match="source or binary payload differs"):
    staged_c3.certify_frozen_staged_memory(forged_payload)


def test_staged_c3_requires_explicit_native_no_wrap_proof(monkeypatch, family):
  native = _native_pass(family.binding.role_spec, family.binding.artifact.program)
  native["native_address_arithmetic"]["all_intermediates_within_uint32_without_overflow_or_wrap"] = False
  monkeypatch.setattr(staged_c3, "certify_native_program_memory", lambda *_args: native)
  with pytest.raises(ValueError, match="no-overflow/no-wrap"):
    staged_c3.certify_frozen_staged_memory(family)


def test_staged_c3_distinguishes_legacy_absent_sink_without_claiming_c3a(family):
  artifact = replace(
    family.binding.artifact,
    manifest={**family.binding.artifact.manifest, "schema": staged_c3.LEGACY_TARGET_SCHEMA},
    sink=None)
  binding = replace(family.binding, artifact=artifact)
  assert staged_c3._pre_lowering_sink_scope(family.manifest, binding) == {
    "artifact_schema": staged_c3.LEGACY_TARGET_SCHEMA,
    "retention": "ABSENT_LEGACY_V1_ARTIFACT",
    "retained": False, "identity_bound": False,
    "evaluated_by_this_c3b_certificate": False, "c3a_claimed": False,
  }


def test_staged_c3a_certifies_retained_compact_sink_without_native_claim(monkeypatch, family):
  seen = []

  def certify(role_spec, sink, epoch):
    seen.append((role_spec, sink, epoch))
    return _source_pass(role_spec, sink)

  monkeypatch.setattr(staged_c3, "certify_source_sink_memory", certify)
  certificate = staged_c3.certify_frozen_staged_source_memory(family)
  assert certificate["schema"] == staged_c3.C3A_SCHEMA
  assert certificate["state"] == "PASS" and certificate["gate"] == "C3a_source_sink"
  assert certificate["family_identity"] == family.family_identity
  assert certificate["compact_program_shape"] == [512, 5120, 256]
  assert certificate["retained_source_sink"] == {
    "artifact_schema": staged_c3.TARGET_SCHEMA,
    "retention": "RETAINED_AND_IDENTITY_BOUND",
    "retained": True, "identity_bound": True,
    "sink_key": family.binding.artifact.sink.key.hex(),
    "serialized_sink_sha256": "e" * 64,
    "evaluated_by_this_c3a_certificate": True, "c3a_claimed": True,
  }
  assert len(seen) == 1 and seen[0][0].k == 256 and seen[0][0].epochs == 1
  assert seen[0][1] is family.binding.artifact.sink and seen[0][2] == 0


def test_staged_full_c3_combines_content_addressed_c3a_and_c3b(monkeypatch, family):
  monkeypatch.setattr(
    staged_c3, "certify_source_sink_memory",
    lambda role_spec, sink, epoch: _source_pass(role_spec, sink))
  monkeypatch.setattr(
    staged_c3, "certify_native_program_memory",
    lambda role_spec, program, epoch: _native_pass(role_spec, program))
  certificate = staged_c3.certify_frozen_staged_full_memory(family)
  assert certificate["schema"] == staged_c3.FULL_C3_SCHEMA
  assert certificate["state"] == "PASS" and certificate["gate"] == "C3_full"
  assert certificate["gates"] == {"C3a": "PASS", "C3b": "PASS", "C3": "PASS"}
  assert certificate["identity_binding"] == {
    "sink_key": family.binding.artifact.sink.key.hex(),
    "serialized_sink_sha256": "e" * 64,
    "program_key": family.binding.program_key,
    "source_sha256": family.binding.source_sha256,
    "binary_sha256": family.binding.binary_sha256,
    "serialized_program_sha256": family.manifest["program"]["serialized_program_sha256"],
  }
  assert certificate["c3a"]["certificate_sha256"]
  assert certificate["c3b"]["certificate_sha256"]
  body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
  assert certificate["certificate_sha256"] == hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize("case", (
  "wrong_authority", "missing_authority", "wrong_epoch", "wrong_grid", "wrong_local",
  "missing_row", "wrong_name", "wrong_kind", "wrong_slot", "wrong_output_flag",
  "wrong_output_range", "wrong_coverage_digest",
))
def test_staged_c3a_rejects_tampered_source_certificate_before_proof_flags(
    monkeypatch, family, case):
  role_spec = family.binding.role_spec
  compact = replace(role_spec, k=256)
  source = _source_pass(compact, family.binding.artifact.sink)
  if case == "wrong_authority": source["authority"] = "final_native"
  elif case == "missing_authority": source.pop("authority")
  elif case == "wrong_epoch": source["epoch"] = 1
  elif case == "wrong_grid": source["full_grid"] = [8, 4, 1]
  elif case == "wrong_local": source["local_size"] = [128, 1, 1]
  elif case == "missing_row": source["rows"].pop()
  elif case == "wrong_name": source["rows"][1]["name"] = "wrong_q4"
  elif case == "wrong_kind": source["rows"][1]["kind"] = "store"
  elif case == "wrong_slot": source["rows"][1]["slot"] = 2
  elif case == "wrong_output_flag": source["output_read_modify_write_complete_once"] = False
  elif case == "wrong_output_range": source["rows"][0]["max_element"] -= 1
  elif case == "wrong_coverage_digest": source["rows"][2]["coverage_sha256"] = "0" * 64
  else: raise AssertionError(case)
  tampered = copy.deepcopy(source)
  monkeypatch.setattr(staged_c3, "certify_source_sink_memory", lambda *_args: tampered)
  with pytest.raises(ValueError, match="C3a source-SINK"):
    staged_c3.certify_frozen_staged_source_memory(family)
