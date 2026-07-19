from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from extra.qk.mmq_compile_evidence import COMPILER_ENV
from extra.qk.mmq_exact_role_spec import ExactRoleSpec, exact_role_spec
from extra.qk.mmq_frozen_target_artifact import FrozenTargetArtifact
from extra.qk.mmq_frozen_staged_family import (
  PROVENANCE_SCHEMA, SCHEMA, build_staged_family_provenance,
  load_frozen_staged_family_manifest, produce_frozen_staged_family_manifest,
)
from extra.qk.prefill.frozen_exact_role_runtime import (
  ABI_DTYPES, ABI_NAMES, FrozenExactRoleBinding,
)
from test.unit.test_frozen_exact_role_runtime import _artifact


def _staged_artifact(role_spec: ExactRoleSpec) -> FrozenTargetArtifact:
  artifact = _artifact(role_spec)
  program = artifact.program.replace(arg=replace(
    artifact.program.arg, outs=(0,), ins=tuple(range(5))))
  manifest = copy.deepcopy(artifact.manifest)
  manifest["program"].update({
    "function": program.arg.function_name, "key": program.key.hex(),
    "device": "AMD", "compile_target": "AMD:ISA:gfx1100",
  })
  manifest["compiler_environment"] = {key: None for key in COMPILER_ENV}
  manifest["files"] = {"fixture.json": {"sha256": "c" * 64, "nbytes": 1}}
  manifest["artifacts"]["serialized_program_sha256"] = "d" * 64
  return FrozenTargetArtifact(
    manifest, program, artifact.binary, artifact.source, artifact.disassembly, artifact.fixture)


def _binding(role_spec: ExactRoleSpec, artifact: FrozenTargetArtifact | None = None,
             artifact_role_spec: ExactRoleSpec | None = None) -> FrozenExactRoleBinding:
  artifact = _staged_artifact(role_spec) if artifact is None else artifact
  return FrozenExactRoleBinding(
    role_spec, role_spec if artifact_role_spec is None else artifact_role_spec,
    artifact, role_spec.candidate_canonical_identity,
    artifact.program.key.hex(), hashlib.sha256(artifact.source.encode()).hexdigest(),
    hashlib.sha256(artifact.binary).hexdigest())


def _loader(binding: FrozenExactRoleBinding):
  def load(role_spec, path, *, inventory):
    _ = path, inventory
    assert role_spec == binding.role_spec
    return binding
  return load


def _provenance(binding: FrozenExactRoleBinding) -> dict:
  return build_staged_family_provenance(
    binding, repository_revision="a" * 40, repository_dirty=False,
    search_generator="extra.qk.machine_search.test_generator",
    search_configuration_identity="b" * 64,
    python_toolchain="CPython test", renderer_toolchain="AMDISARenderer test",
    assembler_toolchain="assemble_linear test",
    compiler_environment={key: None for key in COMPILER_ENV})


def _produce(tmp_path: Path, role: str = "attn_qo"):
  role_spec = exact_role_spec(role)
  binding = _binding(role_spec)
  output = tmp_path / f"{role}-staged-family.json"
  manifest = produce_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle",
    provenance=_provenance(binding), binding_loader=_loader(binding))
  return role_spec, binding, output, manifest


def test_staged_family_serializes_one_compact_program_and_exact_epoch_mappings(tmp_path: Path):
  role_spec, binding, output, manifest = _produce(tmp_path)
  loaded = load_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle", binding_loader=_loader(binding))

  assert loaded.family_identity == manifest["family_identity"]
  assert manifest["schema"] == SCHEMA and manifest["state"] == "FROZEN"
  assert manifest["role"] == {
    "name": "attn_qo", "shape": [512, 5120, 5120], "epoch_k": 256,
    "epoch_count": 20, "candidate_identity": role_spec.candidate_canonical_identity,
  }
  assert manifest["program"]["program_count"] == 1
  assert manifest["program"]["dispatch_count"] == role_spec.epochs
  assert manifest["program"]["key"] == binding.program_key
  assert manifest["program"]["grid"] == [40, 4, 1]
  assert manifest["program"]["local_size"] == [256, 1, 1]
  assert [row["slot"] for row in manifest["program"]["abi"]] == list(range(5))
  assert [row["name"] for row in manifest["program"]["abi"]] == list(ABI_NAMES)
  assert [row["nbytes"] for row in manifest["program"]["abi"]] == [
    elements * dtype.itemsize
    for elements, dtype in zip(role_spec.program.abi_elements, ABI_DTYPES)
  ]
  assert manifest["program"]["abi"][0]["direction"] == "inout"
  assert all(row["direction"] == "in" for row in manifest["program"]["abi"][1:])

  staging = manifest["staging"]
  assert staging["epoch_domain"] == {"start": 0, "stop_exclusive": 20, "count": 20}
  assert staging["fixed_destination_allocations"] is True
  assert staging["overwrite_requires_prior_target_completion"] is True
  q4, values, scales, sums = staging["inputs"]
  assert q4["source"]["shape"] == [5120, 20, 36]
  assert q4["stage"]["shape"] == [5120, 36]
  assert q4["mapping"]["source_index"] == "[n,e,word]"
  assert values["source"]["shape"] == [40, 512, 128] and values["stage"]["shape"] == [2, 512, 128]
  assert scales["source"]["shape"] == sums["source"]["shape"] == [40, 512, 4]
  assert scales["stage"]["shape"] == sums["stage"]["shape"] == [2, 512, 4]
  assert staging["output_recurrence"]["program_effects"] == {"outs": [0], "ins": [0, 1, 2, 3, 4]}
  assert staging["output_recurrence"]["final"] == "output[20]"
  assert manifest["queue_modes"] == {
    "eligible": ["PM4", "AQL"], "qualification": "separate_per_mode",
    "certification_inherited": False,
  }


def test_staged_family_supports_shared_n5120_program_with_distinct_ffn_down_role(tmp_path: Path):
  qo, down = exact_role_spec("attn_qo"), exact_role_spec("ffn_down")
  artifact = _staged_artifact(qo)
  binding = _binding(down, artifact, qo)
  output = tmp_path / "ffn-down.json"
  manifest = produce_frozen_staged_family_manifest(
    output, role_spec=down, frozen_bundle="/frozen/n5120",
    provenance=_provenance(binding), binding_loader=_loader(binding))
  assert manifest["artifact"]["artifact_role"] == "attn_qo"
  assert manifest["artifact"]["shared_program_geometry"] is True
  assert manifest["role"]["name"] == "ffn_down" and manifest["role"]["epoch_count"] == 68
  assert manifest["program"]["dispatch_count"] == 68
  assert manifest["staging"]["inputs"][0]["source"]["shape"] == [5120, 68, 36]
  assert manifest["staging"]["output_recurrence"]["final"] == "output[68]"


def test_staged_family_rejects_missing_or_mismatched_generation_provenance(tmp_path: Path):
  role_spec = exact_role_spec("attn_qo")
  binding = _binding(role_spec)
  provenance = _provenance(binding)
  assert provenance["schema"] == PROVENANCE_SCHEMA
  assert set(provenance["compiler"]["environment"]) == set(COMPILER_ENV)

  missing = copy.deepcopy(provenance)
  missing["toolchain"].pop("assembler")
  with pytest.raises(ValueError, match="toolchain provenance fields differ"):
    produce_frozen_staged_family_manifest(
      tmp_path / "missing.json", role_spec=role_spec, frozen_bundle="/frozen/bundle",
      provenance=missing, binding_loader=_loader(binding))

  mismatched = copy.deepcopy(provenance)
  mismatched["search"]["candidate_identity"] = "0" * 64
  with pytest.raises(ValueError, match="candidate identity differs"):
    produce_frozen_staged_family_manifest(
      tmp_path / "mismatch.json", role_spec=role_spec, frozen_bundle="/frozen/bundle",
      provenance=mismatched, binding_loader=_loader(binding))


def test_staged_family_provenance_is_bound_to_retained_compiler_environment():
  role_spec = exact_role_spec("attn_qo")
  artifact = _staged_artifact(role_spec)
  manifest = copy.deepcopy(artifact.manifest)
  manifest["compiler_environment"] = {key: None for key in COMPILER_ENV} | {
    "PYTHONHASHSEED": "0", "REGALLOC_ADDR_REMAT": "1",
  }
  artifact = replace(artifact, manifest=manifest)
  binding = _binding(role_spec, artifact)
  kwargs = {
    "repository_revision": "a" * 40, "repository_dirty": False,
    "search_generator": "extra.qk.machine_search.test_generator",
    "search_configuration_identity": "b" * 64,
    "python_toolchain": "CPython test", "renderer_toolchain": "AMDISARenderer test",
    "assembler_toolchain": "assemble_linear test",
  }
  derived = build_staged_family_provenance(binding, **kwargs)
  assert derived["compiler"]["environment"]["PYTHONHASHSEED"] == "0"
  assert derived["compiler"]["environment"]["REGALLOC_ADDR_REMAT"] == "1"
  assert all(derived["compiler"]["environment"][key] is None for key in COMPILER_ENV
             if key not in ("PYTHONHASHSEED", "REGALLOC_ADDR_REMAT"))
  with pytest.raises(ValueError, match="compiler environment differs from the frozen artifact"):
    build_staged_family_provenance(
      binding, **kwargs, compiler_environment={key: None for key in COMPILER_ENV})


def test_staged_family_blocks_sparse_legacy_artifact_environment():
  role_spec = exact_role_spec("attn_qo")
  artifact = _staged_artifact(role_spec)
  manifest = copy.deepcopy(artifact.manifest)
  manifest["compiler_environment"] = {}
  binding = _binding(role_spec, replace(artifact, manifest=manifest))
  with pytest.raises(ValueError, match="compiler environment is incomplete"):
    build_staged_family_provenance(
      binding, repository_revision="a" * 40, repository_dirty=False,
      search_generator="extra.qk.machine_search.test_generator",
      search_configuration_identity="b" * 64,
      python_toolchain="CPython test", renderer_toolchain="AMDISARenderer test",
      assembler_toolchain="assemble_linear test")


def test_staged_family_rejects_injected_binding_payload_drift(tmp_path: Path):
  role_spec = exact_role_spec("attn_qo")
  binding = _binding(role_spec)
  forged = replace(binding, binary_sha256="0" * 64)
  with pytest.raises(ValueError, match="payload identity differs"):
    produce_frozen_staged_family_manifest(
      tmp_path / "forged.json", role_spec=role_spec, frozen_bundle="/frozen/bundle",
      provenance=_provenance(forged), binding_loader=_loader(forged))


@pytest.mark.parametrize(("field", "replacement"), [
  (("program", "grid"), [8, 4, 1]),
  (("program", "binary_sha256"), "0" * 64),
  (("staging", "inputs", 0, "source", "shape"), [5120, 1, 36]),
  (("staging", "output_recurrence", "program_effects", "outs"), []),
  (("queue_modes", "eligible"), ["PM4"]),
])
def test_staged_family_loader_rejects_contract_tampering(
    tmp_path: Path, field: tuple[str | int, ...], replacement):
  role_spec, binding, output, _ = _produce(tmp_path)
  manifest = json.loads(output.read_text())
  owner = manifest
  for component in field[:-1]: owner = owner[component]
  owner[field[-1]] = replacement
  output.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="contract differs"):
    load_frozen_staged_family_manifest(
      output, role_spec=role_spec, frozen_bundle="/frozen/bundle", binding_loader=_loader(binding))


def test_staged_family_loader_rejects_identity_tampering_and_existing_output(tmp_path: Path):
  role_spec, binding, output, _ = _produce(tmp_path)
  with pytest.raises(FileExistsError, match="already exists"):
    produce_frozen_staged_family_manifest(
      output, role_spec=role_spec, frozen_bundle="/frozen/bundle",
      provenance=_provenance(binding), binding_loader=_loader(binding))
  manifest = json.loads(output.read_text())
  manifest["family_identity"] = "sha256:" + "0" * 64
  output.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="content identity differs"):
    load_frozen_staged_family_manifest(
      output, role_spec=role_spec, frozen_bundle="/frozen/bundle", binding_loader=_loader(binding))
