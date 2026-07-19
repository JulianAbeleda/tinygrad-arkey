"""Immutable identity contract for one compact K256 PROGRAM reused across a full role.

This module is deliberately launcher-neutral. It joins an inventory-admitted
role to an existing :class:`FrozenTargetArtifact`, records the exact compact
epoch staging recurrence, and writes one content-addressed JSON manifest. It
does not compile, construct a runtime, allocate a device buffer, or dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from extra.qk.mmq_compile_evidence import COMPILER_ENV
from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, EPOCH_K, ExactRoleSpec, admit_exact_role_spec
from extra.qk.mmq_frozen_target_artifact import SCHEMA as TARGET_ARTIFACT_SCHEMA
from extra.qk.prefill.frozen_exact_role_runtime import (
  ABI_DTYPES, ABI_NAMES, PROGRAM_DEVICE, Q4_WORDS_PER_EPOCH_ROW,
  FrozenExactRoleBinding, load_frozen_exact_role_binding,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_family.v1"
PROVENANCE_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_family_provenance.v1"
STATE = "FROZEN"
QUEUE_MODES = ("PM4", "AQL")
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
  return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
  if set(value) != expected:
    raise ValueError(f"{label} fields differ: expected {sorted(expected)!r}, got {sorted(value)!r}")


def _nonempty(value: Any, *, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _digest(value: Any, *, label: str, lengths: tuple[int, ...] = (64,)) -> str:
  value = _nonempty(value, label=label)
  if len(value) not in lengths or any(char not in _HEX for char in value):
    raise ValueError(f"{label} must be a lowercase hexadecimal digest")
  return value


def _sink_identity(binding: FrozenExactRoleBinding) -> str:
  sink = binding.artifact.program.src[0]
  return sink.key.hex()


def _artifact_compiler_environment(binding: FrozenExactRoleBinding) -> dict[str, str | None]:
  retained = binding.artifact.manifest.get("compiler_environment")
  if not isinstance(retained, Mapping) or set(retained) != set(COMPILER_ENV) or \
     any(value is not None and not isinstance(value, str) for value in retained.values()):
    raise ValueError("frozen artifact compiler environment is incomplete or malformed")
  return {key: retained[key] for key in COMPILER_ENV}


def build_staged_family_provenance(
    binding: FrozenExactRoleBinding, *, repository_revision: str, repository_dirty: bool,
    search_generator: str, search_configuration_identity: str,
    python_toolchain: str, renderer_toolchain: str, assembler_toolchain: str,
    compiler_environment: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
  """Build complete generation provenance without inferring missing fields."""
  if not isinstance(binding, FrozenExactRoleBinding):
    raise TypeError("staged family provenance requires a frozen exact role binding")
  environment = (
    _artifact_compiler_environment(binding)
    if compiler_environment is None else dict(compiler_environment)
  )
  provenance = {
    "schema": PROVENANCE_SCHEMA,
    "repository": {"revision": repository_revision, "dirty": repository_dirty},
    "search": {
      "candidate_identity": binding.candidate_identity,
      "sink_identity": _sink_identity(binding),
      "generator": search_generator,
      "configuration_identity": search_configuration_identity,
    },
    "compiler": {
      "target": binding.artifact.manifest["program"]["compile_target"],
      "environment": environment,
    },
    "toolchain": {
      "python": python_toolchain,
      "renderer": renderer_toolchain,
      "assembler": assembler_toolchain,
    },
  }
  return _validate_provenance(provenance, binding)


def _validate_provenance(provenance: Mapping[str, Any], binding: FrozenExactRoleBinding) -> dict[str, Any]:
  if not isinstance(provenance, Mapping):
    raise ValueError("staged family provenance must be a mapping")
  _exact_keys(provenance, {"schema", "repository", "search", "compiler", "toolchain"}, label="provenance")
  if provenance.get("schema") != PROVENANCE_SCHEMA:
    raise ValueError("staged family provenance schema differs")

  repository = provenance["repository"]
  if not isinstance(repository, Mapping):
    raise ValueError("repository provenance must be a mapping")
  _exact_keys(repository, {"revision", "dirty"}, label="repository provenance")
  revision = _digest(repository.get("revision"), label="repository revision", lengths=(40, 64))
  if type(repository.get("dirty")) is not bool:
    raise ValueError("repository dirty provenance must be a bool")

  search = provenance["search"]
  if not isinstance(search, Mapping):
    raise ValueError("search provenance must be a mapping")
  _exact_keys(search, {"candidate_identity", "sink_identity", "generator", "configuration_identity"},
              label="search provenance")
  if _digest(search.get("candidate_identity"), label="search candidate identity") != binding.candidate_identity:
    raise ValueError("search candidate identity differs from the exact role binding")
  if _digest(search.get("sink_identity"), label="search sink identity") != _sink_identity(binding):
    raise ValueError("search sink identity differs from the frozen PROGRAM")
  generator = _nonempty(search.get("generator"), label="search generator")
  configuration_identity = _digest(search.get("configuration_identity"),
                                   label="search configuration identity")

  compiler = provenance["compiler"]
  if not isinstance(compiler, Mapping):
    raise ValueError("compiler provenance must be a mapping")
  _exact_keys(compiler, {"target", "environment"}, label="compiler provenance")
  compile_target = binding.artifact.manifest["program"]["compile_target"]
  if compiler.get("target") != compile_target:
    raise ValueError("compiler target differs from the frozen PROGRAM")
  environment = compiler.get("environment")
  if not isinstance(environment, Mapping) or set(environment) != set(COMPILER_ENV):
    raise ValueError("compiler environment must enumerate every declared codegen input")
  if any(value is not None and not isinstance(value, str) for value in environment.values()):
    raise ValueError("compiler environment values must be strings or null")
  normalized_environment = {key: environment[key] for key in COMPILER_ENV}
  if normalized_environment != _artifact_compiler_environment(binding):
    raise ValueError("compiler environment differs from the frozen artifact")

  toolchain = provenance["toolchain"]
  if not isinstance(toolchain, Mapping):
    raise ValueError("toolchain provenance must be a mapping")
  _exact_keys(toolchain, {"python", "renderer", "assembler"}, label="toolchain provenance")
  normalized_toolchain = {
    key: _nonempty(toolchain.get(key), label=f"{key} toolchain")
    for key in ("python", "renderer", "assembler")
  }
  return {
    "schema": PROVENANCE_SCHEMA,
    "repository": {"revision": revision, "dirty": repository["dirty"]},
    "search": {
      "candidate_identity": binding.candidate_identity,
      "sink_identity": _sink_identity(binding),
      "generator": generator,
      "configuration_identity": configuration_identity,
    },
    "compiler": {
      "target": compile_target,
      "environment": normalized_environment,
    },
    "toolchain": normalized_toolchain,
  }


def _abi_contract(role_spec: ExactRoleSpec, binding: FrozenExactRoleBinding) -> list[dict[str, Any]]:
  manifest_abi = binding.artifact.manifest["program"]["abi"]
  expected = [{
    "slot": slot, "name": name, "dtype": f"{dtype}.ptr({elements})", "elements": elements,
  } for slot, (name, dtype, elements) in enumerate(
    zip(ABI_NAMES, ABI_DTYPES, role_spec.program.abi_elements))]
  if manifest_abi != expected:
    raise ValueError("frozen artifact ABI differs from the exact compact staged ABI")
  return [{
    **row, "nbytes": int(row["elements"]) * dtype.itemsize,
    "direction": "inout" if slot == 0 else "in",
  } for slot, (row, dtype) in enumerate(zip(expected, ABI_DTYPES))]


def _staging_contract(role_spec: ExactRoleSpec) -> dict[str, Any]:
  m, n, epochs = role_spec.m, role_spec.n, role_spec.epochs
  return {
    "mode": "one_compact_k256_stage_reused_by_ordinal",
    "epoch_k": EPOCH_K,
    "epoch_domain": {"start": 0, "stop_exclusive": epochs, "count": epochs},
    "fixed_destination_allocations": True,
    "overwrite_requires_prior_target_completion": True,
    "inputs": [
      {
        "slot": 1, "name": ABI_NAMES[1], "dtype": str(ABI_DTYPES[1]),
        "source": {
          "layout": "q4_k_words[n,k256_epoch,36]",
          "shape": [n, epochs, Q4_WORDS_PER_EPOCH_ROW],
          "elements": n * epochs * Q4_WORDS_PER_EPOCH_ROW,
        },
        "stage": {
          "layout": "q4_k_words[n,36]",
          "shape": [n, Q4_WORDS_PER_EPOCH_ROW],
          "elements": n * Q4_WORDS_PER_EPOCH_ROW,
        },
        "mapping": {
          "epoch_variable": "e", "source_index": "[n,e,word]",
          "stage_index": "[n,word]", "copy": "all n,word",
        },
      },
      {
        "slot": 2, "name": ABI_NAMES[2], "dtype": str(ABI_DTYPES[2]),
        "source": {
          "layout": "q8_1_values[2*k256_epoch,m,128]",
          "shape": [2 * epochs, m, 128], "elements": 2 * epochs * m * 128,
        },
        "stage": {
          "layout": "q8_1_values[2,m,128]",
          "shape": [2, m, 128], "elements": 2 * m * 128,
        },
        "mapping": {
          "epoch_variable": "e", "source_index": "[2*e+r,m,k]",
          "stage_index": "[r,m,k]", "copy": "r in [0,2), all m,k",
        },
      },
      {
        "slot": 3, "name": ABI_NAMES[3], "dtype": str(ABI_DTYPES[3]),
        "source": {
          "layout": "q8_1_scales[2*k256_epoch,m,4]",
          "shape": [2 * epochs, m, 4], "elements": 2 * epochs * m * 4,
        },
        "stage": {
          "layout": "q8_1_scales[2,m,4]",
          "shape": [2, m, 4], "elements": 2 * m * 4,
        },
        "mapping": {
          "epoch_variable": "e", "source_index": "[2*e+r,m,g]",
          "stage_index": "[r,m,g]", "copy": "r in [0,2), all m,g",
        },
      },
      {
        "slot": 4, "name": ABI_NAMES[4], "dtype": str(ABI_DTYPES[4]),
        "source": {
          "layout": "q8_1_original_sums[2*k256_epoch,m,4]",
          "shape": [2 * epochs, m, 4], "elements": 2 * epochs * m * 4,
        },
        "stage": {
          "layout": "q8_1_original_sums[2,m,4]",
          "shape": [2, m, 4], "elements": 2 * m * 4,
        },
        "mapping": {
          "epoch_variable": "e", "source_index": "[2*e+r,m,g]",
          "stage_index": "[r,m,g]", "copy": "r in [0,2), all m,g",
        },
      },
    ],
    "output_recurrence": {
      "slot": 0, "initialization": "fp32_zero",
      "program_effects": {"outs": [0], "ins": [0, 1, 2, 3, 4]},
      "step": "output[e+1]=output[e]+MMQ(q4_stage,q8_values_stage,q8_scales_stage,q8_original_sums_stage)",
      "final": f"output[{epochs}]",
    },
  }


def _manifest_payload(role_spec: ExactRoleSpec, binding: FrozenExactRoleBinding,
                      provenance: Mapping[str, Any]) -> dict[str, Any]:
  artifact_manifest, program = binding.artifact.manifest, binding.artifact.program
  if tuple(program.arg.outs) != (0,) or tuple(program.arg.ins) != tuple(range(5)):
    raise ValueError("compact staged PROGRAM lost its in-place five-buffer effects")
  program_manifest = artifact_manifest["program"]
  if artifact_manifest.get("schema") != TARGET_ARTIFACT_SCHEMA:
    raise ValueError("compact staged family requires a frozen target artifact")
  abi = _abi_contract(role_spec, binding)
  return {
    "schema": SCHEMA, "state": STATE,
    "role": {
      "name": role_spec.role, "shape": list(role_spec.shape), "epoch_k": EPOCH_K,
      "epoch_count": role_spec.epochs, "candidate_identity": binding.candidate_identity,
    },
    "artifact": {
      "schema": artifact_manifest["schema"],
      "manifest_sha256": _sha256(_canonical_bytes(artifact_manifest)),
      "artifact_role": binding.artifact_role_spec.role,
      "artifact_full_role_shape": list(binding.artifact_role_spec.shape),
      "shared_program_geometry": binding.shared_program_geometry,
      "fixture_sha256": artifact_manifest["files"]["fixture.json"]["sha256"],
      "requires_recompile": False,
    },
    "program": {
      "reuse": "single_compact_k256_program_for_every_epoch",
      "program_count": 1, "dispatch_count": role_spec.epochs,
      "function": program.arg.function_name, "key": binding.program_key,
      "sink_identity": _sink_identity(binding),
      "source_sha256": binding.source_sha256, "binary_sha256": binding.binary_sha256,
      "serialized_program_sha256": artifact_manifest["artifacts"]["serialized_program_sha256"],
      "device": PROGRAM_DEVICE, "compile_target": program_manifest["compile_target"],
      "shape": list(role_spec.program.shape), "grid": list(role_spec.program.grid),
      "local_size": list(program.arg.local_size or ()), "globals": list(program.arg.globals),
      "abi": abi,
    },
    "staging": _staging_contract(role_spec),
    "queue_modes": {
      "eligible": list(QUEUE_MODES), "qualification": "separate_per_mode",
      "certification_inherited": False,
    },
    "provenance": dict(provenance),
  }


@dataclass(frozen=True)
class FrozenStagedFamily:
  manifest: Mapping[str, Any]
  binding: FrozenExactRoleBinding
  family_identity: str


def _binding(role_spec: ExactRoleSpec, frozen_bundle: str | Path, *,
             inventory: str | Path | Mapping[str, Any],
             binding_loader: Callable[..., FrozenExactRoleBinding]) -> FrozenExactRoleBinding:
  binding = binding_loader(role_spec, frozen_bundle, inventory=inventory)
  if not isinstance(binding, FrozenExactRoleBinding) or binding.role_spec != role_spec:
    raise ValueError("staged family loader returned a mismatched frozen exact role binding")
  if binding.candidate_identity != role_spec.candidate_canonical_identity:
    raise ValueError("staged family binding candidate identity differs from the exact role")
  if not binding.shared_program_geometry:
    raise ValueError("staged family artifact does not share the exact compact program geometry")
  artifact, program = binding.artifact, binding.artifact.program
  manifest = artifact.manifest
  if not isinstance(manifest, Mapping) or manifest.get("schema") != TARGET_ARTIFACT_SCHEMA or \
     manifest.get("state") != STATE:
    raise ValueError("staged family binding is not a frozen target artifact")
  program_manifest, artifacts, files = manifest.get("program"), manifest.get("artifacts"), manifest.get("files")
  if not all(isinstance(value, Mapping) for value in (program_manifest, artifacts, files)):
    raise ValueError("staged family frozen artifact manifest is incomplete")
  fixture = files.get("fixture.json")
  if not isinstance(fixture, Mapping) or not isinstance(fixture.get("sha256"), str):
    raise ValueError("staged family frozen artifact fixture identity is missing")
  if program.arg.function_name != program_manifest.get("function") or \
     program.key.hex() != program_manifest.get("key") or binding.program_key != program.key.hex() or \
     program_manifest.get("device") != PROGRAM_DEVICE or \
     tuple(program.arg.global_size) != role_spec.program.grid or \
     list(program.arg.global_size) != program_manifest.get("global_size") or \
     tuple(program.arg.local_size or ()) != (256, 1, 1) or \
     list(program.arg.local_size or ()) != program_manifest.get("local_size") or \
     tuple(program.arg.globals) != tuple(range(5)) or \
     list(program.arg.globals) != program_manifest.get("globals"):
    raise ValueError("staged family frozen PROGRAM identity or launch geometry differs")
  source_sha256, binary_sha256 = _sha256(artifact.source.encode()), _sha256(artifact.binary)
  if binding.source_sha256 != source_sha256 or binding.binary_sha256 != binary_sha256 or \
     artifacts.get("source_sha256") != source_sha256 or artifacts.get("binary_sha256") != binary_sha256 or \
     not isinstance(artifacts.get("serialized_program_sha256"), str):
    raise ValueError("staged family frozen PROGRAM payload identity differs")
  return binding


def produce_frozen_staged_family_manifest(
    output: str | Path, *, role_spec: ExactRoleSpec, frozen_bundle: str | Path,
    provenance: Mapping[str, Any],
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    binding_loader: Callable[..., FrozenExactRoleBinding] = load_frozen_exact_role_binding,
    ) -> dict[str, Any]:
  """Validate existing frozen bytes and atomically publish their staged-family identity."""
  role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
  binding = _binding(role_spec, frozen_bundle, inventory=inventory, binding_loader=binding_loader)
  normalized_provenance = _validate_provenance(provenance, binding)
  payload = _manifest_payload(role_spec, binding, normalized_provenance)
  manifest = {**payload, "family_identity": f"sha256:{_sha256(_canonical_bytes(payload))}"}
  path = Path(output)
  if path.exists():
    raise FileExistsError(f"staged family manifest already exists: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
  with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(encoded)
  try:
    load_frozen_staged_family_manifest(
      temporary, role_spec=role_spec, frozen_bundle=frozen_bundle, inventory=inventory,
      binding_loader=binding_loader)
    # Same-directory hard-link publication is atomic and refuses to replace an
    # existing immutable manifest, including a concurrent producer's result.
    os.link(temporary, path)
    temporary.unlink()
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
  return manifest


def load_frozen_staged_family_manifest(
    path: str | Path, *, role_spec: ExactRoleSpec, frozen_bundle: str | Path,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    binding_loader: Callable[..., FrozenExactRoleBinding] = load_frozen_exact_role_binding,
    ) -> FrozenStagedFamily:
  """Fail closed unless JSON, role, artifact, PROGRAM, staging, and provenance all agree."""
  role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
  try: manifest = json.loads(Path(path).read_text())
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError(f"staged family manifest cannot be read: {type(exc).__name__}: {exc}") from exc
  if not isinstance(manifest, dict):
    raise ValueError("staged family manifest must be a JSON object")
  _exact_keys(manifest, {
    "schema", "state", "family_identity", "role", "artifact", "program",
    "staging", "queue_modes", "provenance",
  }, label="staged family manifest")
  if manifest.get("schema") != SCHEMA or manifest.get("state") != STATE:
    raise ValueError("staged family schema or frozen state differs")
  binding = _binding(role_spec, frozen_bundle, inventory=inventory, binding_loader=binding_loader)
  provenance = _validate_provenance(manifest["provenance"], binding)
  expected = _manifest_payload(role_spec, binding, provenance)
  observed_payload = {key: value for key, value in manifest.items() if key != "family_identity"}
  if observed_payload != expected:
    raise ValueError("staged family contract differs from the exact role or frozen artifact")
  identity = f"sha256:{_sha256(_canonical_bytes(expected))}"
  if manifest.get("family_identity") != identity:
    raise ValueError("staged family content identity differs")
  return FrozenStagedFamily(manifest, binding, identity)


__all__ = [
  "PROVENANCE_SCHEMA", "QUEUE_MODES", "SCHEMA", "STATE", "FrozenStagedFamily",
  "build_staged_family_provenance", "load_frozen_staged_family_manifest",
  "produce_frozen_staged_family_manifest",
]
