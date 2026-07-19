"""CPU-only producer and loader for the frozen target in-place MMQ PROGRAM.

The bundle is launcher-neutral: it retains the exact generated PROGRAM, native
AMD code object, source, disassembly, launch ABI, and deterministic input
identity without constructing an AMD runtime.  Consumers load and validate the
serialized PROGRAM directly; they never need to invoke the compiler again.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tarfile
import tempfile
from typing import Any, Callable, Mapping

from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_compile_evidence import COMPILER_ENV
from extra.qk.mmq_exact_role_spec import (
  DEFAULT_EXACT_ROLE_SPEC, DEFAULT_INVENTORY, ExactProgramGeometry, ExactRoleSpec,
  exact_role_spec, exact_role_spec_from_shape,
)
from extra.qk.mmq_llama_five_buffer_full_kernel import AMD_ISA_TARGET
from extra.qk.mmq_target_epoch_orchestrator import (
  FIXTURE_SCHEMA, _validated_target_kernel_compile_proof, compile_target_kernel,
  target_fixture_evidence, target_program_artifact_evidence,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_target_artifact.v2"
LEGACY_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_target_artifact.v1"
AUDIT_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_target_artifact_audit.v1"
TARGET_SHAPE = DEFAULT_EXACT_ROLE_SPEC.program.shape
FULL_ROLE_SHAPE = DEFAULT_EXACT_ROLE_SPEC.shape
FUNCTION_NAME = "mmq_llama_five_buffer_full_grid_accumulate"
BACKEND_ID = "q4k_q8_1_mmq_amd_isa_full_grid_v0"
ACCUMULATION = "target_in_place_fp32_add"
PROGRAM_DEVICE = "AMD"
FILE_NAMES = {
  "binary": "target_accumulate_k256.hsaco",
  "program": "target_accumulate_k256.program.pkl",
  "sink": "target_accumulate_k256.sink.pkl",
  "source": "target_accumulate_k256.source.txt",
  "disassembly": "target_accumulate_k256.isa.txt",
  "fixture": "fixture.json",
}
EXPECTED_ABI = (
  {"slot": 0, "name": "output", "dtype": "dtypes.float.ptr(8912896)", "elements": 8_912_896},
  {"slot": 1, "name": "q4", "dtype": "dtypes.uint.ptr(626688)", "elements": 626_688},
  {"slot": 2, "name": "q8_values", "dtype": "dtypes.char.ptr(131072)", "elements": 131_072},
  {"slot": 3, "name": "q8_scales", "dtype": "dtypes.float.ptr(4096)", "elements": 4_096},
  {"slot": 4, "name": "q8_original_sums", "dtype": "dtypes.float.ptr(4096)", "elements": 4_096},
)


def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _program_payload(program: UOp) -> tuple[bytes, str, dict[str, Any]]:
  binaries = [u.arg for u in program.src if u.op is Ops.BINARY]
  sources = [u.arg for u in program.src if u.op is Ops.SOURCE]
  if len(binaries) != 1 or not isinstance(binaries[0], bytes) or not binaries[0]:
    raise ValueError("PROGRAM must retain exactly one nonempty BINARY")
  if len(sources) != 1 or not isinstance(sources[0], str) or not sources[0]:
    raise ValueError("PROGRAM must retain exactly one nonempty SOURCE")
  binary, source, evidence = target_program_artifact_evidence(program)
  if binary != binaries[0] or source != sources[0]:
    raise ValueError("shared artifact capture differs from retained PROGRAM payload")
  if not isinstance(evidence.get("resources"), dict):
    raise ValueError("shared artifact capture lacks authoritative resources")
  return binary, source, {
    "source_sha256": evidence["source_sha256"], "source_nbytes": evidence["source_nbytes"],
    "binary_sha256": evidence["binary_sha256"], "binary_nbytes": evidence["binary_nbytes"],
    "resources": evidence.get("resources"),
  }


def _expected_abi(geometry: ExactProgramGeometry) -> tuple[dict[str, Any], ...]:
  names = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")
  dtypes = ("float", "uint", "char", "float", "float")
  return tuple({"slot": slot, "name": name, "dtype": f"dtypes.{dtype}.ptr({elements})", "elements": elements}
               for slot, (name, dtype, elements) in enumerate(zip(names, dtypes, geometry.abi_elements)))


def _abi(program: UOp, geometry: ExactProgramGeometry = DEFAULT_EXACT_ROLE_SPEC.program) -> list[dict[str, Any]]:
  params = sorted({u for u in program.src[0].toposort() if u.op is Ops.PARAM}, key=lambda u: u.arg.slot)
  if [u.arg.slot for u in params] != list(range(5)): raise ValueError("target PROGRAM must expose PARAM slots 0..4")
  expected = _expected_abi(geometry)
  rows = [{"slot": int(u.arg.slot), "name": expected[u.arg.slot]["name"],
           "dtype": str(u.dtype), "elements": int(u.max_numel())} for u in params]
  if tuple(rows) != expected: raise ValueError(f"target PROGRAM ABI changed: {rows}")
  return rows


def deterministic_fixture_identity(*, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC) -> dict[str, Any]:
  """Return the orchestrator's existing deterministic full-role identity."""
  return target_fixture_evidence(role_spec=role_spec)


def _default_compile_once(role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC) -> Any:
  return compile_target_kernel(accumulate=True, target=AMD_ISA_TARGET, role_spec=role_spec)


def _validate_source_sink(sink: Any, role_spec: ExactRoleSpec) -> UOp:
  """Validate the retained pre-lowering authority without compiling it."""
  if not isinstance(sink, UOp) or sink.op is not Ops.SINK:
    raise ValueError("target compile result lacks a pre-lowering SINK")
  params = sorted({u for u in sink.toposort() if u.op is Ops.PARAM}, key=lambda u: u.arg.slot)
  expected = _expected_abi(role_spec.program)
  observed = [{"slot": int(u.arg.slot), "name": expected[u.arg.slot]["name"],
               "dtype": str(u.dtype), "elements": int(u.max_numel())} for u in params]
  if observed != list(expected):
    raise ValueError("pre-lowering SINK five-buffer ABI changed")
  return sink


def _program_disassembly(program: UOp, binary: bytes) -> tuple[str, str]:
  """Recreate the renderer's final typed stream and bind it byte-for-byte to the HSACO."""
  from tinygrad.helpers import Target
  from tinygrad.renderer.amd.elf import assemble_linear
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.runtime.support.elf import elf_loader

  if program.src[1].op is not Ops.DEVICE or program.src[1].arg != PROGRAM_DEVICE:
    raise ValueError("frozen target PROGRAM is not an AMD:ISA program")
  renderer = AMDISARenderer(Target.parse(AMD_ISA_TARGET))
  final_linear = renderer._final_linear(program.src[2])
  proof = program.src[2].arg
  assembly_program = renderer._assembly_program(program, proof)
  rebuilt = assemble_linear(assembly_program, final_linear, renderer.target.arch)
  if rebuilt != binary: raise ValueError("renderer final stream does not reproduce retained HSACO")
  _, sections, _ = elf_loader(binary)
  text = next((section for section in sections if section.name == ".text"), None)
  if text is None: raise ValueError("retained HSACO has no .text section")
  disassembly = renderer._final_disassembly(final_linear, start_pc=int(text.header.sh_addr))
  return disassembly, "renderer-final-stream-byte-reassembled"


def _validate_program(program: Any, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC) -> UOp:
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("compile result is not a PROGRAM")
  programs = [u for u in program.toposort() if u.op is Ops.PROGRAM]
  if programs != [program]: raise ValueError(f"expected one closed PROGRAM, found {len(programs)}")
  if program.arg.function_name != FUNCTION_NAME:
    raise ValueError(f"target function changed: {program.arg.function_name}")
  if tuple(program.arg.globals) != tuple(range(5)): raise ValueError(f"target globals changed: {program.arg.globals}")
  if tuple(program.arg.global_size) != role_spec.program.grid:
    raise ValueError(f"target grid changed: {program.arg.global_size}")
  if tuple(program.arg.local_size or ()) != (256, 1, 1): raise ValueError(f"target local size changed: {program.arg.local_size}")
  if program.src[1].op is not Ops.DEVICE or program.src[1].arg != PROGRAM_DEVICE:
    raise ValueError(f"target PROGRAM device changed: {program.src[1]}")
  _abi(program, role_spec.program)
  _program_payload(program)
  return program


def _inventory(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
  return {name: {"sha256": _sha256(data), "nbytes": len(data)} for name, data in sorted(files.items())}


def _write_archive(directory: Path, archive: Path) -> None:
  if archive.exists(): raise FileExistsError(f"archive already exists: {archive}")
  archive.parent.mkdir(parents=True, exist_ok=True)
  with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tf:
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
      data = path.read_bytes()
      info = tarfile.TarInfo(path.name)
      info.size, info.mtime, info.mode = len(data), 0, 0o644
      info.uid = info.gid = 0
      info.uname = info.gname = ""
      from io import BytesIO
      tf.addfile(info, BytesIO(data))


def produce_frozen_target_artifact(output_dir: str | Path, *, archive: str | Path | None = None,
                                   role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC,
                                   compile_once: Callable[[], Any] | None = None,
                                   disassemble: Callable[[bytes], tuple[str, str]] | None = None,
                                   fixture_builder: Callable[[], dict[str, Any]] | None = None,
                                   ) -> dict[str, Any]:
  """Compile exactly once and atomically produce a self-validating CPU-only bundle."""
  output = Path(output_dir)
  if output.exists(): raise FileExistsError(f"output already exists: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  try:
    selected_compile = compile_once or (lambda: _default_compile_once(role_spec))
    selected_fixture = fixture_builder or (lambda: deterministic_fixture_identity(role_spec=role_spec))
    compiled = selected_compile()  # sole compile invocation owned by this producer
    proof = _validated_target_kernel_compile_proof(compiled)
    sink = _validate_source_sink(proof.sink, role_spec)
    program = _validate_program(proof.program, role_spec)
    binary, source, shared_artifacts = _program_payload(program)
    disassembly, disassembly_tool = (
      _program_disassembly(program, binary) if disassemble is None else disassemble(binary))
    if not isinstance(disassembly, str) or not disassembly.strip():
      raise ValueError("AMDGPU disassembly is empty")
    fixture = selected_fixture()
    if not isinstance(fixture, dict) or fixture.get("schema") != FIXTURE_SCHEMA:
      raise ValueError("fixture builder returned the wrong schema")
    if tuple(fixture.get("shape", ())) != role_spec.shape:
      raise ValueError("fixture full-role shape differs from admitted role")
    if "role" in fixture and fixture["role"] != role_spec.role:
      raise ValueError("fixture role differs from admitted role")
    serialized_sink = pickle.dumps(sink, protocol=pickle.HIGHEST_PROTOCOL)
    serialized = pickle.dumps(program, protocol=pickle.HIGHEST_PROTOCOL)
    files = {
      FILE_NAMES["binary"]: binary,
      FILE_NAMES["program"]: serialized,
      FILE_NAMES["sink"]: serialized_sink,
      FILE_NAMES["source"]: source.encode(),
      FILE_NAMES["disassembly"]: disassembly.encode(),
      FILE_NAMES["fixture"]: _json_bytes(fixture),
    }
    manifest = {
      "schema": SCHEMA, "state": "FROZEN", "compile_calls": 1,
      "compile_only_cpu": True, "gpu_runtime_initialized": False, "gpu_dispatch_performed": False,
      "compiler_environment": {
        key: os.environ.get(key) for key in COMPILER_ENV
      },
      "backend_id": BACKEND_ID, "accumulation": ACCUMULATION, "accumulate": True,
      "shape": list(role_spec.program.shape), "full_role_shape": list(role_spec.shape),
      "program": {
        "function": program.arg.function_name, "key": program.key.hex(),
        "device": program.src[1].arg, "compile_target": AMD_ISA_TARGET,
        "globals": list(program.arg.globals), "global_size": list(program.arg.global_size),
        "local_size": list(program.arg.local_size or ()), "abi": _abi(program, role_spec.program),
      },
      "source_sink": {
        "authority": "same_session_pre_lowering_sink_passed_to_compiler",
        "key": sink.key.hex(),
        "serialized_sha256": _sha256(serialized_sink),
        "serialized_nbytes": len(serialized_sink),
      },
      "artifacts": {
        **shared_artifacts,
        "serialized_program_sha256": _sha256(serialized), "serialized_program_nbytes": len(serialized),
        "disassembly_sha256": _sha256(disassembly.encode()), "disassembly_nbytes": len(disassembly.encode()),
        "disassembly_tool": disassembly_tool,
      },
      "fixture": fixture, "files": _inventory(files), "archive_format": "ustar",
      "consumer": {"requires_recompile": False, "entrypoint": "load_frozen_target_artifact"},
    }
    for name, data in files.items(): (staging / name).write_bytes(data)
    (staging / "manifest.json").write_bytes(_json_bytes(manifest))
    # Validate the staged bytes and the deserialized PROGRAM before publishing.
    load_frozen_target_artifact(staging)
    os.replace(staging, output)
    if archive is not None:
      _write_archive(output, Path(archive))
      load_frozen_target_artifact(Path(archive))
    return manifest
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


@dataclass(frozen=True)
class FrozenTargetArtifact:
  manifest: Mapping[str, Any]
  program: UOp
  binary: bytes
  source: str
  disassembly: str
  fixture: Mapping[str, Any]
  sink: UOp | None = None


def _read_bundle(path: Path) -> dict[str, bytes]:
  if path.is_dir(): return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
  if not path.is_file(): raise FileNotFoundError(path)
  with tarfile.open(path, "r") as tf:
    members = tf.getmembers()
    if any(not m.isfile() or Path(m.name).name != m.name for m in members):
      raise ValueError("archive must contain only top-level regular files")
    if len({m.name for m in members}) != len(members): raise ValueError("archive contains duplicate names")
    return {m.name: tf.extractfile(m).read() for m in members}  # type: ignore[union-attr]


def load_frozen_target_artifact(path: str | Path) -> FrozenTargetArtifact:
  """Load and fully validate a directory or tar bundle without compilation."""
  files = _read_bundle(Path(path))
  try: manifest = json.loads(files["manifest.json"])
  except KeyError as exc: raise ValueError("bundle has no manifest.json") from exc
  except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ValueError(f"invalid manifest JSON: {exc}") from exc
  schema = manifest.get("schema")
  if schema not in (SCHEMA, LEGACY_SCHEMA) or manifest.get("state") != "FROZEN":
    raise ValueError("bundle does not contain a frozen target manifest")
  legacy = schema == LEGACY_SCHEMA
  retained_names = {name for kind, name in FILE_NAMES.items() if not legacy or kind != "sink"}
  required = {"manifest.json", *retained_names}
  if set(files) != required: raise ValueError(f"bundle file set changed: expected {sorted(required)}, got {sorted(files)}")
  if manifest.get("compile_calls") != 1 or manifest.get("accumulate") is not True:
    raise ValueError("manifest does not attest one accumulate=True compile")
  compiler_environment = manifest.get("compiler_environment")
  if not isinstance(compiler_environment, dict) or set(compiler_environment) - set(COMPILER_ENV) or \
     any(value is not None and not isinstance(value, str) for value in compiler_environment.values()):
    raise ValueError("manifest compiler environment is malformed")
  if manifest.get("gpu_runtime_initialized") is not False or manifest.get("gpu_dispatch_performed") is not False:
    raise ValueError("frozen artifact must be produced without GPU runtime or dispatch")
  retained = {name: files[name] for name in retained_names}
  if manifest.get("files") != _inventory(retained): raise ValueError("retained file inventory identity mismatch")
  try: role_spec = exact_role_spec_from_shape(tuple(manifest.get("full_role_shape", ())))
  except (TypeError, ValueError) as exc: raise ValueError(f"frozen full-role shape is not inventory-admitted: {exc}") from exc
  if tuple(manifest.get("shape", ())) != role_spec.program.shape:
    raise ValueError("frozen program geometry differs from admitted full role")
  try: program = pickle.loads(files[FILE_NAMES["program"]])
  except BaseException as exc: raise ValueError(f"serialized PROGRAM cannot be loaded: {type(exc).__name__}: {exc}") from exc
  program = _validate_program(program, role_spec)
  sink = None
  if not legacy:
    try: sink = pickle.loads(files[FILE_NAMES["sink"]])
    except BaseException as exc:
      raise ValueError(f"serialized pre-lowering SINK cannot be loaded: {type(exc).__name__}: {exc}") from exc
    sink = _validate_source_sink(sink, role_spec)
    serialized_sink = files[FILE_NAMES["sink"]]
    if manifest.get("source_sink") != {
        "authority": "same_session_pre_lowering_sink_passed_to_compiler",
        "key": sink.key.hex(),
        "serialized_sha256": _sha256(serialized_sink),
        "serialized_nbytes": len(serialized_sink),
    }:
      raise ValueError("pre-lowering SINK identity differs from manifest")
  elif "source_sink" in manifest:
    raise ValueError("legacy frozen target artifact cannot claim pre-lowering SINK authority")
  binary, source, shared_artifacts = _program_payload(program)
  if binary != files[FILE_NAMES["binary"]]: raise ValueError("serialized PROGRAM binary differs from retained HSACO")
  if source.encode() != files[FILE_NAMES["source"]]: raise ValueError("serialized PROGRAM source differs from retained source")
  disassembly = files[FILE_NAMES["disassembly"]].decode()
  fixture = json.loads(files[FILE_NAMES["fixture"]])
  if tuple(fixture.get("shape", ())) != role_spec.shape or \
     ("role" in fixture and fixture["role"] != role_spec.role):
    raise ValueError("fixture role/shape differs from admitted frozen role")
  artifacts = manifest.get("artifacts", {})
  expected_artifacts = {
    **shared_artifacts,
    "serialized_program_sha256": _sha256(files[FILE_NAMES["program"]]),
    "serialized_program_nbytes": len(files[FILE_NAMES["program"]]),
    "disassembly_sha256": _sha256(disassembly.encode()), "disassembly_nbytes": len(disassembly.encode()),
  }
  if any(artifacts.get(k) != v for k, v in expected_artifacts.items()):
    raise ValueError("artifact hash or size identity mismatch")
  if manifest.get("fixture") != fixture: raise ValueError("fixture identity differs from manifest")
  program_manifest = manifest.get("program", {})
  if program_manifest != {
      "function": program.arg.function_name, "key": program.key.hex(),
      "device": program.src[1].arg, "compile_target": AMD_ISA_TARGET,
      "globals": list(program.arg.globals), "global_size": list(program.arg.global_size),
      "local_size": list(program.arg.local_size or ()), "abi": _abi(program, role_spec.program)}:
    raise ValueError("serialized PROGRAM launch identity differs from manifest")
  return FrozenTargetArtifact(manifest, program, binary, source, disassembly, fixture, sink)


def audit_frozen_target_artifact(path: str | Path) -> dict[str, Any]:
  """Run the independent static HSACO audit once against one validated frozen bundle."""
  from extra.qk.mmq_hsaco_static_audit import audit_hsaco

  artifact = load_frozen_target_artifact(path)
  static_audit = audit_hsaco(artifact.binary, artifact.disassembly)
  manifest_artifacts = artifact.manifest["artifacts"]
  expected_identity = {
    "binary_sha256": manifest_artifacts["binary_sha256"],
    "binary_nbytes": manifest_artifacts["binary_nbytes"],
    "disassembly_sha256": manifest_artifacts["disassembly_sha256"],
    "disassembly_nbytes": manifest_artifacts["disassembly_nbytes"],
  }
  observed_identity = {
    "binary_sha256": static_audit.get("binary_sha256"),
    "binary_nbytes": static_audit.get("binary_nbytes"),
    "disassembly_sha256": static_audit.get("disassembly_sha256"),
    "disassembly_nbytes": len(artifact.disassembly.encode()),
  }
  mismatches = [
    f"static audit {key} differs from frozen manifest"
    for key, expected in expected_identity.items() if observed_identity[key] != expected
  ]
  identity = {
    "passed": not mismatches, "expected": expected_identity, "observed": observed_identity,
    "findings": mismatches,
  }
  static_passed = static_audit.get("passed") is True and static_audit.get("verdict") == "PASS"
  passed = static_passed and identity["passed"]
  findings = [*mismatches]
  if not static_passed:
    audit_findings = static_audit.get("findings")
    if audit_findings: findings.extend(str(finding) for finding in audit_findings)
    else: findings.append("static HSACO audit did not pass")
  return {
    "schema": AUDIT_SCHEMA, "passed": passed, "verdict": "PASS" if passed else "BLOCKED",
    "bundle": {
      "schema": artifact.manifest["schema"], "state": artifact.manifest["state"],
      "program_key": artifact.manifest["program"]["key"],
      "function": artifact.manifest["program"]["function"],
    },
    "identity": identity, "static_audit": static_audit, "findings": findings,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  produce = sub.add_parser("produce", help="compile once and freeze the exact accumulate=True target PROGRAM")
  produce.add_argument("--output-dir", type=Path, required=True)
  produce.add_argument("--archive", type=Path)
  produce.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
  produce.add_argument("--role", default=DEFAULT_EXACT_ROLE_SPEC.role)
  verify = sub.add_parser("verify", help="validate and inspect a frozen directory or tar without recompiling")
  verify.add_argument("bundle", type=Path)
  audit = sub.add_parser("audit", help="run the CPU-only static HSACO audit against a frozen bundle")
  audit.add_argument("bundle", type=Path)
  args = parser.parse_args(argv)
  if args.command == "produce":
    result = produce_frozen_target_artifact(args.output_dir, archive=args.archive,
                                            role_spec=exact_role_spec(args.role, inventory=args.inventory))
  elif args.command == "verify":
    result = dict(load_frozen_target_artifact(args.bundle).manifest)
  else:
    result = audit_frozen_target_artifact(args.bundle)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if args.command != "audit" or result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "ACCUMULATION", "AUDIT_SCHEMA", "FILE_NAMES", "FIXTURE_SCHEMA", "FrozenTargetArtifact", "LEGACY_SCHEMA", "SCHEMA",
  "audit_frozen_target_artifact", "deterministic_fixture_identity", "load_frozen_target_artifact",
  "produce_frozen_target_artifact",
]
