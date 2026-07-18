"""Frozen v2 artifact for a full-role family of static-offset K256 PROGRAMs.

This schema is deliberately separate from the v1 singular K256 artifact.  It
retains one PROGRAM/source/binary triple per full-role epoch and validates the
compile-time input offsets from each serialized PROGRAM's UOp graph without
constructing a runtime or recompiling.
"""
from __future__ import annotations

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

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import (
  DEFAULT_INVENTORY, ExactRoleSpec, admit_exact_role_spec, exact_role_spec,
)
from extra.qk.mmq_frozen_target_artifact import (
  ACCUMULATION, BACKEND_ID, FUNCTION_NAME, PROGRAM_DEVICE,
)
from extra.qk.mmq_llama_five_buffer_full_kernel import (
  AMD_ISA_TARGET, LlamaFiveBufferEpochOffsetFamily,
)
from extra.qk.mmq_llama_five_buffer_graph import FiveBufferEpochOffsets, five_buffer_parameters


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_epoch_program_set.v2"
BINDING_SCHEMA = "tinygrad.prefill_frozen_epoch_program_set_binding.v2"
LOCAL_SIZE = (256, 1, 1)
ABI_NAMES = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")


def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _variant_files(epoch: int) -> dict[str, str]:
  stem = f"epoch_{epoch:03d}"
  return {"program": f"{stem}.program.pkl", "source": f"{stem}.source.txt", "binary": f"{stem}.hsaco"}


def _inventory(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
  return {name: {"sha256": _sha256(data), "nbytes": len(data)} for name, data in sorted(files.items())}


def _expected_offsets(role_spec: ExactRoleSpec, epoch: int) -> FiveBufferEpochOffsets:
  if not 0 <= epoch < role_spec.epochs: raise ValueError("epoch ordinal is outside the admitted full role")
  records = epoch * 2
  return FiveBufferEpochOffsets(
    q4=epoch * 36,
    values=records * role_spec.m * 128,
    scales=records * role_spec.m * 4,
    sums=records * role_spec.m * 4,
  )


def _offset_row(offsets: FiveBufferEpochOffsets) -> dict[str, int]:
  return {"q4": offsets.q4, "q8_values": offsets.values,
          "q8_scales": offsets.scales, "q8_original_sums": offsets.sums}


def _expected_abi(role_spec: ExactRoleSpec) -> tuple[dict[str, Any], ...]:
  return tuple({
    "slot": parameter.slot, "name": parameter.name, "dtype": str(parameter.dtype.ptr(parameter.size)),
    "elements": parameter.size,
  } for parameter in five_buffer_parameters(*role_spec.shape))


def _program_payload(program: UOp) -> tuple[bytes, str]:
  binaries = [node.arg for node in program.src if node.op is Ops.BINARY]
  sources = [node.arg for node in program.src if node.op is Ops.SOURCE]
  if len(binaries) != 1 or not isinstance(binaries[0], bytes) or not binaries[0]:
    raise ValueError("epoch PROGRAM must retain exactly one nonempty BINARY")
  if len(sources) != 1 or not isinstance(sources[0], str) or not sources[0]:
    raise ValueError("epoch PROGRAM must retain exactly one nonempty SOURCE")
  return binaries[0], sources[0]


def _program_abi(program: UOp, role_spec: ExactRoleSpec) -> tuple[dict[str, Any], ...]:
  params = sorted({node for node in program.src[0].toposort() if node.op is Ops.PARAM},
                  key=lambda node: node.arg.slot)
  expected = _expected_abi(role_spec)
  rows = tuple({
    "slot": int(node.arg.slot), "name": expected[node.arg.slot]["name"],
    "dtype": str(node.dtype), "elements": int(node.max_numel()),
  } for node in params)
  if rows != expected: raise ValueError("epoch PROGRAM does not expose the shared full-role five-buffer ABI")
  return rows


def _constant_tile_zero_offset(value: UOp) -> int:
  replacements = {
    node: UOp.const(dtypes.weakint, 0)
    for node in value.toposort() if node.op is Ops.SPECIAL and str(node.arg).startswith("gidx")
  }
  reduced = value.substitute(replacements).simplify()
  if reduced.op is not Ops.CONST or type(reduced.arg) is not int:
    rendered = reduced.render()
    try: return int(rendered)
    except ValueError as exc: raise ValueError(f"epoch PROGRAM input base is not a constant at tile zero: {rendered}") from exc
  return int(reduced.arg)


def _program_offsets(program: UOp) -> dict[str, int]:
  offsets: dict[int, int] = {}
  for node in program.src[0].toposort():
    if node.op is not Ops.INDEX or node.src[0].op is not Ops.PARAM: continue
    slot = int(node.src[0].arg.slot)
    if slot not in range(1, 5): continue
    offset = _constant_tile_zero_offset(node.src[1])
    if slot in offsets and offsets[slot] != offset:
      raise ValueError(f"epoch PROGRAM has ambiguous direct base offsets for ABI slot {slot}")
    offsets[slot] = offset
  if set(offsets) != set(range(1, 5)):
    raise ValueError("epoch PROGRAM lacks one direct base offset for every input ABI slot")
  return {name: offsets[slot] for slot, name in enumerate(ABI_NAMES[1:], start=1)}


def _validate_program(program: Any, role_spec: ExactRoleSpec, epoch: int) -> UOp:
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("epoch variant is not an Ops.PROGRAM")
  if [node for node in program.toposort() if node.op is Ops.PROGRAM] != [program]:
    raise ValueError("epoch variant must retain exactly one closed PROGRAM")
  if program.arg.function_name != FUNCTION_NAME or tuple(program.arg.globals) != tuple(range(5)):
    raise ValueError("epoch PROGRAM function or five-buffer globals changed")
  if tuple(program.arg.global_size) != role_spec.program.grid or tuple(program.arg.local_size or ()) != LOCAL_SIZE:
    raise ValueError("epoch PROGRAM grid or local size differs from the admitted role")
  if len(program.src) < 5 or program.src[1].op is not Ops.DEVICE or program.src[1].arg != PROGRAM_DEVICE:
    raise ValueError("epoch PROGRAM is not a native AMD PROGRAM")
  _program_abi(program, role_spec)
  _program_payload(program)
  if _program_offsets(program) != _offset_row(_expected_offsets(role_spec, epoch)):
    raise ValueError("epoch PROGRAM compile-time offsets differ from its ordinal")
  return program


def _write_archive(directory: Path, archive: Path) -> None:
  if archive.exists(): raise FileExistsError(f"archive already exists: {archive}")
  archive.parent.mkdir(parents=True, exist_ok=True)
  with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tf:
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
      data = path.read_bytes()
      info = tarfile.TarInfo(path.name)
      info.size, info.mtime, info.mode = len(data), 0, 0o644
      info.uid = info.gid = 0
      info.uname = info.gname = ""
      from io import BytesIO
      tf.addfile(info, BytesIO(data))


def _read_bundle(path: Path) -> dict[str, bytes]:
  if path.is_dir(): return {entry.name: entry.read_bytes() for entry in path.iterdir() if entry.is_file()}
  if not path.is_file(): raise FileNotFoundError(path)
  with tarfile.open(path, "r") as tf:
    members = tf.getmembers()
    if any(not member.isfile() or Path(member.name).name != member.name for member in members):
      raise ValueError("archive must contain only top-level regular files")
    if len({member.name for member in members}) != len(members):
      raise ValueError("archive contains duplicate names")
    return {member.name: tf.extractfile(member).read() for member in members}  # type: ignore[union-attr]


@dataclass(frozen=True)
class FrozenEpochProgramSetArtifact:
  manifest: Mapping[str, Any]
  programs: tuple[UOp, ...]
  binaries: tuple[bytes, ...]
  sources: tuple[str, ...]


def produce_frozen_epoch_program_set(output_dir: str | Path, *,
                                     role_spec: ExactRoleSpec,
                                     build_once: Callable[[], LlamaFiveBufferEpochOffsetFamily],
                                     archive: str | Path | None = None,
                                     inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY
                                     ) -> dict[str, Any]:
  """Atomically freeze one already-emitted CPU-built epoch PROGRAM family."""
  role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
  family = build_once()
  if not isinstance(family, LlamaFiveBufferEpochOffsetFamily) or not family.emitted:
    raise ValueError("v2 producer requires one fully emitted epoch-offset family")
  if tuple(variant.epoch_offset for variant in family.variants) != tuple(range(role_spec.epochs)):
    raise ValueError("emitted family ordinals differ from the admitted full role")
  if tuple(parameter.size for parameter in family.proof_graph.parameters) != \
     tuple(parameter.size for parameter in five_buffer_parameters(*role_spec.shape)):
    raise ValueError("emitted family ABI differs from the admitted full role")
  if family.topology.grid != role_spec.program.grid:
    raise ValueError("emitted family grid differs from the admitted full role")

  output = Path(output_dir)
  if output.exists(): raise FileExistsError(f"output already exists: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  try:
    retained: dict[str, bytes] = {}
    variants = []
    for epoch, program in enumerate(family.programs):
      program = _validate_program(program, role_spec, epoch)
      binary, source = _program_payload(program)
      serialized = pickle.dumps(program, protocol=pickle.HIGHEST_PROTOCOL)
      names = _variant_files(epoch)
      files = {"program": serialized, "source": source.encode(), "binary": binary}
      retained.update({names[kind]: data for kind, data in files.items()})
      variants.append({
        "epoch": epoch, "offsets": _offset_row(_expected_offsets(role_spec, epoch)),
        "program_key": program.key.hex(), "files": names,
        "artifacts": {
          kind: {"sha256": _sha256(data), "nbytes": len(data)}
          for kind, data in files.items()
        },
      })
    keys = [row["program_key"] for row in variants]
    if len(set(keys)) != role_spec.epochs:
      raise ValueError("epoch PROGRAM keys must be unique across compile-time offsets")
    family_identity = _sha256(_json_bytes({
      "role": role_spec.role, "shape": list(role_spec.shape),
      "candidate_identity": role_spec.candidate_canonical_identity,
      "program_keys": keys,
    }))
    manifest = {
      "schema": SCHEMA, "state": "FROZEN", "family_builder_calls": 1,
      "variant_count": role_spec.epochs,
      "compile_only_cpu": True, "gpu_runtime_initialized": False, "gpu_dispatch_performed": False,
      "backend_id": BACKEND_ID, "accumulation": ACCUMULATION, "accumulate": True,
      "role": {
        "name": role_spec.role, "shape": list(role_spec.shape), "epochs": role_spec.epochs,
        "candidate_identity": role_spec.candidate_canonical_identity,
      },
      "shared_program": {
        "function": FUNCTION_NAME, "device": PROGRAM_DEVICE, "compile_target": AMD_ISA_TARGET,
        "globals": list(range(5)), "global_size": list(role_spec.program.grid),
        "local_size": list(LOCAL_SIZE), "abi": list(_expected_abi(role_spec)),
      },
      "family_identity": family_identity, "variants": variants,
      "files": _inventory(retained), "archive_format": "ustar",
      "consumer": {"requires_recompile": False, "entrypoint": "load_frozen_epoch_program_set"},
    }
    for name, data in retained.items(): (staging / name).write_bytes(data)
    (staging / "manifest.json").write_bytes(_json_bytes(manifest))
    load_frozen_epoch_program_set(staging, inventory=inventory)
    os.replace(staging, output)
    if archive is not None:
      _write_archive(output, Path(archive))
      load_frozen_epoch_program_set(archive, inventory=inventory)
    return manifest
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


def load_frozen_epoch_program_set(path: str | Path, *,
                                  inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY
                                  ) -> FrozenEpochProgramSetArtifact:
  """Load and validate a v2 family without compilation, runtime creation, or dispatch."""
  files = _read_bundle(Path(path))
  if "manifest.json" not in files: raise ValueError("bundle does not contain a v2 manifest")
  try: manifest = json.loads(files["manifest.json"])
  except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ValueError(f"invalid manifest JSON: {exc}") from exc
  if manifest.get("schema") != SCHEMA or manifest.get("state") != "FROZEN":
    raise ValueError("bundle does not contain a frozen epoch program set")
  role = manifest.get("role")
  if not isinstance(role, Mapping):
    raise ValueError("v2 manifest role identity is malformed")
  role_spec = exact_role_spec(str(role.get("name", "")), shape=tuple(role.get("shape", ())), inventory=inventory)
  expected_role = {
    "name": role_spec.role, "shape": list(role_spec.shape), "epochs": role_spec.epochs,
    "candidate_identity": role_spec.candidate_canonical_identity,
  }
  if dict(role) != expected_role: raise ValueError("v2 manifest role identity differs from admission")
  if manifest.get("family_builder_calls") != 1 or manifest.get("variant_count") != role_spec.epochs:
    raise ValueError("v2 manifest builder census differs from the exact epoch family")
  if manifest.get("compile_only_cpu") is not True or manifest.get("gpu_runtime_initialized") is not False or \
     manifest.get("gpu_dispatch_performed") is not False:
    raise ValueError("v2 artifact crossed the CPU-only production boundary")
  if manifest.get("backend_id") != BACKEND_ID or manifest.get("accumulation") != ACCUMULATION or \
     manifest.get("accumulate") is not True:
    raise ValueError("v2 artifact backend or accumulation contract changed")
  expected_shared = {
    "function": FUNCTION_NAME, "device": PROGRAM_DEVICE, "compile_target": AMD_ISA_TARGET,
    "globals": list(range(5)), "global_size": list(role_spec.program.grid),
    "local_size": list(LOCAL_SIZE), "abi": list(_expected_abi(role_spec)),
  }
  if manifest.get("shared_program") != expected_shared:
    raise ValueError("v2 shared full-role ABI or launch identity changed")
  consumer = manifest.get("consumer")
  if not isinstance(consumer, Mapping) or consumer.get("requires_recompile") is not False:
    raise ValueError("v2 consumer contract permits recompilation")

  expected_names = {
    name for epoch in range(role_spec.epochs) for name in _variant_files(epoch).values()
  }
  if set(files) != {"manifest.json", *expected_names}:
    raise ValueError("v2 bundle file set differs from the exact epoch family")
  retained = {name: files[name] for name in expected_names}
  if manifest.get("files") != _inventory(retained):
    raise ValueError("v2 retained file inventory identity mismatch")
  variants = manifest.get("variants")
  if not isinstance(variants, list) or len(variants) != role_spec.epochs:
    raise ValueError("v2 manifest must contain exactly one variant per epoch")

  programs, binaries, sources, keys = [], [], [], []
  for epoch, row in enumerate(variants):
    names = _variant_files(epoch)
    expected_offsets = _offset_row(_expected_offsets(role_spec, epoch))
    if not isinstance(row, Mapping) or row.get("epoch") != epoch or row.get("offsets") != expected_offsets or \
       row.get("files") != names:
      raise ValueError("v2 variant ordinal, offsets, or filenames changed")
    expected_artifacts = {
      kind: {"sha256": _sha256(files[name]), "nbytes": len(files[name])}
      for kind, name in names.items()
    }
    if row.get("artifacts") != expected_artifacts:
      raise ValueError("v2 variant artifact hash or size identity mismatch")
    try: program = pickle.loads(files[names["program"]])
    except BaseException as exc:
      raise ValueError(f"epoch {epoch} serialized PROGRAM cannot be loaded: {type(exc).__name__}: {exc}") from exc
    program = _validate_program(program, role_spec, epoch)
    binary, source = _program_payload(program)
    if binary != files[names["binary"]] or source.encode() != files[names["source"]]:
      raise ValueError("serialized epoch PROGRAM payload differs from retained files")
    if row.get("program_key") != program.key.hex():
      raise ValueError("serialized epoch PROGRAM key differs from the manifest")
    programs.append(program); binaries.append(binary); sources.append(source); keys.append(program.key.hex())
  if len(set(keys)) != role_spec.epochs:
    raise ValueError("v2 epoch PROGRAM keys are not unique")
  expected_identity = _sha256(_json_bytes({
    "role": role_spec.role, "shape": list(role_spec.shape),
    "candidate_identity": role_spec.candidate_canonical_identity,
    "program_keys": keys,
  }))
  if manifest.get("family_identity") != expected_identity:
    raise ValueError("v2 family identity differs from the retained PROGRAM set")
  return FrozenEpochProgramSetArtifact(manifest, tuple(programs), tuple(binaries), tuple(sources))


@dataclass(frozen=True)
class FrozenEpochProgramSetBinding:
  schema: str
  role_spec: ExactRoleSpec
  artifact: FrozenEpochProgramSetArtifact
  candidate_identity: str
  family_identity: str
  program_keys: tuple[str, ...]


def load_frozen_epoch_program_set_binding(role_spec: ExactRoleSpec, bundle: str | Path, *,
                                          inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
                                          artifact_loader: Callable[..., FrozenEpochProgramSetArtifact] =
                                            load_frozen_epoch_program_set
                                          ) -> FrozenEpochProgramSetBinding:
  """Bind one admitted role to an exact v2 full-role PROGRAM family."""
  role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
  artifact = artifact_loader(bundle, inventory=inventory)
  if not isinstance(artifact, FrozenEpochProgramSetArtifact):
    raise TypeError("v2 artifact loader returned the wrong artifact type")
  role = artifact.manifest["role"]
  if role["name"] != role_spec.role or tuple(role["shape"]) != role_spec.shape or \
     role["candidate_identity"] != role_spec.candidate_canonical_identity:
    raise ValueError("v2 frozen family differs from the requested admitted role")
  keys = tuple(row["program_key"] for row in artifact.manifest["variants"])
  return FrozenEpochProgramSetBinding(
    BINDING_SCHEMA, role_spec, artifact, role_spec.candidate_canonical_identity,
    str(artifact.manifest["family_identity"]), keys,
  )


__all__ = [
  "BINDING_SCHEMA", "FrozenEpochProgramSetArtifact", "FrozenEpochProgramSetBinding", "SCHEMA",
  "load_frozen_epoch_program_set", "load_frozen_epoch_program_set_binding",
  "produce_frozen_epoch_program_set",
]
