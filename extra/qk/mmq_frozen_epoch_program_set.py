"""Frozen full-role family of static-offset K256 PROGRAMs.

The provenance-complete v3 schema is deliberately separate from both the v1
singular K256 artifact and legacy v2 epoch families. It
retains each variant's pre-lowering sink beside its final PROGRAM/source/binary
triple. The sink is the structural offset authority; the PROGRAM is the
executable ABI/payload authority. Their relationship is trusted only because
the producer receives both from one emitted family variant in the same build
session. Loading constructs no runtime and performs no recompilation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable, Mapping

from tinygrad import dtypes
from tinygrad.dtype import PtrDType
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_compile_evidence import COMPILER_ENV
from extra.qk.mmq_exact_role_spec import (
  DEFAULT_INVENTORY, EPOCH_K, ExactRoleSpec, admit_exact_role_spec, exact_role_spec,
)
from extra.qk.mmq_frozen_target_artifact import (
  ACCUMULATION, BACKEND_ID, FUNCTION_NAME, PROGRAM_DEVICE,
)
from extra.qk.mmq_llama_five_buffer_full_kernel import (
  AMD_ISA_TARGET, LlamaFiveBufferEpochOffsetFamily,
)
from extra.qk.mmq_llama_five_buffer_graph import FiveBufferEpochOffsets, five_buffer_parameters
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_epoch_program_set.v3"
LEGACY_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_epoch_program_set.v2"
BINDING_SCHEMA = "tinygrad.prefill_frozen_epoch_program_set_binding.v2"
PROVENANCE_SCHEMA = "tinygrad.mmq_q4k_q8_1.generation_provenance.v1"
ROLE_CONTRACT_SCHEMA = "tinygrad.mmq_q4k_q8_1.full_role_contract.v1"
LOCAL_SIZE = (256, 1, 1)
ABI_NAMES = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")
REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_CONTEXT = (
  "ALLOW_TF32", "DISABLE_FAST_IDIV", "EMULATED_DTYPES", "IMAGE", "NOLOCALS",
  "NOOPT", "SPEC", "TC", "TC_OPT", "TC_SELECT", "TUPLE_ORDER",
)
CODEGEN_ENV = tuple(sorted(set(COMPILER_ENV) | {
  "ALLOW_HALF8", "DECODE_FAST_EXP2", "DEVECTORIZE_NO_PTR_GROUP",
  "PYTHONHASHSEED",
  "REGALLOC_ADDR_REMAT", "REGALLOC_ADDR_REMAT_END_NO_EMIT", "REGALLOC_ADDR_REMAT_NO_END",
  "SCHED_MODULO", "SCHED_MODULO_PROBE", "UNSAFE_DISABLE_MASK", "V_DOT2_LOWERING",
}))


def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _variant_files(epoch: int) -> dict[str, str]:
  stem = f"epoch_{epoch:03d}"
  return {"sink": f"{stem}.sink.pkl", "program": f"{stem}.program.pkl",
          "source": f"{stem}.source.txt", "binary": f"{stem}.hsaco"}


def _inventory(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
  return {name: {"sha256": _sha256(data), "nbytes": len(data)} for name, data in sorted(files.items())}


def _toolchain_inputs() -> dict[str, Any]:
  from tinygrad.codegen import _lower_cache_table
  return {
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "python_cache_tag": sys.implementation.cache_tag,
    "pickle_protocol": pickle.HIGHEST_PROTOCOL,
    "compile_target": AMD_ISA_TARGET,
    "program_device": PROGRAM_DEVICE,
    "renderer": "tinygrad.renderer.isa.amd.AMDISARenderer",
    "lowering_table": _lower_cache_table(),
    "generator_schema": SCHEMA,
  }


def _git_value(*args: str) -> str:
  try:
    return subprocess.run(("git", *args), cwd=REPO_ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
  except (OSError, subprocess.SubprocessError) as exc:
    raise ValueError(f"cannot establish frozen-family source revision: {exc}") from exc


def collect_generation_provenance() -> dict[str, Any]:
  """Collect a clean, exact source/toolchain/config declaration before generation.

  A dirty checkout is not a reproducible revision. Tests and offline importers may
  pass an explicit, schema-valid provenance record to the producer, but the
  default production path only admits a clean git worktree.
  """
  status = _git_value("status", "--porcelain", "--untracked-files=all")
  if status: raise ValueError("frozen-family generation requires a clean source worktree")
  commit, tree = _git_value("rev-parse", "HEAD"), _git_value("rev-parse", "HEAD^{tree}")
  toolchain = _toolchain_inputs()
  context = {
    key: ContextVar._cache[key].value
    for key in CODEGEN_CONTEXT
    if key in ContextVar._cache
  }
  missing = set(CODEGEN_CONTEXT) - set(context)
  if missing: raise ValueError(f"codegen context declaration is unavailable: {sorted(missing)}")
  environment = {
    key: {"present": key in os.environ, "value": os.environ.get(key)}
    for key in CODEGEN_ENV
  }
  return {
    "schema": PROVENANCE_SCHEMA,
    "source_revision": {"vcs": "git", "commit": commit, "tree": tree, "clean": True},
    "toolchain": {
      "inputs": toolchain,
      "fingerprint": "sha256:" + _sha256(_json_bytes(toolchain)),
    },
    "codegen": {"context": context, "environment": environment},
  }


def _validate_generation_provenance(value: Any) -> dict[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {"schema", "source_revision", "toolchain", "codegen"} or \
     value.get("schema") != PROVENANCE_SCHEMA:
    raise ValueError("generation provenance schema or fields are malformed")
  revision = value.get("source_revision")
  if not isinstance(revision, Mapping) or set(revision) != {"vcs", "commit", "tree", "clean"} or \
     revision.get("vcs") != "git" or revision.get("clean") is not True or \
     any(not isinstance(revision.get(key), str) or re.fullmatch(r"[0-9a-f]{40,64}", revision[key]) is None
         for key in ("commit", "tree")):
    raise ValueError("generation provenance source revision is not one clean exact git revision")
  toolchain = value.get("toolchain")
  if not isinstance(toolchain, Mapping) or set(toolchain) != {"inputs", "fingerprint"} or \
     not isinstance(toolchain.get("inputs"), Mapping):
    raise ValueError("generation provenance toolchain is malformed")
  expected_toolchain_fields = set(_toolchain_inputs())
  if set(toolchain["inputs"]) != expected_toolchain_fields or \
     any(not isinstance(toolchain["inputs"].get(key), (str, int)) or isinstance(toolchain["inputs"].get(key), bool)
         for key in expected_toolchain_fields):
    raise ValueError("generation provenance toolchain inputs are incomplete or malformed")
  expected_fingerprint = "sha256:" + _sha256(_json_bytes(toolchain["inputs"]))
  if toolchain.get("fingerprint") != expected_fingerprint:
    raise ValueError("generation provenance toolchain fingerprint differs from its declared inputs")
  codegen = value.get("codegen")
  if not isinstance(codegen, Mapping) or set(codegen) != {"context", "environment"}:
    raise ValueError("generation provenance codegen declaration is malformed")
  context, environment = codegen.get("context"), codegen.get("environment")
  if not isinstance(context, Mapping) or set(context) != set(CODEGEN_CONTEXT) or \
     any(not isinstance(item, (str, int, float, bool)) and item is not None for item in context.values()):
    raise ValueError("generation provenance effective codegen context is incomplete or malformed")
  if not isinstance(environment, Mapping) or set(environment) != set(CODEGEN_ENV):
    raise ValueError("generation provenance codegen environment is incomplete or malformed")
  for key, row in environment.items():
    if not isinstance(row, Mapping) or set(row) != {"present", "value"} or not isinstance(row["present"], bool) or \
       (row["value"] is not None and not isinstance(row["value"], str)) or \
       (row["present"] != (row["value"] is not None)):
      raise ValueError(f"generation provenance environment entry {key!r} is malformed")
  try: return json.loads(_json_bytes(value))
  except (TypeError, ValueError) as exc: raise ValueError(f"generation provenance is not canonical JSON: {exc}") from exc


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


def _expected_physical_layout(role_spec: ExactRoleSpec) -> dict[str, Any]:
  return {
    "q4": {
      "shape": ["N", "epochs", 36],
      "epoch_base_words": 36,
      "row_stride_words": role_spec.epochs * 36,
    },
    "q8_values": {
      "shape": ["K/128", "M", 128],
      "epoch_records": 2,
      "record_stride_elements": role_spec.m * 128,
    },
    "q8_scales": {
      "shape": ["K/128", "M", 4],
      "epoch_records": 2,
      "record_stride_elements": role_spec.m * 4,
    },
    "q8_original_sums": {
      "shape": ["K/128", "M", 4],
      "epoch_records": 2,
      "record_stride_elements": role_spec.m * 4,
    },
  }


def _role_contract(role_spec: ExactRoleSpec) -> dict[str, Any]:
  return {
    "schema": ROLE_CONTRACT_SCHEMA,
    "role": role_spec.role,
    "shape": {"M": role_spec.m, "N": role_spec.n, "K": role_spec.k},
    "candidate_identity": role_spec.candidate_canonical_identity,
    "quantization": {"weights": "Q4_K", "activations": "Q8_1", "accumulation": ACCUMULATION},
    "epoch": {"K": EPOCH_K, "count": role_spec.epochs, "ordered_ordinals": list(range(role_spec.epochs))},
    "five_buffer_abi": list(_expected_abi(role_spec)),
    "physical_layout": _expected_physical_layout(role_spec),
    "launch": {
      "function": FUNCTION_NAME, "backend_id": BACKEND_ID, "device": PROGRAM_DEVICE,
      "compile_target": AMD_ISA_TARGET, "global_size": list(role_spec.program.grid),
      "local_size": list(LOCAL_SIZE),
    },
    "source_contract": {"llama_commit": LLAMA_SOURCE_COMMIT},
  }


def _ordered_identities(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [{
    "epoch": row["epoch"], "sink_key": row["sink_key"], "program_key": row["program_key"],
    "sink_sha256": row["artifacts"]["sink"]["sha256"],
    "program_sha256": row["artifacts"]["program"]["sha256"],
    "source_sha256": row["artifacts"]["source"]["sha256"],
    "binary_sha256": row["artifacts"]["binary"]["sha256"],
  } for row in variants]


def _manifest_content_digest(manifest: Mapping[str, Any]) -> str:
  content = {key: value for key, value in manifest.items() if key not in ("family_identity", "content_address")}
  return _sha256(_json_bytes(content))


def _program_payload(program: UOp) -> tuple[bytes, str]:
  binaries = [node.arg for node in program.src if node.op is Ops.BINARY]
  sources = [node.arg for node in program.src if node.op is Ops.SOURCE]
  if len(binaries) != 1 or not isinstance(binaries[0], bytes) or not binaries[0]:
    raise ValueError("epoch PROGRAM must retain exactly one nonempty BINARY")
  if len(sources) != 1 or not isinstance(sources[0], str) or not sources[0]:
    raise ValueError("epoch PROGRAM must retain exactly one nonempty SOURCE")
  return binaries[0], sources[0]


def _graph_abi(graph: UOp, role_spec: ExactRoleSpec, *, authority: str) -> tuple[dict[str, Any], ...]:
  params = sorted({node for node in graph.toposort() if node.op is Ops.PARAM},
                  key=lambda node: node.arg.slot)
  expected = _expected_abi(role_spec)
  rows = tuple({
    "slot": int(node.arg.slot), "name": expected[node.arg.slot]["name"],
    "dtype": str(node.dtype), "elements": int(node.max_numel()),
  } for node in params)
  if rows != expected: raise ValueError(f"epoch {authority} does not expose the shared full-role five-buffer ABI")
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
    except ValueError as exc: raise ValueError(f"epoch sink input base is not a constant at tile zero: {rendered}") from exc
  return int(reduced.arg)


def _sink_offsets(sink: UOp) -> dict[str, int]:
  offsets: dict[int, int] = {}
  for node in sink.toposort():
    if node.op is not Ops.INDEX or node.src[0].op is not Ops.PARAM: continue
    slot = int(node.src[0].arg.slot)
    if slot not in range(1, 5): continue
    offset = _constant_tile_zero_offset(node.src[1])
    if slot in offsets and offsets[slot] != offset:
      raise ValueError(f"epoch sink has ambiguous direct base offsets for ABI slot {slot}")
    offsets[slot] = offset
  if set(offsets) != set(range(1, 5)):
    raise ValueError("epoch sink lacks one direct base offset for every input ABI slot")
  return {name: offsets[slot] for slot, name in enumerate(ABI_NAMES[1:], start=1)}


def _peel_pointer_address(address: UOp) -> tuple[UOp, UOp | None] | None:
  offsets, cursor = [], address
  while True:
    if cursor.op is Ops.INDEX:
      if len(cursor.src) < 2: return None
      offsets.append(cursor.src[1])
      cursor = cursor.src[0]
      continue
    if cursor.op is Ops.AFTER and isinstance(cursor.dtype, PtrDType):
      if not cursor.src: return cursor, None
      cursor = cursor.src[0]
      continue
    break
  if not offsets: return cursor, None
  total = offsets[0]
  for offset in offsets[1:]: total = total + offset
  return cursor, total


def _effective_param_index(address: UOp) -> tuple[int, UOp] | None:
  peeled = _peel_pointer_address(address)
  if peeled is None: return None
  cursor, total = peeled
  if cursor.op is not Ops.PARAM or total is None: return None
  return int(cursor.arg.slot), total


def _special_inference_expression(value: UOp) -> UOp:
  replacements = {
    node: UOp.variable(str(node.arg), int(node.vmin), int(node.vmax))
    for node in value.toposort() if node.op is Ops.SPECIAL
  }
  return value.substitute(replacements).simplify()


def _effective_address_counter(values: list[UOp], coordinates: tuple[dict[str, int], ...]
                               ) -> dict[int, int]:
  counts: dict[int, int] = {}
  prepared = tuple(_special_inference_expression(value) for value in values)
  try:
    for coordinate in coordinates:
      for value in prepared:
        address = int(value.sym_infer(coordinate))
        counts[address] = counts.get(address, 0) + 1
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"epoch sink LOAD address cannot be evaluated over the admitted grid: {exc}") from exc
  return counts


def _endpoints(count: int) -> tuple[int, ...]:
  if count <= 0: raise ValueError("admitted grid extent must be positive")
  return tuple(sorted({0, count-1}))


def _validate_sink_physical_strides(sink: UOp, role_spec: ExactRoleSpec, epoch: int) -> None:
  effective: dict[int, list[UOp]] = {slot: [] for slot in range(1, 5)}
  for node in sink.toposort():
    if node.op is not Ops.LOAD: continue
    peeled = _peel_pointer_address(node.src[0])
    terminal = node.src[0] if peeled is None else peeled[0]
    flattened = _effective_param_index(node.src[0])
    terminal_input_slots = {
      int(value.arg.slot) for value in terminal.toposort()
      if value.op is Ops.PARAM and int(value.arg.slot) in effective
    }
    if terminal_input_slots and flattened is None:
      raise ValueError("epoch sink contains an unsupported global input LOAD pointer chain")
    if flattened is not None and flattened[0] in effective:
      if terminal_input_slots != {flattened[0]}:
        raise ValueError("epoch sink global input LOAD address mixes ABI slots")
      effective[flattened[0]].append(flattened[1])

  local_ids = tuple(range(256))
  n_tiles, m_tiles = role_spec.n//128, role_spec.m//128
  q4_coordinates = tuple(
    {"lidx0": local, "gidx0": tile_n, "gidx1": 0}
    for tile_n in _endpoints(n_tiles) for local in local_ids)
  q4_actual = _effective_address_counter(effective[1], q4_coordinates)
  q4_expected: dict[int, int] = {}
  word_multiplicity = (8, 32, 32, 16) + (2,)*32
  for tile_n in _endpoints(n_tiles):
    base = (tile_n*128*role_spec.epochs+epoch)*36
    for row in range(128):
      for word, multiplicity in enumerate(word_multiplicity):
        q4_expected[base+row*role_spec.epochs*36+word] = multiplicity
  if q4_actual != q4_expected:
    raise ValueError("epoch sink Q4 LOAD coverage does not match the admitted full-role physical layout")

  q8_coordinates = tuple(
    {"lidx0": local, "gidx0": 0, "gidx1": tile_m}
    for tile_m in _endpoints(m_tiles) for local in local_ids)
  expected_geometry = {2: 128, 3: 4, 4: 4}
  for slot, width in expected_geometry.items():
    actual = _effective_address_counter(effective[slot], q8_coordinates)
    expected: dict[int, int] = {}
    for tile_m in _endpoints(m_tiles):
      base = (epoch*2*role_spec.m+tile_m*128)*width
      for phase in range(2):
        for row in range(128):
          for element in range(width):
            expected[base+(phase*role_spec.m+row)*width+element] = 1
    if actual != expected:
      raise ValueError(f"epoch sink {ABI_NAMES[slot]} LOAD coverage does not match the admitted full-role physical layout")


def _validate_sink(sink: Any, role_spec: ExactRoleSpec, epoch: int) -> UOp:
  if not isinstance(sink, UOp) or sink.op is not Ops.SINK:
    raise ValueError("epoch variant does not retain its pre-lowering sink")
  _graph_abi(sink, role_spec, authority="sink")
  if _sink_offsets(sink) != _offset_row(_expected_offsets(role_spec, epoch)):
    raise ValueError("epoch sink compile-time offsets differ from its ordinal")
  _validate_sink_physical_strides(sink, role_spec, epoch)
  return sink


def _validate_program(program: Any, role_spec: ExactRoleSpec) -> UOp:
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
  _graph_abi(program.src[0], role_spec, authority="PROGRAM")
  _program_payload(program)
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
  sinks: tuple[UOp, ...] = tuple()


def produce_frozen_epoch_program_set(output_dir: str | Path, *,
                                     role_spec: ExactRoleSpec,
                                     build_once: Callable[[], LlamaFiveBufferEpochOffsetFamily],
                                     archive: str | Path | None = None,
                                     inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
                                     generation_provenance: Mapping[str, Any] | None = None,
                                     ) -> dict[str, Any]:
  """Atomically freeze one already-emitted CPU-built epoch PROGRAM family."""
  role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
  automatic_provenance = generation_provenance is None
  provenance = _validate_generation_provenance(
    collect_generation_provenance() if automatic_provenance else generation_provenance)
  family = build_once()
  if automatic_provenance and _validate_generation_provenance(collect_generation_provenance()) != provenance:
    raise ValueError("source, toolchain, or codegen configuration changed during frozen-family generation")
  if not isinstance(family, LlamaFiveBufferEpochOffsetFamily) or not family.emitted:
    raise ValueError("frozen-family producer requires one fully emitted epoch-offset family")
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
    for epoch, (variant, program) in enumerate(zip(family.variants, family.programs)):
      sink = _validate_sink(variant.sink, role_spec, epoch)
      program = _validate_program(program, role_spec)
      binary, source = _program_payload(program)
      serialized_sink = pickle.dumps(sink, protocol=pickle.HIGHEST_PROTOCOL)
      serialized = pickle.dumps(program, protocol=pickle.HIGHEST_PROTOCOL)
      names = _variant_files(epoch)
      files = {"sink": serialized_sink, "program": serialized, "source": source.encode(), "binary": binary}
      retained.update({names[kind]: data for kind, data in files.items()})
      variants.append({
        "epoch": epoch, "offsets": _offset_row(_expected_offsets(role_spec, epoch)),
        "sink_key": sink.key.hex(), "program_key": program.key.hex(), "files": names,
        "artifacts": {
          kind: {"sha256": _sha256(data), "nbytes": len(data)}
          for kind, data in files.items()
        },
      })
    sink_keys, keys = [row["sink_key"] for row in variants], [row["program_key"] for row in variants]
    if len(set(sink_keys)) != role_spec.epochs or len(set(keys)) != role_spec.epochs:
      raise ValueError("epoch sink and PROGRAM keys must be unique across variants")
    manifest: dict[str, Any] = {
      "schema": SCHEMA, "state": "FROZEN", "family_builder_calls": 1,
      "variant_count": role_spec.epochs,
      "compile_only_cpu": True, "gpu_runtime_initialized": False, "gpu_dispatch_performed": False,
      "generation_provenance": provenance,
      "c1_certification": {
        "gate": "C1", "certified": True, "provenance_schema": PROVENANCE_SCHEMA,
        "content_addressed": True,
      },
      "backend_id": BACKEND_ID, "accumulation": ACCUMULATION, "accumulate": True,
      "role": {
        "name": role_spec.role, "shape": list(role_spec.shape), "epochs": role_spec.epochs,
        "candidate_identity": role_spec.candidate_canonical_identity,
      },
      "role_contract": _role_contract(role_spec),
      "shared_program": {
        "function": FUNCTION_NAME, "device": PROGRAM_DEVICE, "compile_target": AMD_ISA_TARGET,
        "globals": list(range(5)), "global_size": list(role_spec.program.grid),
        "local_size": list(LOCAL_SIZE), "abi": list(_expected_abi(role_spec)),
        "physical_layout": _expected_physical_layout(role_spec),
      },
      "compiler_boundary": {
        "authority": "producer_same_session_emitted_family_variant",
        "offset_authority": "retained_pre_lowering_sink",
        "executable_authority": "retained_final_program",
        "final_program_structural_offsets_claimed": False,
      },
      "variants": variants, "ordered_identities": _ordered_identities(variants),
      "files": _inventory(retained), "archive_format": "ustar",
      "consumer": {"requires_recompile": False, "entrypoint": "load_frozen_epoch_program_set"},
    }
    manifest["family_identity"] = _manifest_content_digest(manifest)
    manifest["content_address"] = "sha256:" + manifest["family_identity"]
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
                                  inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
                                  require_c1: bool = False,
                                  ) -> FrozenEpochProgramSetArtifact:
  """Load a v2/v3 family without compilation, runtime creation, or dispatch.

  Legacy v2 remains available for diagnostic replay. Promotion callers set
  ``require_c1=True`` so missing reproducible provenance fails before payload
  deserialization.
  """
  files = _read_bundle(Path(path))
  if "manifest.json" not in files: raise ValueError("bundle does not contain a frozen-family manifest")
  try: manifest = json.loads(files["manifest.json"])
  except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ValueError(f"invalid manifest JSON: {exc}") from exc
  legacy = manifest.get("schema") == LEGACY_SCHEMA
  if require_c1 and legacy:
    raise ValueError("legacy v2 frozen epoch bundle is C1-uncertified and must be regenerated for promotion")
  if manifest.get("schema") not in (SCHEMA, LEGACY_SCHEMA) or manifest.get("state") != "FROZEN":
    raise ValueError("bundle does not contain a frozen epoch program set")
  common_fields = {
    "schema", "state", "family_builder_calls", "variant_count", "compile_only_cpu",
    "gpu_runtime_initialized", "gpu_dispatch_performed", "backend_id", "accumulation",
    "accumulate", "role", "shared_program", "compiler_boundary", "family_identity",
    "variants", "files", "archive_format", "consumer",
  }
  version_fields = set() if legacy else {
    "generation_provenance", "c1_certification", "role_contract", "ordered_identities", "content_address",
  }
  if set(manifest) != common_fields | version_fields:
    raise ValueError("frozen family manifest fields differ from its versioned schema")
  if not legacy:
    _validate_generation_provenance(manifest.get("generation_provenance"))
    if manifest.get("c1_certification") != {
        "gate": "C1", "certified": True, "provenance_schema": PROVENANCE_SCHEMA,
        "content_addressed": True}:
      raise ValueError("frozen family C1 certification declaration is malformed")
  role = manifest.get("role")
  if not isinstance(role, Mapping):
    raise ValueError("v2 manifest role identity is malformed")
  role_spec = exact_role_spec(str(role.get("name", "")), shape=tuple(role.get("shape", ())), inventory=inventory)
  expected_role = {
    "name": role_spec.role, "shape": list(role_spec.shape), "epochs": role_spec.epochs,
    "candidate_identity": role_spec.candidate_canonical_identity,
  }
  if dict(role) != expected_role: raise ValueError("v2 manifest role identity differs from admission")
  if not legacy and manifest.get("role_contract") != _role_contract(role_spec):
    raise ValueError("frozen family role contract differs from exact admission")
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
    "physical_layout": _expected_physical_layout(role_spec),
  }
  if manifest.get("shared_program") != expected_shared:
    raise ValueError("v2 shared full-role ABI or launch identity changed")
  if manifest.get("compiler_boundary") != {
      "authority": "producer_same_session_emitted_family_variant",
      "offset_authority": "retained_pre_lowering_sink",
      "executable_authority": "retained_final_program",
      "final_program_structural_offsets_claimed": False}:
    raise ValueError("v2 same-session compiler authority boundary changed")
  consumer = manifest.get("consumer")
  if consumer != {"requires_recompile": False, "entrypoint": "load_frozen_epoch_program_set"}:
    raise ValueError("v2 consumer contract permits recompilation")
  if manifest.get("archive_format") != "ustar":
    raise ValueError("frozen family archive format changed")

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

  sinks, programs, binaries, sources, sink_keys, keys = [], [], [], [], [], []
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
    try: sink = pickle.loads(files[names["sink"]])
    except BaseException as exc:
      raise ValueError(f"epoch {epoch} serialized sink cannot be loaded: {type(exc).__name__}: {exc}") from exc
    sink = _validate_sink(sink, role_spec, epoch)
    if row.get("sink_key") != sink.key.hex():
      raise ValueError("serialized epoch sink key differs from the manifest")
    try: program = pickle.loads(files[names["program"]])
    except BaseException as exc:
      raise ValueError(f"epoch {epoch} serialized PROGRAM cannot be loaded: {type(exc).__name__}: {exc}") from exc
    program = _validate_program(program, role_spec)
    binary, source = _program_payload(program)
    if binary != files[names["binary"]] or source.encode() != files[names["source"]]:
      raise ValueError("serialized epoch PROGRAM payload differs from retained files")
    if row.get("program_key") != program.key.hex():
      raise ValueError("serialized epoch PROGRAM key differs from the manifest")
    sinks.append(sink); programs.append(program); binaries.append(binary); sources.append(source)
    sink_keys.append(sink.key.hex()); keys.append(program.key.hex())
  if len(set(sink_keys)) != role_spec.epochs or len(set(keys)) != role_spec.epochs:
    raise ValueError("v2 epoch sink or PROGRAM keys are not unique")
  if legacy:
    expected_identity = _sha256(_json_bytes({
      "role": role_spec.role, "shape": list(role_spec.shape),
      "candidate_identity": role_spec.candidate_canonical_identity,
      "physical_layout": _expected_physical_layout(role_spec),
      "sink_keys": sink_keys, "program_keys": keys,
    }))
  else:
    if manifest.get("ordered_identities") != _ordered_identities(variants):
      raise ValueError("frozen family ordered sink/PROGRAM/payload identities changed")
    expected_identity = _manifest_content_digest(manifest)
  if manifest.get("family_identity") != expected_identity:
    raise ValueError("frozen family content identity differs from its manifest and retained files")
  if not legacy and manifest.get("content_address") != "sha256:" + expected_identity:
    raise ValueError("frozen family content address differs from its content identity")
  loaded_manifest = dict(manifest)
  if legacy:
    loaded_manifest["c1_certification"] = {
      "gate": "C1", "certified": False, "status": "legacy_v2_missing_generation_provenance",
      "content_addressed": False,
    }
  return FrozenEpochProgramSetArtifact(
    loaded_manifest, tuple(programs), tuple(binaries), tuple(sources), tuple(sinks))


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
                                          require_c1: bool = False,
                                          artifact_loader: Callable[..., FrozenEpochProgramSetArtifact] =
                                            load_frozen_epoch_program_set
                                          ) -> FrozenEpochProgramSetBinding:
  """Bind one admitted role to an exact frozen full-role PROGRAM family."""
  role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
  artifact = artifact_loader(bundle, inventory=inventory, **({"require_c1": True} if require_c1 else {}))
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
  "BINDING_SCHEMA", "CODEGEN_CONTEXT", "CODEGEN_ENV", "FrozenEpochProgramSetArtifact",
  "FrozenEpochProgramSetBinding", "LEGACY_SCHEMA", "PROVENANCE_SCHEMA", "ROLE_CONTRACT_SCHEMA", "SCHEMA",
  "collect_generation_provenance",
  "load_frozen_epoch_program_set", "load_frozen_epoch_program_set_binding",
  "produce_frozen_epoch_program_set",
]
