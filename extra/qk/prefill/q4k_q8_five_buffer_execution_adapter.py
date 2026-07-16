"""Guarded execution provider for the admitted manual Q4_K/Q8_1 five-buffer MMQ."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from math import prod
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from extra.qk.prefill.executable_artifact_preparation import compile_transport_evidence
from extra.qk.prefill.host_safety_canary import make_tiny_health_probe
from extra.qk.prefill.isolated_guarded_executor import (ExecutableBundle, build_tinygrad_bundle,
  make_tinygrad_bundle_builder)
from extra.qk.prefill.operand_path_execution_worker import PreparedExecution
from extra.qk.prefill.execution_bridge_contracts import ExecutionRequest
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import (AMD_ISA_TARGET, admitted_buffer_descriptors,
  admit_q4k_q8_five_buffer_compile, compile_q4k_q8_five_buffer_program)
from extra.qk.runtime_specs import full_kernel_workload

ADAPTER_ID = "tinygrad.amd.gfx1100.q4k_q8_five_buffer.manual.v1"
_COMPILE_TARGET = AMD_ISA_TARGET
_RUNTIME_DEVICE = "AMD"
_NP_DTYPES = {"float32": np.dtype(np.float32), "uint32": np.dtype(np.uint32), "int8": np.dtype(np.int8)}


def _runtime_alive() -> bool: return True


def _descriptor_json(descriptors) -> list[dict[str, Any]]:
  return [{"slot": x.slot, "name": x.name, "logical_shape": list(x.logical_shape),
           "flat_shape": list(x.flat_shape), "storage_dtype": x.storage_dtype,
           "direction": x.direction} for x in descriptors]


def _abi_digest(descriptors) -> str:
  raw = json.dumps(_descriptor_json(descriptors), sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(raw).hexdigest()


def prepare_q4k_q8_five_buffer_compile(payload: dict[str, Any], canonical_identity: str,
                                       *, target: str = _COMPILE_TARGET):
  """Compile statically and bind exact source, binary, candidate, and ABI identities."""
  program, admission = compile_q4k_q8_five_buffer_program(payload, canonical_identity, target=target)
  descriptors = admitted_buffer_descriptors(admission)
  schedule = admission.normalized_payload["schedule"]
  workload = full_kernel_workload(admission.normalized_payload)
  evidence = compile_transport_evidence(program, transport="direct_global",
    canonical_identity=admission.canonical_identity,
    schedule={"threads": schedule["threads"], "lds_bytes": 0, "tile": dict(schedule["tile"]),
              "pipeline": dict(schedule["pipeline"])},
    surface={"source_kind": "manual_uop_emitter", "consumer": "q4k_q8_1_physical_ds4", "role": workload.role},
    runtime_binding=dict(admission.normalized_payload["workload"]))
  if evidence.get("passed") is not True: raise ValueError("five-buffer compile evidence failed")
  source_sha = evidence["program"]["source_sha256"]
  binary_sha = evidence["binary_sha256"]
  abi_digest = _abi_digest(descriptors)
  binary = next((u.arg for u in program.src if u.op.name == "BINARY" and isinstance(u.arg, bytes)), None)
  if binary is None: raise ValueError("final static PROGRAM binary is unavailable")
  from extra.qk.mmq_compile_evidence import parse_amdgpu_metadata
  from tinygrad.renderer.amd.elf import descriptor_register_counts, kernel_descriptor_from_elf
  metadata = parse_amdgpu_metadata(binary)
  descriptor = kernel_descriptor_from_elf(binary)
  allocated_vgpr, allocated_sgpr = descriptor_register_counts(descriptor, is_cdna=False)
  descriptor_lds = int(descriptor.group_segment_fixed_size)
  if descriptor_lds != admission.active_lds_bytes or metadata["lds_bytes"] != descriptor_lds:
    raise ValueError("final code-object LDS differs from admitted active LDS")
  if metadata["wavefront_size"] != 32: raise ValueError("final code-object wavefront differs from admitted wave32")
  if any(metadata[field] != 0 for field in ("scratch_bytes", "vgpr_spills", "sgpr_spills")):
    raise ValueError("final code object unexpectedly uses scratch or spills")
  local_size, global_size = getattr(program.arg, "local_size", None), getattr(program.arg, "global_size", None)
  if not isinstance(local_size, tuple) or prod(local_size) != schedule["threads"]:
    raise ValueError("final PROGRAM workgroup differs from admitted threads")
  resources = {"schema": "tinygrad.amd.final_resource_summary.v1", "stage": "final_program",
    "authority": "final_code_object_metadata_descriptor_and_program_launch", "vgpr": metadata["vgpr"],
    "allocated_vgpr": allocated_vgpr, "sgpr": metadata["sgpr"], "allocated_sgpr": allocated_sgpr,
    "lds_bytes": descriptor_lds, "admitted_active_lds_bytes": admission.active_lds_bytes,
    "scratch_bytes": metadata["scratch_bytes"], "vgpr_spills": metadata["vgpr_spills"],
    "sgpr_spills": metadata["sgpr_spills"], "workgroup": list(local_size),
    "workgroup_threads": prod(local_size), "grid": list(global_size) if global_size else None,
    "wavefront_size": metadata["wavefront_size"], "source_sha256": source_sha,
    "binary_sha256": binary_sha, "canonical_identity": admission.canonical_identity, "target": "gfx1100"}
  evidence.update(source_sha256=source_sha, binary_sha256=binary_sha, target_id="amd_gfx1100",
    target="gfx1100", compile_target=target, canonical_identity=admission.canonical_identity,
    candidate_digest=admission.canonical_identity, abi_digest=abi_digest,
    abi={"argument_order": [x.name for x in descriptors], "buffers": _descriptor_json(descriptors)},
    resource_summary=resources, executed_binary_matches_compile=True,
    child_recompile_binary_identity_contract={"enabled": True, "reject_sha256_mismatch_before_dispatch": True,
      "canonical_identity": admission.canonical_identity, "source_sha256": source_sha,
      "binary_sha256": binary_sha, "abi_digest": abi_digest, "target": "gfx1100", "compile_target": target})
  return program, evidence


def build_q4k_q8_five_buffer_bundle(*, payload: dict[str, Any], canonical_identity: str,
                                    compile_evidence: Mapping[str, Any], compile_target: str = _COMPILE_TARGET,
                                    runtime_device: str = _RUNTIME_DEVICE) -> ExecutableBundle:
  """Spawn-child entry: recompile and reject every identity mismatch before runtime construction."""
  admission = admit_q4k_q8_five_buffer_compile(payload, canonical_identity)
  descriptors = admitted_buffer_descriptors(admission)
  contract = compile_evidence.get("child_recompile_binary_identity_contract")
  if not isinstance(contract, Mapping) or contract.get("enabled") is not True or \
     contract.get("reject_sha256_mismatch_before_dispatch") is not True:
    raise ValueError("spawn-child binary identity contract is missing")
  expected = {"canonical_identity": admission.canonical_identity, "abi_digest": _abi_digest(descriptors),
              "compile_target": compile_target}
  for field, value in expected.items():
    if contract.get(field) != value or compile_evidence.get(field) != value:
      raise ValueError(f"spawn-child {field} differs from admitted parent compile")
  program, child = prepare_q4k_q8_five_buffer_compile(payload, admission.canonical_identity, target=compile_target)
  for field in ("canonical_identity", "source_sha256", "binary_sha256", "abi_digest", "target", "compile_target"):
    if child.get(field) != compile_evidence.get(field) or contract.get(field) != compile_evidence.get(field):
      raise ValueError(f"spawn-child {field} differs from admitted parent compile")
  child = dict(child)
  for field in ("input_identity", "reference_identity", "content_identities", "input_identity_detail"):
    if field in compile_evidence: child[field] = compile_evidence[field]
  return build_tinygrad_bundle(program=program, compile_evidence=child, device=runtime_device,
    argument_order=tuple(x.name for x in descriptors), health=_runtime_alive)


def _array_identity(array: np.ndarray) -> str:
  value = np.ascontiguousarray(array)
  header = f"{value.dtype.str}:{','.join(map(str, value.shape))}:".encode()
  return hashlib.sha256(header + value.tobytes()).hexdigest()


def load_q4k_q8_five_buffer_npz(path: str, admission) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
  artifact = Path(path)
  if not artifact.is_file(): raise ValueError(f"input artifact does not exist: {artifact}")
  descriptors = admitted_buffer_descriptors(admission)
  input_names = tuple(x.name for x in descriptors if x.direction == "in")
  expected_names = {*input_names, "reference"}
  with np.load(artifact, allow_pickle=False) as row:
    if set(row.files) != expected_names: raise ValueError(f"input NPZ must contain exactly {', '.join(sorted(expected_names))}")
    arrays = {name: np.ascontiguousarray(row[name]) for name in (*input_names, "reference")}
  output = next(x for x in descriptors if x.direction == "out")
  for descriptor in descriptors:
    name = "reference" if descriptor.direction == "out" else descriptor.name
    value, dtype = arrays[name], _NP_DTYPES.get(descriptor.storage_dtype)
    if dtype is None or value.shape != descriptor.flat_shape or value.dtype != dtype:
      raise ValueError(f"NPZ {name} must be {descriptor.storage_dtype}{descriptor.flat_shape}")
  reference = arrays.pop("reference")
  identities = {name: _array_identity(value) for name, value in {**arrays, "reference": reference}.items()}
  detail = {"schema": "tinygrad.execution_input_identity.v1", "algorithm": "sha256",
            "input_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "content_sha256": identities, "reference_sha256": identities["reference"],
            "reference_identity_basis": "dtype_shape_and_c_contiguous_bytes"}
  return arrays, reference, detail


@dataclass(frozen=True)
class Q4KQ8FiveBufferAdapter:
  compile_prepare: Callable[..., tuple[Any, Mapping[str, Any]]] = prepare_q4k_q8_five_buffer_compile

  def prepare(self, request: ExecutionRequest) -> PreparedExecution:
    context = request.compiler_context
    payload, identity = context.get("candidate_payload"), context.get("canonical_identity")
    if not isinstance(payload, dict) or not isinstance(identity, str):
      raise ValueError("compiler_context requires candidate_payload and canonical_identity")
    admission = admit_q4k_q8_five_buffer_compile(payload, identity)
    if request.transport_plan.transport != "direct_global":
      raise ValueError("typed transport does not match admitted direct_global")
    inputs, reference, detail = load_q4k_q8_five_buffer_npz(str(context.get("input_npz", "")), admission)
    _, evidence = self.compile_prepare(payload, admission.canonical_identity, target=_COMPILE_TARGET)
    evidence = dict(evidence)
    input_identity, reference_identity = "sha256:" + detail["input_artifact_sha256"], "sha256:" + detail["reference_sha256"]
    if request.target_context.get("input_identity") not in (None, input_identity):
      raise ValueError("request input identity does not match the loaded NPZ")
    if request.target_context.get("reference_identity") not in (None, reference_identity):
      raise ValueError("request reference identity does not match the loaded reference")
    evidence.update(input_identity=input_identity, reference_identity=reference_identity,
                    content_identities=dict(detail["content_sha256"]), input_identity_detail=detail)
    builder = make_tinygrad_bundle_builder(build=build_q4k_q8_five_buffer_bundle, payload=payload,
      canonical_identity=admission.canonical_identity, compile_evidence=evidence,
      compile_target=_COMPILE_TARGET, runtime_device=_RUNTIME_DEVICE)
    return PreparedExecution(builder, inputs, reference, evidence,
      health_probe=make_tiny_health_probe(device=_RUNTIME_DEVICE), output_dtype=np.float32)


def register_q4k_q8_five_buffer_adapter(registry: Any) -> None:
  registry.register(ADAPTER_ID, Q4KQ8FiveBufferAdapter())


__all__ = ["ADAPTER_ID", "Q4KQ8FiveBufferAdapter", "prepare_q4k_q8_five_buffer_compile",
  "build_q4k_q8_five_buffer_bundle", "load_q4k_q8_five_buffer_npz", "register_q4k_q8_five_buffer_adapter"]
