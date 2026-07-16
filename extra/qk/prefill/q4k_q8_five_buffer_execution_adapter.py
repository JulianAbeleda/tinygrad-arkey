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
from extra.qk.prefill.guarded_execution import make_tinygrad_guarded_hooks
from extra.qk.prefill.operand_path_execution_worker import PreparedExecution
from extra.qk.prefill.execution_bridge_contracts import ExecutionRequest
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import (AMD_ISA_TARGET, admitted_buffer_descriptors,
  admit_q4k_q8_five_buffer_compile, compile_q4k_q8_five_buffer_program)
from extra.qk.runtime_specs import full_kernel_workload

ADAPTER_ID = "tinygrad.amd.gfx1100.q4k_q8_five_buffer.manual.v1"
_COMPILE_TARGET = AMD_ISA_TARGET
_RUNTIME_DEVICE = "AMD"
_NP_DTYPES = {"float32": np.dtype(np.float32), "uint32": np.dtype(np.uint32), "int8": np.dtype(np.int8)}
_PIPELINE_INPUT_FORMAT = "fp32_activation"


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
  return program, _compile_evidence(program, admission, target)


def _compile_evidence(program, admission, target: str) -> dict[str, Any]:
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
  return evidence


def prepare_q4k_q8_five_buffer_pipeline_compile(payload: dict[str, Any], canonical_identity: str,
                                                 *, target: str = _COMPILE_TARGET):
  """Compile the exact producer/MMQ pair and bind both binaries to one admission."""
  from extra.qk.prefill.q4k_q8_five_buffer_pipeline import compile_q4k_q8_five_buffer_pipeline
  pipeline = compile_q4k_q8_five_buffer_pipeline(payload, canonical_identity, target=target)
  evidence = _compile_evidence(pipeline.mmq, pipeline.admission, target)
  producer_evidence = _compile_evidence(pipeline.producer, pipeline.admission, target)
  producer_source_sha, producer_binary_sha = producer_evidence["source_sha256"], producer_evidence["binary_sha256"]
  pipeline_sha = hashlib.sha256(json.dumps({
    "mmq_source_sha256": evidence["source_sha256"], "mmq_binary_sha256": evidence["binary_sha256"],
    "producer_source_sha256": producer_source_sha, "producer_binary_sha256": producer_binary_sha,
  }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  if any(getattr(program.src[0].arg, "candidate_context", None) is not pipeline.admission.context
         for program in (pipeline.producer, pipeline.mmq)):
    raise ValueError("pipeline PROGRAM admission context drift")
  evidence.update(program_count=2, producer_source_sha256=producer_source_sha,
    producer_binary_sha256=producer_binary_sha, producer_resource_summary=producer_evidence["resource_summary"],
    pipeline_binary_sha256=pipeline_sha, execution_input_format=_PIPELINE_INPUT_FORMAT)
  evidence["child_recompile_binary_identity_contract"].update(producer_source_sha256=producer_source_sha,
    producer_binary_sha256=producer_binary_sha, producer_resource_summary=producer_evidence["resource_summary"],
    pipeline_binary_sha256=pipeline_sha, program_count=2, execution_input_format=_PIPELINE_INPUT_FORMAT)
  return pipeline, evidence


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


@dataclass
class _PipelineExecutable:
  producer: Any
  mmq: Any
  values: Any
  scales: Any
  sums: Any
  device: str

  def dispatch(self, output, q4_packed_words, activation):
    args = lambda *buffers: tuple(buffer.get_buf(self.device) for buffer in buffers)
    producer_time = self.producer.dispatch(*args(self.values, self.scales, self.sums, activation))
    mmq_time = self.mmq.dispatch(*args(output, q4_packed_words, self.values, self.scales, self.sums))
    if isinstance(producer_time, (int, float)) and isinstance(mmq_time, (int, float)):
      return producer_time + mmq_time
    return None

  def close(self):
    for buffer in (self.values, self.scales, self.sums):
      if buffer.is_allocated(): buffer.deallocate()
    self.producer.close(); self.mmq.close()


def build_q4k_q8_five_buffer_pipeline_bundle(*, payload: dict[str, Any], canonical_identity: str,
                                             compile_evidence: Mapping[str, Any],
                                             compile_target: str = _COMPILE_TARGET,
                                             runtime_device: str = _RUNTIME_DEVICE) -> ExecutableBundle:
  """Spawn-child entry for the exact physical-DS4 producer then MMQ route."""
  contract = compile_evidence.get("child_recompile_binary_identity_contract")
  required = ("canonical_identity", "abi_digest", "compile_target", "target", "source_sha256", "binary_sha256",
              "producer_source_sha256", "producer_binary_sha256", "producer_resource_summary",
              "pipeline_binary_sha256", "program_count", "execution_input_format")
  if not isinstance(contract, Mapping) or contract.get("enabled") is not True or \
     contract.get("reject_sha256_mismatch_before_dispatch") is not True:
    raise ValueError("spawn-child pipeline binary identity contract is missing")
  pipeline, child = prepare_q4k_q8_five_buffer_pipeline_compile(payload, canonical_identity, target=compile_target)
  for field in required:
    if child.get(field) != compile_evidence.get(field) or contract.get(field) != compile_evidence.get(field):
      raise ValueError(f"spawn-child pipeline {field} differs from admitted parent compile")
  if child.get("execution_input_format") != _PIPELINE_INPUT_FORMAT or pipeline.admission.canonical_identity != canonical_identity:
    raise ValueError("spawn-child pipeline input format or admission identity drift")
  from tinygrad.device import Buffer
  from tinygrad.runtime.bridge import prepare_executable
  descriptors = {row.name: row for row in admitted_buffer_descriptors(pipeline.admission)}
  buffers = [Buffer(runtime_device, prod(descriptors[name].flat_shape), descriptors[name].dtype, preallocate=True)
             for name in ("q8_ds4_values", "q8_scales", "q8_weighted_sums")]
  producer_evidence = {"passed": True, "binary_sha256": child["producer_binary_sha256"]}
  executable = _PipelineExecutable(prepare_executable(pipeline.producer, producer_evidence, device=runtime_device),
    prepare_executable(pipeline.mmq, child, device=runtime_device), *buffers, runtime_device)
  def dispatch(target, guarded):
    try: payloads = tuple(guarded[name].resource["payload"] for name in ("output", "q4_packed_words", "activation"))
    except KeyError as exc: raise ValueError(f"pipeline ABI buffer is missing: {exc.args[0]}") from exc
    return target.dispatch(*payloads)
  return ExecutableBundle(executable, make_tinygrad_guarded_hooks(runtime_device, dispatch, _runtime_alive))


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


def load_q4k_q8_five_buffer_pipeline_npz(path: str, admission) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
  """Load the explicit flat FP32 activation + flat Q4 pipeline input contract."""
  artifact = Path(path)
  if not artifact.is_file(): raise ValueError(f"input artifact does not exist: {artifact}")
  descriptors = {row.name: row for row in admitted_buffer_descriptors(admission)}
  workload = full_kernel_workload(admission.normalized_payload)
  expected = {"q4_packed_words": (np.dtype(np.uint32), descriptors["q4_packed_words"].flat_shape),
              "activation": (np.dtype(np.float32), (workload.shape[0] * workload.shape[2],)),
              "reference": (np.dtype(np.float32), descriptors["output"].flat_shape)}
  with np.load(artifact, allow_pickle=False) as row:
    if set(row.files) != set(expected): raise ValueError("pipeline input NPZ must contain exactly activation, q4_packed_words, reference")
    arrays = {name: np.ascontiguousarray(row[name]) for name in expected}
  for name, (dtype, shape) in expected.items():
    if arrays[name].dtype != dtype or arrays[name].shape != shape:
      raise ValueError(f"pipeline NPZ {name} must be {dtype.name}{shape}")
  reference = arrays.pop("reference")
  identities = {name: _array_identity(value) for name, value in {**arrays, "reference": reference}.items()}
  detail = {"schema": "tinygrad.execution_input_identity.v1", "algorithm": "sha256",
    "input_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "content_sha256": identities,
    "reference_sha256": identities["reference"], "reference_identity_basis": "dtype_shape_and_c_contiguous_bytes",
    "input_format": _PIPELINE_INPUT_FORMAT}
  return arrays, reference, detail


@dataclass(frozen=True)
class Q4KQ8FiveBufferAdapter:
  compile_prepare: Callable[..., tuple[Any, Mapping[str, Any]]] = prepare_q4k_q8_five_buffer_compile
  pipeline_compile_prepare: Callable[..., tuple[Any, Mapping[str, Any]]] = prepare_q4k_q8_five_buffer_pipeline_compile

  def prepare(self, request: ExecutionRequest) -> PreparedExecution:
    context = request.compiler_context
    payload, identity = context.get("candidate_payload"), context.get("canonical_identity")
    if not isinstance(payload, dict) or not isinstance(identity, str):
      raise ValueError("compiler_context requires candidate_payload and canonical_identity")
    admission = admit_q4k_q8_five_buffer_compile(payload, identity)
    if request.transport_plan.transport != "direct_global":
      raise ValueError("typed transport does not match admitted direct_global")
    input_format = context.get("input_format", "prequantized")
    if input_format == "prequantized":
      inputs, reference, detail = load_q4k_q8_five_buffer_npz(str(context.get("input_npz", "")), admission)
      _, evidence = self.compile_prepare(payload, admission.canonical_identity, target=_COMPILE_TARGET)
      bundle_build = build_q4k_q8_five_buffer_bundle
    elif input_format == _PIPELINE_INPUT_FORMAT:
      inputs, reference, detail = load_q4k_q8_five_buffer_pipeline_npz(str(context.get("input_npz", "")), admission)
      _, evidence = self.pipeline_compile_prepare(payload, admission.canonical_identity, target=_COMPILE_TARGET)
      bundle_build = build_q4k_q8_five_buffer_pipeline_bundle
    else: raise ValueError(f"unsupported five-buffer input_format {input_format!r}")
    evidence = dict(evidence)
    input_identity, reference_identity = "sha256:" + detail["input_artifact_sha256"], "sha256:" + detail["reference_sha256"]
    if request.target_context.get("input_identity") not in (None, input_identity):
      raise ValueError("request input identity does not match the loaded NPZ")
    if request.target_context.get("reference_identity") not in (None, reference_identity):
      raise ValueError("request reference identity does not match the loaded reference")
    evidence.update(input_identity=input_identity, reference_identity=reference_identity,
                    content_identities=dict(detail["content_sha256"]), input_identity_detail=detail)
    builder = make_tinygrad_bundle_builder(build=bundle_build, payload=payload,
      canonical_identity=admission.canonical_identity, compile_evidence=evidence,
      compile_target=_COMPILE_TARGET, runtime_device=_RUNTIME_DEVICE)
    return PreparedExecution(builder, inputs, reference, evidence,
      health_probe=make_tiny_health_probe(device=_RUNTIME_DEVICE), output_dtype=np.float32)


def register_q4k_q8_five_buffer_adapter(registry: Any) -> None:
  registry.register(ADAPTER_ID, Q4KQ8FiveBufferAdapter())


__all__ = ["ADAPTER_ID", "Q4KQ8FiveBufferAdapter", "prepare_q4k_q8_five_buffer_compile",
  "prepare_q4k_q8_five_buffer_pipeline_compile", "build_q4k_q8_five_buffer_bundle",
  "build_q4k_q8_five_buffer_pipeline_bundle", "load_q4k_q8_five_buffer_npz",
  "load_q4k_q8_five_buffer_pipeline_npz", "register_q4k_q8_five_buffer_adapter"]
