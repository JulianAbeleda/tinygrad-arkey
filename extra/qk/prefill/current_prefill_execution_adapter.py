"""Production adapter for the promoted scheduler-generated prefill GEMM.

The adapter consumes an exact full-kernel candidate payload.  It never selects
from a route name and never substitutes another transport.  Parent-side work is
compile-only; the PROGRAM and runtime are reconstructed in the spawned child.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import prod
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from extra.qk.prefill.executable_artifact_preparation import compile_transport_evidence
from extra.qk.prefill.host_safety_canary import make_tiny_health_probe
from extra.qk.prefill.isolated_guarded_executor import (ExecutableBundle, build_tinygrad_bundle,
  make_tinygrad_bundle_builder)
from extra.qk.prefill.operand_path_execution_worker import PreparedExecution
from extra.qk.mmq_epoch_manifest_export import DEFAULT_MAX_ROWS, build_amd_isa_proof_manifest_bundle
from extra.qk.runtime_specs import (GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, admit_full_kernel_candidate,
                                    capability_transport)
from tinygrad.runtime.execution_bridge_contracts import ExecutionRequest

ADAPTER_ID = "tinygrad.amd.gfx1100.current_prefill.v1"
# ProgramInfo.globals=(0,1,2), outs=(0,), ins=(1,2): output, A, B.
_ARGUMENT_ORDER = ("output", "a", "b")
_TARGET = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}
_COMPILE_DEVICE = "AMD"
_RUNTIME_DEVICE = "AMD"


def _runtime_alive() -> bool:
  """The guarded child already owns a live runtime; independent health is probed separately."""
  return True


def _workload(payload: Mapping[str, Any]) -> tuple[str, str, tuple[int, int, int], dict[str, Any]]:
  row = payload.get("workload")
  if not isinstance(row, Mapping): raise ValueError("candidate payload has no workload")
  shape, target = row.get("shape"), row.get("target")
  if not isinstance(shape, Mapping) or not isinstance(target, Mapping): raise ValueError("candidate workload is malformed")
  return str(row.get("profile", "")), str(row.get("role", "")), tuple(int(shape[x]) for x in ("m", "n", "k")), dict(target)


def admit_current_prefill(payload: dict[str, Any], canonical_identity: str):
  """Reuse canonical admission and additionally restrict this adapter's production surface."""
  profile, role, shape, target = _workload(payload)
  if role != "attn_qo" or shape != (512, 4096, 4096):
    raise ValueError("current prefill adapter supports only attn_qo 512x4096x4096")
  if target != _TARGET: raise ValueError("current prefill adapter requires AMD:gfx1100:wave32")
  return admit_full_kernel_candidate(payload, canonical_identity, profile=profile, role=role, shape=shape, target=target,
                                     capability=GFX1100_TWO_BUFFER_STAGE1_CAPABILITY)


def compile_current_prefill_program(payload: dict[str, Any], canonical_identity: str, *, device: str):
  """Compile the admitted current Tensor GEMM; never allocate runtime buffers or dispatch."""
  admission = admit_current_prefill(payload, canonical_identity)
  _, _, (m, n, k), _ = _workload(admission.normalized_payload)
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program_cache
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.codegen.opt.postrange import warmstart_candidate_state
  from tinygrad.engine.realize import compile_linear
  from tinygrad.helpers import Context, getenv
  from tinygrad.uop.ops import Ops

  key = (frozenset({m, n}), k)
  opts = {key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}
  getenv.cache_clear(); to_program_cache.clear()
  try:
    with warmstart_candidate_state(opts, {key: admission.context}), Context(DEV=device):
      a = Tensor.empty(m, k, dtype=dtypes.half)
      if (transform := admission.context.packed_weight) is None:
        b = Tensor.empty(n, k, dtype=dtypes.half)
      else:
        # Movement-only logical carrier: it preserves the real packed PARAM plus N/K ownership for ordinary matmul
        # matching. Postrange replaces these dummy half values with transform.dequant at the B tile producer.
        blocks, halfwords = n*k//transform.block_elems, transform.block_bytes//2
        packed = Tensor.empty(transform.packed_bytes//transform.storage_width, dtype=transform.storage_dtype)
        b = packed.bitcast(dtypes.uint16).reshape(blocks, halfwords).pad(((0,0),(0,128-halfwords))) \
          .reshape(blocks,128,1).expand(blocks,128,2).reshape(n,k).bitcast(dtypes.half)
      compiled = compile_linear((a @ b.transpose()).schedule_linear())
  finally:
    getenv.cache_clear(); to_program_cache.clear()
  all_programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM]
  programs = [u for u in all_programs if
              getattr(getattr(u.src[0].arg, "candidate_context", None), "canonical_identity", None) == canonical_identity]
  if len(programs) != 1 or len(all_programs) != 1:
    raise ValueError(f"expected one total identity-bound candidate PROGRAM, found {len(programs)} bound/{len(all_programs)} total")
  return programs[0], admission


def prepare_current_prefill_compile(payload: dict[str, Any], canonical_identity: str, *, device: str = _COMPILE_DEVICE):
  from tinygrad.renderer.isa.amd import capture_amd_isa_proof_manifest
  try:
    with capture_amd_isa_proof_manifest(max_rows=DEFAULT_MAX_ROWS) as proof_rows:
      program, admission = compile_current_prefill_program(payload, canonical_identity, device=device)
  except Exception as exc:
    raise ValueError(f"required final_isa_manifest/resource_summary unavailable: final compile failed ({type(exc).__name__}: {exc})") from exc
  schedule = admission.normalized_payload["schedule"]
  transform = admission.context.packed_weight
  evidence = compile_transport_evidence(program, transport=capability_transport(admission.capability),
    canonical_identity=canonical_identity,
    schedule={"threads": schedule["threads"], "lds_bytes": admission.active_lds_bytes,
              "tile": dict(schedule["tile"]), "pipeline": dict(schedule["pipeline"])},
    surface={"source_kind": "tinygrad_scheduler", "consumer": "packed_dequant_wmma" if transform else "dense_gemm",
             "role": "attn_qo"},
    runtime_binding=dict(admission.normalized_payload["workload"]))
  if evidence.get("passed") is not True: raise ValueError("current prefill compile evidence failed")
  source = next((u.arg for u in program.src if u.op.name == "SOURCE" and isinstance(u.arg, str)), None)
  binary = next((u.arg for u in program.src if u.op.name == "BINARY" and isinstance(u.arg, bytes)), None)
  compiled_target = next((u.arg for u in program.src if u.op.name == "DEVICE"), None)
  target = "gfx1100"
  if not source or not binary or not isinstance(compiled_target, str): raise ValueError("final compile identity inputs are unavailable")
  if tuple(program.arg.globals) != (0, 1, 2) or tuple(program.arg.outs) != (0,) or tuple(program.arg.ins) != (1, 2):
    raise ValueError("compiled PROGRAM ABI differs from output,a,b execution contract")
  source_sha, binary_sha = hashlib.sha256(source.encode()).hexdigest(), hashlib.sha256(binary).hexdigest()
  if binary_sha != evidence.get("binary_sha256"): raise ValueError("compile evidence binary identity mismatch")
  manifest = build_amd_isa_proof_manifest_bundle(candidate_id=canonical_identity,
    kernel_name=str(getattr(program.arg, "name", "")), rows=proof_rows, source_sha256=source_sha, binary_sha256=binary_sha,
    abi_metadata={"argument_order": list(_ARGUMENT_ORDER), "globals": list(program.arg.globals),
                  "outputs": list(program.arg.outs), "inputs": list(program.arg.ins)},
    ownership_metadata={"semantic_operands": [
      {"operand_id": "C", "abi_index": 0, "abi_argument": "output", "semantic_role": "output"},
      {"operand_id": "A", "abi_index": 1, "abi_argument": "a", "semantic_role": "lhs_activation"},
      {"operand_id": "B", "abi_index": 2, "abi_argument": "b", "semantic_role": "rhs_weight", **(
        {"representation": "packed_scalar_decoder", "quant_format": transform.quant_format,
         "storage_dtype": admission.normalized_payload["operand_sources"]["b"]["storage_dtype"],
         "packed_bytes": transform.packed_bytes,
         "decoder_version": admission.normalized_payload["operand_sources"]["b"]["decoder_version"]}
        if transform is not None else {"representation": "dense", "storage_dtype": "half"})}]},
    digest_metadata={"canonical_identity": canonical_identity})
  # The shipping AMD/HIP compiler does not pass through the ISA renderer's row
  # tags. Analyze its exact code object and leave semantic row ownership
  # explicitly unavailable rather than borrowing rows from the distinct
  # AMD:ISA binary.
  from extra.qk.mmq_compile_evidence import analyze_final_isa, disassemble_amdgpu, parse_amdgpu_metadata
  from extra.qk.operand_attribution import attribute_operands, operand_paths_for_manifest
  disassembly, disassembly_tool = disassemble_amdgpu(binary)
  metadata = parse_amdgpu_metadata(binary)
  final_isa = analyze_final_isa(disassembly, wavefront_size=metadata["wavefront_size"])
  # Derive semantic operand ownership from the EXACT shipping code object via a bounded ABI-rooted
  # dataflow pass (kernarg pointer -> global load -> LDS stage -> WMMA). Rows that cannot be traced
  # (e.g. the double-buffered LDS windows) remain explicit unknown with a named discriminator; nothing
  # is inferred from route names or the distinct AMD:ISA binary.
  attribution = attribute_operands(final_isa["instructions"],
    {"outs": list(program.arg.outs), "ins": list(program.arg.ins)})
  operand_paths = operand_paths_for_manifest(attribution, final_isa["instructions"], binary_sha256=binary_sha)
  attributed_rows = [p for p in operand_paths if p["operand_id"] != "unknown"]
  unknown_discriminators = sorted({p["missing"] for p in operand_paths if p["operand_id"] == "unknown"})
  structure = {"schema": "tinygrad.amd_isa_structure_summary.v1",
    "row_count": final_isa["instruction_count"],
    "counts": {"global_load": final_isa["global_load_sites"], "global_store": final_isa["global_store_sites"],
      "ds_load": final_isa["ds_load_sites"], "ds_store": final_isa["ds_store_sites"],
      "wait": final_isa["waitcnt_sites"], "barrier": final_isa["barrier_sites"],
      "wmma": sum("wmma" in row["mnemonic"] for row in final_isa["instructions"]),
      "scratch": final_isa["scratch_sites"]},
    "operand_paths": operand_paths,
    "operand_ownership_authority": attribution["authority"],
    "operand_ownership_binary_sha256": binary_sha,
    "attributed_row_count": len(attributed_rows),
    "unknown_row_count": len(operand_paths) - len(attributed_rows),
    "missing_evidence": unknown_discriminators}
  from tinygrad.renderer.amd.elf import descriptor_register_counts, kernel_descriptor_from_elf
  desc = kernel_descriptor_from_elf(binary)
  allocated_vgpr, _ = descriptor_register_counts(desc, is_cdna=False)
  descriptor_lds = int(desc.group_segment_fixed_size)
  if descriptor_lds != admission.active_lds_bytes:
    raise ValueError(f"final descriptor LDS {descriptor_lds} does not match admitted active LDS {admission.active_lds_bytes}")
  if metadata["lds_bytes"] != descriptor_lds or metadata["wavefront_size"] != 32:
    raise ValueError("final AMD metadata disagrees with descriptor or admitted wavefront")
  if any(metadata[field] for field in ("vgpr_spills", "sgpr_spills", "scratch_bytes")):
    raise ValueError("shipping prefill binary unexpectedly uses spills or scratch")
  local_size, global_size = getattr(program.arg, "local_size", None), getattr(program.arg, "global_size", None)
  resources = {"schema": "tinygrad.amd.final_resource_summary.v1", "stage": "final_program",
    "authority": "final_code_object_metadata_descriptor_and_program_launch", "vgpr": metadata["vgpr"],
    "allocated_vgpr": allocated_vgpr, "sgpr": metadata["sgpr"],
    "lds_bytes": descriptor_lds, "admitted_active_lds_bytes": admission.active_lds_bytes,
    "scratch_bytes": metadata["scratch_bytes"], "vgpr_spills": metadata["vgpr_spills"],
    "sgpr_spills": metadata["sgpr_spills"], "workgroup": list(local_size) if local_size else None,
    "workgroup_threads": prod(local_size) if local_size else None,
    "grid": list(global_size) if global_size else None, "wavefront_size": 32,
    "source_sha256": source_sha, "binary_sha256": binary_sha, "canonical_identity": canonical_identity, "target": target,
    "compiled_device": compiled_target}
  packed_gate = None
  if transform is not None:
    from extra.qk.packed_wmma_compile_gate import (CandidateEvidence, ProgramEvidence, ResourceEvidence,
      classify_registered_packed_wmma_candidate)
    wmma_families = tuple(sorted({row["mnemonic"] for row in final_isa["instructions"] if "wmma" in row["mnemonic"]}))
    packed_candidate = CandidateEvidence(canonical_identity, transform.quant_format, transform.rows, transform.k, (
      ProgramEvidence(str(getattr(program.arg, "name", "")), True,
        wmma_families,
        ("activation", "packed_weight"), ("packed_weight",), resources=ResourceEvidence(
          descriptor_lds, metadata["scratch_bytes"], metadata["vgpr_spills"], metadata["sgpr_spills"])),))
    packed_gate = classify_registered_packed_wmma_candidate(transform.quant_format, packed_candidate).to_json()
    if packed_gate["passed"] is not True: raise ValueError(f"packed WMMA compile gate failed: {packed_gate['reasons']}")
  evidence.update(source_sha256=source_sha, binary_sha256=binary_sha, target_id="amd_gfx1100", target=target,
    compiled_device=compiled_target, compile_target=device,
    canonical_identity=canonical_identity, final_isa_manifest=manifest,
    final_isa={"schema": "tinygrad.amd.final_isa.v1", "sha256": hashlib.sha256(disassembly.encode()).hexdigest(),
      "tool": disassembly_tool, "instruction_count": final_isa["instruction_count"], "binary_sha256": binary_sha},
    resource_summary=resources, isa_structure=structure,
    artifacts={"final_isa": {"status": "satisfied"},
      "final_isa_manifest": {"status": "satisfied" if not unknown_discriminators else "partial",
        "attributed_rows": len(attributed_rows), "unknown_rows": len(operand_paths) - len(attributed_rows),
        "missing": unknown_discriminators},
      "resource_summary": {"status": "satisfied", "unavailable_fields": []}},
    executed_binary_matches_compile=True,
    child_recompile_binary_identity_contract={"enabled": True, "reject_sha256_mismatch_before_dispatch": True,
      "canonical_identity": canonical_identity, "source_sha256": source_sha, "binary_sha256": binary_sha,
      "target": target, "compile_target": device})
  if packed_gate is not None: evidence["packed_wmma_compile_gate"] = packed_gate
  return program, evidence


def build_current_prefill_bundle(*, payload: dict[str, Any], canonical_identity: str,
                                 compile_evidence: Mapping[str, Any], compile_device: str = _COMPILE_DEVICE,
                                 runtime_device: str = _RUNTIME_DEVICE) -> ExecutableBundle:
  """Spawn-child entry point: recompile, verify identity, then construct runtime hooks."""
  contract = compile_evidence.get("child_recompile_binary_identity_contract")
  if not isinstance(contract, Mapping) or contract.get("enabled") is not True or \
     contract.get("reject_sha256_mismatch_before_dispatch") is not True:
    raise ValueError("spawn-child binary identity contract is missing")
  if contract.get("canonical_identity") != canonical_identity or contract.get("binary_sha256") != compile_evidence.get("binary_sha256"):
    raise ValueError("spawn-child binary identity contract disagrees with parent evidence")
  if contract.get("compile_target") != compile_device or compile_evidence.get("compile_target") != compile_device:
    raise ValueError("spawn-child compile target differs from admitted parent compile")
  program, child_evidence = prepare_current_prefill_compile(payload, canonical_identity, device=compile_device)
  for field in ("canonical_identity", "source_sha256", "binary_sha256", "target", "compile_target"):
    if child_evidence.get(field) != compile_evidence.get(field):
      raise ValueError(f"spawn-child {field} differs from admitted parent compile")
  child_evidence = dict(child_evidence)
  for field in ("input_identity", "reference_identity", "input_identity_detail"):
    if field in compile_evidence: child_evidence[field] = compile_evidence[field]
  return build_tinygrad_bundle(program=program, compile_evidence=child_evidence, device=runtime_device,
                               argument_order=_ARGUMENT_ORDER, health=_runtime_alive)


def _arrays(path: str, shape: tuple[int, int, int], packed_weight: Any | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
  artifact = Path(path)
  if not artifact.is_file(): raise ValueError(f"input artifact does not exist: {artifact}")
  with np.load(artifact, allow_pickle=False) as row:
    if set(row.files) != {"a", "b", "reference"}: raise ValueError("input NPZ must contain exactly a, b, reference")
    a, b, reference = (np.ascontiguousarray(row[x]) for x in ("a", "b", "reference"))
  m, n, k = shape
  if a.shape != (m, k) or reference.shape != (m, n): raise ValueError("input NPZ A/reference shapes do not match the exact candidate workload")
  if a.dtype != np.float16 or reference.dtype != np.float16: raise ValueError("current prefill A and reference must be fp16")
  if packed_weight is None:
    if b.shape != (n, k) or b.dtype != np.float16: raise ValueError("dense prefill B must be fp16 with exact (N,K) shape")
  else:
    expected_dtype = np.dtype(np.uint32 if packed_weight.quant_format == "Q4_K" else np.uint16)
    expected_shape = (packed_weight.packed_bytes // packed_weight.storage_width,)
    if b.shape != expected_shape or b.dtype != expected_dtype or b.nbytes != packed_weight.packed_bytes:
      raise ValueError(f"packed prefill B must be {expected_dtype.name}{expected_shape} with {packed_weight.packed_bytes} bytes")
  return {"a": a, "b": b}, reference


def _input_artifact_identities(path: str, reference: np.ndarray) -> dict[str, Any]:
  """Bind execution evidence to exact container bytes and the loaded reference value."""
  artifact = Path(path)
  if not artifact.is_file(): raise ValueError(f"input artifact does not exist: {artifact}")
  artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
  contiguous = np.ascontiguousarray(reference)
  reference_header = f"{contiguous.dtype.str}:{','.join(map(str, contiguous.shape))}:".encode()
  return {"schema": "tinygrad.execution_input_identity.v1", "algorithm": "sha256",
          "input_artifact_sha256": artifact_sha,
          "reference_sha256": hashlib.sha256(reference_header + contiguous.tobytes()).hexdigest(),
          "reference_identity_basis": "dtype_shape_and_c_contiguous_bytes"}


@dataclass(frozen=True)
class CurrentPrefillAdapter:
  compile_prepare: Callable[..., tuple[Any, Mapping[str, Any]]] = prepare_current_prefill_compile

  def prepare(self, request: ExecutionRequest) -> PreparedExecution:
    context = request.compiler_context
    payload, identity = context.get("candidate_payload"), context.get("canonical_identity")
    if not isinstance(payload, dict) or not isinstance(identity, str):
      raise ValueError("compiler_context requires candidate_payload and canonical_identity")
    admission = admit_current_prefill(payload, identity)
    declared = capability_transport(admission.capability)
    if request.transport_plan.transport != declared:
      raise ValueError(f"typed transport {request.transport_plan.transport!r} does not match admitted {declared!r}")
    inputs, reference = _arrays(str(context.get("input_npz", "")), _workload(payload)[2], admission.context.packed_weight)
    _, evidence = self.compile_prepare(payload, identity, device=_COMPILE_DEVICE)
    evidence = dict(evidence)
    identity_detail = _input_artifact_identities(str(context.get("input_npz", "")), reference)
    input_identity = "sha256:" + identity_detail["input_artifact_sha256"]
    reference_identity = "sha256:" + identity_detail["reference_sha256"]
    if request.target_context.get("input_identity") not in (None, input_identity):
      raise ValueError("request input identity does not match the loaded NPZ")
    if request.target_context.get("reference_identity") not in (None, reference_identity):
      raise ValueError("request reference identity does not match the loaded reference")
    evidence.update(input_identity=input_identity, reference_identity=reference_identity,
                    input_identity_detail=identity_detail)
    builder = make_tinygrad_bundle_builder(build=build_current_prefill_bundle, payload=payload,
      canonical_identity=identity, compile_evidence=evidence, compile_device=_COMPILE_DEVICE,
      runtime_device=_RUNTIME_DEVICE)
    return PreparedExecution(builder, inputs, reference, evidence,
                             health_probe=make_tiny_health_probe(device=_RUNTIME_DEVICE), output_dtype=np.float16)


def register_current_prefill_adapter(registry: Any) -> None:
  """Explicit registration hook; importing this module does not mutate global worker state."""
  registry.register(ADAPTER_ID, CurrentPrefillAdapter())


__all__ = ["ADAPTER_ID", "CurrentPrefillAdapter", "admit_current_prefill", "compile_current_prefill_program",
           "prepare_current_prefill_compile", "build_current_prefill_bundle", "register_current_prefill_adapter"]
