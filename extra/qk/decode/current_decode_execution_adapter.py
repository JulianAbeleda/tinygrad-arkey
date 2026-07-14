"""Compile-only execution evidence for the promoted Qwen3-8B Q4_K decode GEMV.

This module deliberately stops before runtime construction.  The repository has
no authority artifact containing the exact packed model operand, activation and
reference output for this route, so manufacturing a ``PreparedExecution`` here
would weaken the guarded-execution contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import prod
from typing import Any, Mapping

from extra.qk.gemv_g2_lanemap import Q4KGateUpLaneMap
from extra.qk.gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
from extra.qk.mmq_epoch_manifest_export import (DEFAULT_MAX_ROWS, build_amd_isa_proof_manifest_bundle,
                                                summarize_amd_isa_proof_rows)
from extra.qk.route_manifest import PROFILE_DECODE, ROUTES
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops, UOp

ADAPTER_ID = "tinygrad.amd.gfx1100.current_qwen3_8b_decode_q4k_g3.compile_only.v1"
ROUTE_ID = "decode_q4k_g3_generated"
TARGET = "AMD:HIP:gfx1100"
_Q4K_WORDS_PER_BLOCK = 36


@dataclass(frozen=True)
class CurrentDecodeCompileRequest:
  """Exact typed request; no model or route names are interpreted by a worker."""
  adapter_id: str
  route_id: str
  role: str
  rows: int
  k: int
  target: str = TARGET

  def __post_init__(self) -> None:
    if self.adapter_id != ADAPTER_ID: raise ValueError(f"adapter_id must be {ADAPTER_ID}")
    if self.route_id != ROUTE_ID: raise ValueError(f"route_id must be {ROUTE_ID}")
    route = ROUTES[ROUTE_ID]
    if route["status"] != "promoted_default" or route["profile_id"] != PROFILE_DECODE:
      raise ValueError("decode route manifest no longer identifies the promoted Qwen3-8B default")
    if self.role not in route["roles"]: raise ValueError(f"role {self.role!r} is not admitted by {ROUTE_ID}")
    if self.rows <= 0 or self.k <= 0: raise ValueError("rows and k must be positive")
    lane_map = Q4KGateUpLaneMap(k=self.k, n=self.rows)
    lane_map.validate()
    if self.rows % 32: raise ValueError("promoted G3 decode requires rows divisible by 32")
    if self.target != TARGET: raise ValueError(f"target must be {TARGET}")


@dataclass(frozen=True)
class DecodeExecutionBlocker:
  code: str
  phase: str
  recoverable: bool
  detail: Mapping[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return {"code": self.code, "phase": self.phase, "recoverable": self.recoverable, "detail": dict(self.detail)}


class DecodeExecutionBlocked(RuntimeError):
  def __init__(self, blocker: DecodeExecutionBlocker):
    self.blocker = blocker
    super().__init__(f"{blocker.code}: {blocker.detail['reason']}")


def build_current_decode_sink(request: CurrentDecodeCompileRequest) -> UOp:
  """Bind the exact final ABI: output, packed Q4_K words, fp16 activation."""
  words = request.rows * (request.k // 256) * _Q4K_WORDS_PER_BLOCK
  return q4k_g3_lanemap_gemv_kernel(request.rows, request.k)(
    UOp.placeholder((request.rows,), dtypes.float32, 0),
    UOp.placeholder((words,), dtypes.uint32, 1),
    UOp.placeholder((request.k,), dtypes.float16, 2))


def compile_current_decode_program(request: CurrentDecodeCompileRequest) -> UOp:
  """Lower through the existing promoted compiler authority; never dispatch."""
  from tinygrad.codegen import to_program
  program = to_program(build_current_decode_sink(request), HIPRenderer(Target.parse(request.target)))
  if program.op is not Ops.PROGRAM or len(program.src) < 5 or program.src[3].op is not Ops.SOURCE or program.src[4].op is not Ops.BINARY:
    raise RuntimeError("current decode lowering did not produce a final source-bound binary PROGRAM")
  expected = f"q4k_g3_lanemap_gemv_{request.rows}_{request.k}"
  if getattr(program.arg, "name", None) != expected: raise RuntimeError("compiled PROGRAM is not the admitted G3 lowering")
  return program


def _resource_summary(program: UOp, request: CurrentDecodeCompileRequest, source_sha: str, binary_sha: str) -> dict[str, Any]:
  from tinygrad.renderer.amd.elf import kernel_descriptor_from_elf
  from tinygrad.runtime.autogen import amdgpu_kd
  desc = kernel_descriptor_from_elf(program.src[4].arg)
  rsrc1 = int(desc.compute_pgm_rsrc1)
  vgpr = ((rsrc1 >> amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT) + 1) * 8
  local_size, global_size = getattr(program.arg, "local_size", None), getattr(program.arg, "global_size", None)
  return {"schema": "tinygrad.amd.final_resource_summary.v1", "stage": "final_program",
          "authority": "final_code_object_descriptor_and_program_launch", "vgpr": vgpr,
          "sgpr": None, "sgpr_status": "unavailable_in_rdna3_code_object_descriptor",
          "lds_bytes": int(desc.group_segment_fixed_size), "scratch_bytes": None,
          "scratch_status": "unavailable_at_final_elf_boundary",
          "workgroup": list(local_size) if local_size else None,
          "workgroup_threads": prod(local_size) if local_size else None,
          "grid": list(global_size) if global_size else None, "wavefront_size": 32,
          "source_sha256": source_sha, "binary_sha256": binary_sha, "route_id": request.route_id,
          "target": "gfx1100", "compiled_device": program.src[1].arg}


def prepare_current_decode_compile(request: CurrentDecodeCompileRequest) -> tuple[UOp, dict[str, Any]]:
  """Return final binary/source/resource/ISA evidence with dispatch forbidden."""
  from tinygrad.renderer.isa.amd import capture_amd_isa_proof_manifest
  with capture_amd_isa_proof_manifest(max_rows=DEFAULT_MAX_ROWS) as proof_rows:
    program = compile_current_decode_program(request)
  source, binary = program.src[3].arg, program.src[4].arg
  source_sha, binary_sha = hashlib.sha256(source.encode()).hexdigest(), hashlib.sha256(binary).hexdigest()
  from extra.qk.mmq_compile_evidence import disassemble_amdgpu, parse_amdgpu_metadata
  final_isa, disassembler = disassemble_amdgpu(binary)
  metadata = parse_amdgpu_metadata(binary)
  operands = ({"abi_index": 0, "abi_argument": "output", "semantic_role": "decode_output", "dtype": "float32",
               "shape": [request.rows]},
              {"abi_index": 1, "abi_argument": "words", "semantic_role": "packed_weight", "format": "Q4_K",
               "layout": "ggml_q4_k_36xu32_per_256_elements", "dtype": "uint32",
               "shape": [request.rows * (request.k // 256) * _Q4K_WORDS_PER_BLOCK]},
              {"abi_index": 2, "abi_argument": "x", "semantic_role": "decode_activation", "format": "fp16",
               "layout": "contiguous_k", "dtype": "float16", "shape": [request.k]})
  manifest = build_amd_isa_proof_manifest_bundle(candidate_id=ADAPTER_ID,
    kernel_name=program.arg.name, rows=proof_rows, source_sha256=source_sha, binary_sha256=binary_sha,
    abi_metadata={"argument_order": [x["abi_argument"] for x in operands]},
    ownership_metadata={"semantic_operands": list(operands)})
  blocker = DecodeExecutionBlocker("exact_decode_input_reference_authority_unavailable", "execution", True,
    {"reason": "no existing safe artifact binds the exact packed model bytes, decode activation, and reference output",
     "required_artifact": "immutable packed_weight + fp16 activation + float32 reference with content identities"})
  evidence = {"schema": "tinygrad.current_decode_compile_classification.v1", "passed": True,
    "classification": "compile_only", "adapter_id": ADAPTER_ID, "route_id": request.route_id,
    "profile_id": PROFILE_DECODE, "role": request.role, "source_sha256": source_sha,
    "binary_sha256": binary_sha, "program": {"name": program.arg.name, "target": program.src[1].arg},
    "semantic_operands": list(operands), "final_isa_manifest": manifest,
    "final_isa": {"schema": "tinygrad.amd.final_isa.v1", "text": final_isa,
                  "sha256": hashlib.sha256(final_isa.encode()).hexdigest(), "tool": disassembler},
    "isa_structure": summarize_amd_isa_proof_rows(proof_rows),
    "resource_summary": _resource_summary(program, request, source_sha, binary_sha),
    "capture": {"mode": "compile_only", "dispatch_permitted": False},
    "execution": {"status": "blocked", "dispatch_state": "not_attempted", "blocker": blocker.to_dict()},
    "counter_evidence": {"status": "not_collected", "reason": "compile-only adapter; no dispatch"}}
  evidence["resource_summary"]["amdgpu_metadata"] = metadata
  evidence["resource_summary"].update(vgpr=metadata["vgpr"], sgpr=metadata["sgpr"],
    sgpr_status="available_in_amdgpu_metadata", lds_bytes=metadata["lds_bytes"],
    scratch_bytes=metadata["scratch_bytes"], scratch_status="available_in_amdgpu_metadata")
  return program, evidence


@dataclass(frozen=True)
class CurrentDecodeExecutionAdapter:
  """Classification adapter. It intentionally cannot yield PreparedExecution."""
  def classify(self, request: CurrentDecodeCompileRequest) -> tuple[UOp, dict[str, Any]]:
    return prepare_current_decode_compile(request)

  def prepare(self, request: CurrentDecodeCompileRequest):
    _, evidence = self.classify(request)
    raise DecodeExecutionBlocked(DecodeExecutionBlocker(**evidence["execution"]["blocker"]))


__all__ = ["ADAPTER_ID", "ROUTE_ID", "TARGET", "CurrentDecodeCompileRequest", "DecodeExecutionBlocker",
           "DecodeExecutionBlocked", "CurrentDecodeExecutionAdapter", "build_current_decode_sink",
           "compile_current_decode_program", "prepare_current_decode_compile"]
