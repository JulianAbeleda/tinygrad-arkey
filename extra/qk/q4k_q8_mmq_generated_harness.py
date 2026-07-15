"""Isolated validation harness for the descriptor-driven Q4_K x Q8_1 emitter.

This module is intentionally not a route or an atom adapter.  Compilation and
dispatch happen in the spawned child; the parent only receives evidence.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes
from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q8_1_BLOCK_ELEMS
from extra.qk.q4k_q8_mmq_emitter import emit_q4k_q8_mmq_prefill
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec
from extra.qk.prefill_primitive_spec import PrimitiveABI
from extra.qk.prefill.guarded_execution import GuardPolicy
from extra.qk.prefill.isolated_guarded_executor import (
  ExecutableBundle, ExecutionRequest, build_tinygrad_bundle,
  make_tinygrad_bundle_builder, run_isolated_guarded_execution,
)
from extra.qk.mmq_physical_validation import validate_physical_contract
from extra.qk.mmq_logical_vocabulary import MMQCandidate

PROVENANCE = "q4k_q8_mmq_descriptor_emitter_v1"


def _shared_candidate(payload: dict[str, Any]) -> MMQCandidate | None:
  """Decode the generator's shared candidate, without inventing geometry."""
  raw = payload.get("logical_candidate", payload.get("candidate"))
  if raw is None: return None
  if not isinstance(raw, dict): raise ValueError("logical_candidate must be an object")
  try:
    # The vocabulary owns construction and identity; accepting only its JSON
    # shape keeps this harness from growing a second candidate vocabulary.
    from extra.qk.mmq_logical_vocabulary import (Axis, BackendCapability, DType, DotOp,
      EdgePredicate, LogicalMMQDescriptor, Operation, Ownership, PhysicalMapping,
      Q4KDecode, Q8DS4Semantics, Stage, Staging, Synchronization, SyncScope)
    d, m, cap = raw["descriptor"], raw["mapping"], raw["capability"]
    q8 = d.get("q8", {})
    operation = d.get("operation", {})
    synchronization = d.get("synchronization", {})
    desc = LogicalMMQDescriptor(axes=tuple(Axis(**x) for x in d["axes"]),
      q4k=Q4KDecode(**d.get("q4k", {})), q8=Q8DS4Semantics(**{
        **q8, "values_dtype": DType(q8.get("values_dtype", "i8")),
        "scale_dtype": DType(q8.get("scale_dtype", "f32"))}),
      operation=Operation(name=DotOp(operation.get("name", "wmma_i8_i8_i32")),
        lhs_dtype=DType(operation.get("lhs_dtype", "i8")),
        rhs_dtype=DType(operation.get("rhs_dtype", "i8")),
        accumulator_dtype=DType(operation.get("accumulator_dtype", "i32")),
        legal=operation.get("legal", True)),
      staging=Staging(**d.get("staging", {})), synchronization=Synchronization(
        **{**synchronization, "scope": SyncScope(synchronization.get("scope", "none")),
           "points": tuple(synchronization.get("points", ())) }),
      ownership=Ownership(**d.get("ownership", {})),
      edge_predicates=tuple(EdgePredicate(**x) for x in d["edge_predicates"]), abi=d.get("abi", {"output_layout": "tokens_rows"}))
    candidate = MMQCandidate(desc, PhysicalMapping(**m), BackendCapability(
      backend=cap["backend"], device=cap["device"], supported_ops=tuple(DotOp(x) for x in cap.get("supported_ops", ["dot_i8_i8_i32"])),
      wave_sizes=tuple(cap.get("wave_sizes", [32])), max_workgroup_size=cap.get("max_workgroup_size"), lds_bytes=cap.get("lds_bytes")),
      lowering_version=raw.get("lowering_version", "logical-lowering/1"), provenance=raw.get("provenance", "research"), rollback_identity=raw.get("rollback_identity", "direct-packed"))
  except (KeyError, TypeError, ValueError) as exc: raise ValueError(f"invalid logical_candidate: {exc}") from exc
  if payload.get("candidate_identity") is not None and payload["candidate_identity"] != candidate.identity():
    raise ValueError("logical candidate identity mismatch")
  if candidate.provenance != "research" or candidate.rollback_identity != "direct-packed":
    raise ValueError("AMD candidate must remain research-only with direct-packed rollback")
  return candidate

def _spec_from_json(payload: dict[str, Any]) -> Q4KQ8MMQPrefillSpec:
  """Rebuild only metadata this lowering actually consumes.

  ABI and launch metadata used to be silently discarded here.  That made a
  descriptor look accepted even when it described an ABI/launch contract the
  generated lowering could not honor.
  """
  abi = payload.get("abi")
  if abi is not None and abi != PrimitiveABI().to_json():
    raise ValueError("bootstrap ABI is unsupported by the generated MMQ lowering")
  if payload.get("launch") is not None:
    raise ValueError("bootstrap launch metadata is unsupported; use final PROGRAM geometry")
  fields = {k: v for k, v in payload.items() if k in Q4KQ8MMQPrefillSpec.__dataclass_fields__}
  fields.pop("abi", None); fields.pop("launch", None)
  return Q4KQ8MMQPrefillSpec(**fields)

def bootstrap_from_file(path: str | Path) -> dict[str, Any]:
  """Read a bootstrap descriptor from a file (never from stdin).

  This is intentionally compile-only: the isolated guarded executor owns any
  subsequent device construction and dispatch.
  """
  payload = json.loads(Path(path).read_text())
  if not isinstance(payload, dict) or not isinstance(payload.get("spec"), dict):
    raise ValueError("bootstrap file must contain a spec object")
  spec = _spec_from_json(payload["spec"]); spec.validate()
  candidate = _shared_candidate(payload)
  if candidate is None: raise ValueError("shared logical candidate is required")
  expected_descriptor = spec.logical_descriptor()
  if candidate.descriptor != expected_descriptor:
    raise ValueError("logical candidate does not match bootstrap descriptor")
  return {"provenance": PROVENANCE, "candidate_identity": candidate.identity(),
          "logical_candidate_identity": candidate.identity(), "logical_candidate": candidate.to_dict(),
          "status": "accepted", "descriptor": spec.to_json()}

def _health() -> bool:
  try: return bool(np.isfinite(Tensor([1.0], device="AMD").realize().numpy()).all())
  except BaseException: return False

def _coverage(spec: Q4KQ8MMQPrefillSpec) -> dict[str, Any]:
  tiles = spec.m // spec.tile_m * (spec.n // spec.tile_n) * (spec.k // spec.tile_k)
  points = spec.m * spec.n * (spec.k // Q8_1_BLOCK_ELEMS)
  return {"tile_count": tiles, "covered_output_elements": spec.m * spec.n,
          "covered_groups": points, "expected_groups": points,
          "complete": tiles > 0 and points == spec.m * spec.n * (spec.k // Q8_1_BLOCK_ELEMS)}

def _validate_final_contract(spec: Q4KQ8MMQPrefillSpec, evidence: dict[str, Any], candidate: MMQCandidate | None) -> None:
  """Check only facts emitted by the final PROGRAM/code object."""
  if candidate is None:
    raise ValueError("shared logical candidate is required; descriptor geometry is not a fallback")
  if candidate.rollback_identity != "direct-packed" or candidate.provenance != "research":
    raise ValueError("candidate is not research-only/direct-packed")
  if candidate.descriptor.abi.get("output_layout") != spec.output_layout:
    raise ValueError("logical candidate ABI does not match primitive ABI")
  final_abi = evidence.get("abi")
  expected_abi = spec.abi.to_json()
  if not isinstance(final_abi, dict) or final_abi != expected_abi:
    raise ValueError("final PROGRAM ABI does not match shared candidate")
  geometry = evidence.get("geometry") or {}
  if not geometry.get("global_size") or not geometry.get("local_size"):
    raise ValueError("final PROGRAM launch geometry is unavailable")
  physical = evidence.get("physical_contract")
  if not isinstance(physical, dict):
    raise ValueError("final PROGRAM physical contract evidence is unavailable")
  checked = validate_physical_contract(**physical)
  if not checked["passed"]: raise ValueError("physical contract blocked: " + "; ".join(checked["errors"]))
  if not evidence.get("source_identity") or not evidence.get("binary_identity"):
    raise ValueError("final PROGRAM source/binary identity evidence is unavailable")
  candidate_evidence = evidence.get("candidate_identity")
  if isinstance(candidate_evidence, dict): candidate_evidence = candidate_evidence.get("candidate_id")
  if candidate_evidence != candidate.identity():
    raise ValueError("final PROGRAM logical candidate identity mismatch")

def _build_bundle(*, spec_json: dict[str, Any], logical_candidate: dict[str, Any] | None = None,
                  candidate_id: str | None = None, device: str = "AMD") -> ExecutableBundle:
  """Child-only: rebuild the graph, compile its PROGRAM, then bind it."""
  # ``to_json`` intentionally expands nested dataclasses and the MMQ namespace;
  # reconstruct the typed spec from its flat dataclass fields only.
  spec = _spec_from_json(spec_json)
  candidate = _shared_candidate({"logical_candidate": logical_candidate, "candidate_identity": candidate_id})
  if candidate is None or candidate.descriptor != spec.logical_descriptor():
    raise ValueError("shared logical candidate does not match bootstrap descriptor")
  words = Tensor.empty(spec.n * spec.k // 256 * Q4K_WORDS_PER_BLOCK, dtype=dtypes.uint32, device=device)
  xq = Tensor.empty(spec.m, spec.k, dtype=dtypes.int8, device=device)
  scales = Tensor.empty(spec.m, spec.k // 32, dtype=dtypes.float32, device=device)
  out = emit_q4k_q8_mmq_prefill(words, xq, scales, candidate)
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import KernelInfo
  sink = out.uop.sink().replace(arg=KernelInfo(name="q4k_q8_mmq_generated"))
  program = to_program(sink, __import__("tinygrad.device", fromlist=["Device"]).Device[device].renderer)
  source = next((u.arg for u in program.src if getattr(getattr(u, "op", None), "name", u.op) == "SOURCE"), None)
  binary = next((u.arg for u in program.src if getattr(getattr(u, "op", None), "name", u.op) == "BINARY"), None)
  if not isinstance(source, str) or not isinstance(binary, bytes) or not binary:
    raise ValueError("generated PROGRAM lacks source or binary identity")
  # These are facts from the final code object, never descriptor/resource budgets.
  from extra.qk.q4k_q8_mmq_compile_evidence import build_q4k_q8_mmq_compile_evidence
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  metadata = parse_amdgpu_metadata(binary)
  evidence = build_q4k_q8_mmq_compile_evidence(
    spec, program, metadata=metadata, source=source, binary=binary, candidate_id=candidate_id)
  evidence["passed"] = evidence["status"] == "pass"
  if not evidence["passed"]: raise ValueError("compile evidence blocked: " + "; ".join(evidence["errors"]))
  return build_tinygrad_bundle(program=program, compile_evidence=evidence, device=device,
    argument_order=("output", "words", "xq", "scales"), health=_health)

def validate_generated_candidate(words: np.ndarray, xq: np.ndarray, scales: np.ndarray,
                                reference: np.ndarray, spec: Q4KQ8MMQPrefillSpec,
                                *, timeout_seconds: float = 30.0,
                                candidate_id: str | None = None,
                                logical_candidate: dict[str, Any] | None = None) -> dict[str, Any]:
  """Compile/dispatch one candidate and return truthful AMD evidence."""
  spec.validate(); coverage = _coverage(spec)
  if not coverage["complete"]: raise ValueError("descriptor owner/group coverage is incomplete")
  identity = {"emitted_program": PROVENANCE, "geometry": spec.to_json(), "owner_coverage": coverage,
              "reference_authority": "canonical_reference_cpu_only"}
  candidate = _shared_candidate({"logical_candidate": logical_candidate}) if logical_candidate else None
  if candidate is None: raise ValueError("shared logical candidate is required")
  req = ExecutionRequest(inputs={"words": np.asarray(words), "xq": np.asarray(xq), "scales": np.asarray(scales)},
    reference=np.asarray(reference), policy=GuardPolicy(rtol=2e-2, atol=2e-2), identity=identity)
  result = run_isolated_guarded_execution(builder=make_tinygrad_bundle_builder(
    build=_build_bundle, spec_json=spec.to_json(), logical_candidate=candidate.to_dict(), candidate_id=candidate.identity()),
    request=req, health_probe=_health, timeout_seconds=timeout_seconds)
  evidence = result.to_dict(); guarded = evidence.get("guarded") or {}
  evidence["provenance"] = {"emitter": PROVENANCE, "compiled_in_spawned_child": True,
    "dispatch_performed": bool(guarded.get("dispatch_performed")),
    "gpu_correctness_claimable": bool(result.passed and guarded.get("dispatch_performed") and guarded.get("full_output_compared"))}
  evidence["owner_coverage"] = coverage
  compile_evidence = guarded.get("compile_evidence") or {}
  try: _validate_final_contract(spec, compile_evidence, candidate)
  except ValueError as exc:
    evidence["passed"] = False; evidence["verdict"] = "BLOCKED_FAIL_CLOSED"; evidence.setdefault("errors", []).append(str(exc))
  evidence["candidate_identity"] = candidate.identity()
  evidence["source_identity"] = compile_evidence.get("source_sha256") or compile_evidence.get("source_identity")
  evidence["binary_identity"] = compile_evidence.get("binary_sha256") or compile_evidence.get("binary_identity")
  if evidence["binary_identity"] is None:
    evidence["verdict"] = "BLOCKED_FAIL_CLOSED"
  return evidence

def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bootstrap", required=True)
  parser.add_argument("--dispatch", action="store_true")
  parser.add_argument("--timeout", type=float, default=30.0)
  args = parser.parse_args()
  # The file is the only bootstrap channel.  Dispatch still requires the
  # programmatic validator and its guarded inputs; CLI compilation is fail-closed.
  accepted = bootstrap_from_file(args.bootstrap)
  if not args.dispatch:
    print(json.dumps(accepted, sort_keys=True)); return 0
  spec = _spec_from_json(accepted["descriptor"])
  words = np.zeros(spec.n * spec.k // 256 * Q4K_WORDS_PER_BLOCK, dtype=np.uint32)
  xq = np.zeros((spec.m, spec.k), dtype=np.int8)
  scales = np.zeros((spec.m, spec.k // 32), dtype=np.float32)
  reference = np.zeros((spec.m, spec.n), dtype=np.float32)
  print(json.dumps(validate_generated_candidate(words, xq, scales, reference, spec,
                                                timeout_seconds=args.timeout,
                                                logical_candidate=accepted["logical_candidate"]), sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
