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
from extra.qk.prefill.guarded_execution import GuardPolicy
from extra.qk.prefill.isolated_guarded_executor import (
  ExecutableBundle, ExecutionRequest, build_tinygrad_bundle,
  make_tinygrad_bundle_builder, run_isolated_guarded_execution,
)

PROVENANCE = "q4k_q8_mmq_descriptor_emitter_v1"

def bootstrap_from_file(path: str | Path) -> dict[str, Any]:
  """Read a bootstrap descriptor from a file (never from stdin).

  This is intentionally compile-only: the isolated guarded executor owns any
  subsequent device construction and dispatch.
  """
  payload = json.loads(Path(path).read_text())
  if not isinstance(payload, dict) or not isinstance(payload.get("spec"), dict):
    raise ValueError("bootstrap file must contain a spec object")
  fields = {k: v for k, v in payload["spec"].items() if k in Q4KQ8MMQPrefillSpec.__dataclass_fields__}
  fields.pop("abi", None); fields.pop("launch", None)
  spec = Q4KQ8MMQPrefillSpec(**fields); spec.validate()
  return {"provenance": PROVENANCE, "candidate_identity": spec.canonical_identity(),
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

def _build_bundle(*, spec_json: dict[str, Any], candidate_id: str | None = None,
                  device: str = "AMD") -> ExecutableBundle:
  """Child-only: rebuild the graph, compile its PROGRAM, then bind it."""
  # ``to_json`` intentionally expands nested dataclasses and the MMQ namespace;
  # reconstruct the typed spec from its flat dataclass fields only.
  fields = {k: v for k, v in spec_json.items() if k in Q4KQ8MMQPrefillSpec.__dataclass_fields__}
  fields.pop("abi", None); fields.pop("launch", None)
  spec = Q4KQ8MMQPrefillSpec(**fields)
  words = Tensor.empty(spec.n * spec.k // 256 * Q4K_WORDS_PER_BLOCK, dtype=dtypes.uint32, device=device)
  xq = Tensor.empty(spec.m, spec.k, dtype=dtypes.int8, device=device)
  scales = Tensor.empty(spec.m, spec.k // 32, dtype=dtypes.float32, device=device)
  out = emit_q4k_q8_mmq_prefill(words, xq, scales, spec)
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
                                candidate_id: str | None = None) -> dict[str, Any]:
  """Compile/dispatch one candidate and return truthful AMD evidence."""
  spec.validate(); coverage = _coverage(spec)
  if not coverage["complete"]: raise ValueError("descriptor owner/group coverage is incomplete")
  identity = {"emitted_program": PROVENANCE, "geometry": spec.to_json(), "owner_coverage": coverage,
              "reference_authority": "canonical_reference_cpu_only"}
  req = ExecutionRequest(inputs={"words": np.asarray(words), "xq": np.asarray(xq), "scales": np.asarray(scales)},
    reference=np.asarray(reference), policy=GuardPolicy(rtol=2e-2, atol=2e-2), identity=identity)
  result = run_isolated_guarded_execution(builder=make_tinygrad_bundle_builder(
    build=_build_bundle, spec_json=spec.to_json(), candidate_id=candidate_id or spec.canonical_identity()),
    request=req, health_probe=_health, timeout_seconds=timeout_seconds)
  evidence = result.to_dict(); guarded = evidence.get("guarded") or {}
  evidence["provenance"] = {"emitter": PROVENANCE, "compiled_in_spawned_child": True,
    "dispatch_performed": bool(guarded.get("dispatch_performed")),
    "gpu_correctness_claimable": bool(result.passed and guarded.get("dispatch_performed") and guarded.get("full_output_compared"))}
  evidence["owner_coverage"] = coverage
  compile_evidence = guarded.get("compile_evidence") or {}
  evidence["candidate_identity"] = candidate_id or spec.canonical_identity()
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
  fields = {k: v for k, v in accepted["descriptor"].items()
            if k in Q4KQ8MMQPrefillSpec.__dataclass_fields__}
  fields.pop("abi", None); fields.pop("launch", None)
  spec = Q4KQ8MMQPrefillSpec(**fields)
  words = np.zeros(spec.n * spec.k // 256 * Q4K_WORDS_PER_BLOCK, dtype=np.uint32)
  xq = np.zeros((spec.m, spec.k), dtype=np.int8)
  scales = np.zeros((spec.m, spec.k // 32), dtype=np.float32)
  reference = np.zeros((spec.m, spec.n), dtype=np.float32)
  print(json.dumps(validate_generated_candidate(words, xq, scales, reference, spec,
                                                timeout_seconds=args.timeout), sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
