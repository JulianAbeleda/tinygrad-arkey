"""Compile/resource evidence for descriptor-generated Q4_K x Q8_1 MMQ candidates.

This is deliberately a metadata consumer.  It does not lower a descriptor or
inspect ISA; absent compiler metadata is recorded as unavailable (and cannot
produce a passing evidence row).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SCHEMA = "tinygrad.q4k_q8_mmq_compile_evidence.v1"


def _sha256(value: str | bytes) -> str:
  return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _program_value(program: Any, name: str) -> Any:
  if isinstance(program, Mapping): return program.get(name)
  return getattr(getattr(program, "arg", None), name, None)


def _json(value: Any) -> Any:
  if value is None or isinstance(value, (str, int, float, bool)): return value
  if isinstance(value, Mapping): return {str(k): _json(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)): return [_json(v) for v in value]
  return None


def build_q4k_q8_mmq_compile_evidence(spec: Any, program: Any, *, metadata: Mapping[str, Any] | None = None,
                                      source: str | bytes | None = None, binary: bytes | None = None,
                                      instruction_summary: Mapping[str, Any] | None = None,
                                      candidate_id: str | None = None) -> dict[str, Any]:
  """Build one fail-closed evidence row from a descriptor and emitted program.

  ``metadata`` is the compiler/code-object resource metadata supplied by the
  emitter.  No resource value is guessed from descriptor budgets or source.
  """
  errors: list[str] = []
  try:
    spec.validate()
    canonical = spec.canonical_identity()
    descriptor = spec.to_json()
  except Exception as exc:
    canonical, descriptor = None, None
    errors.append(f"invalid descriptor: {exc}")

  global_size, local_size = _program_value(program, "global_size"), _program_value(program, "local_size")
  function_name = _program_value(program, "function_name")
  if not global_size or not local_size: errors.append("program launch geometry unavailable")
  if not function_name: errors.append("program function identity unavailable")
  metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
  lowering = metadata.get("lowering") or metadata.get("lowering_strategy")
  backend = metadata.get("backend") or metadata.get("target", {}).get("backend") if isinstance(metadata.get("target"), Mapping) else metadata.get("backend")
  if not isinstance(lowering, str) or not lowering: errors.append("lowering provenance unavailable")
  if not isinstance(backend, str) or not backend: errors.append("backend provenance unavailable")
  # Emitters commonly wrap compiler facts under ``resources`` and launch facts
  # under ``launch``.  Flatten only those known containers; retain the original
  # metadata below for provenance.
  resource_metadata = metadata.get("resources")
  if isinstance(resource_metadata, Mapping): metadata = {**metadata, **resource_metadata}
  launch_metadata = metadata.get("launch")
  if isinstance(launch_metadata, Mapping):
    global_size = global_size or launch_metadata.get("global_size") or launch_metadata.get("grid")
    local_size = local_size or launch_metadata.get("local_size")
  resource_keys = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills",
                   "dynamic_stack", "wavefront_size")
  resources = {key: metadata.get(key) for key in resource_keys}
  missing = [key for key in resource_keys if resources[key] is None]
  if missing: errors.append("compiler resource metadata unavailable: " + ", ".join(missing))
  if source is None: errors.append("rendered source unavailable")
  if binary is None: errors.append("code object unavailable")
  source_bytes = source.encode() if isinstance(source, str) else source
  identity = {"canonical_sha256": canonical,
              "source_sha256": _sha256(source_bytes) if source_bytes is not None else None,
              "binary_sha256": _sha256(binary) if binary is not None else None}
  # Keep the audit vocabulary explicit: these are independent identities, and
  # absent artifacts must never be represented by a route/default identity.
  provenance = {"lowering": lowering, "backend": backend,
                "function_name": function_name, "metadata": _json(metadata)}
  candidate_identity = {"candidate_id": candidate_id, "descriptor": descriptor,
                        "descriptor_sha256": canonical, "provenance": provenance,
                        "lowering": lowering, "backend": backend}
  if not candidate_id: errors.append("candidate identity unavailable")
  if not candidate_identity: errors.append("candidate identity unavailable")
  if identity["source_sha256"] is None: errors.append("source identity unavailable")
  if identity["binary_sha256"] is None: errors.append("binary identity unavailable")
  abi = descriptor.get("abi") if descriptor else None
  if isinstance(metadata.get("abi"), Mapping): abi = _json(metadata["abi"])
  row = {"schema": SCHEMA, "status": "pass" if not errors else "blocked", "errors": errors,
         "canonical_identity": canonical, "candidate_id": candidate_id,
         "candidate_identity": candidate_identity, "source_identity": identity["source_sha256"],
         "binary_identity": identity["binary_sha256"], "descriptor": descriptor,
         "identity": identity, "function_name": function_name,
         "abi": abi, "geometry": {"global_size": _json(global_size), "local_size": _json(local_size)},
         "resources": resources, "instruction_summary": _json(instruction_summary),
         "metadata": _json(metadata)}
  return row


def capture_q4k_q8_mmq_compile_evidence(spec: Any, program: Any, **kwargs: Any) -> dict[str, Any]:
  """Compatibility-shaped capture entry point for emitted-program callers."""
  return build_q4k_q8_mmq_compile_evidence(spec, program, **kwargs)


__all__ = ["SCHEMA", "build_q4k_q8_mmq_compile_evidence", "capture_q4k_q8_mmq_compile_evidence"]
