"""Join compiler-owned capture evidence to a real tinygrad PROGRAM.

This is the narrow route seam between compile-only proof and the existing
runtime bridge.  It does not lower, allocate, or dispatch by itself.
"""
from __future__ import annotations

import hashlib
from math import prod
from typing import Any, Mapping

from extra.qk.prefill.pure_register_compile_capture import capture_final_program_compile_only
from tinygrad.runtime.bridge import ExecutableHandle, prepare_executable
from tinygrad.uop.ops import Ops, UOp


def capture_record(program: UOp) -> Any:
  """Extract the backend-owned capture attached to a compiled PROGRAM."""
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("compiled PROGRAM UOp is required")
  aux = getattr(program.arg, "aux", ())
  records = [item.record for item in aux if hasattr(item, "record")]
  if not records: raise ValueError("compiled PROGRAM has no compiler-owned capture")
  return records[-1]


def compile_evidence(program: UOp | Mapping[str, Any], *, pipeline: Mapping[str, Any], wait: Mapping[str, Any],
                     abi_contract: Mapping[str, Any], surface: Mapping[str, Any],
                     runtime_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Convert the attached final capture into the existing gated evidence schema."""
  record = capture_record(program) if isinstance(program, UOp) else program
  return capture_final_program_compile_only(record, pipeline=pipeline, wait=wait,
                                            abi_contract=abi_contract, surface=surface,
                                            runtime_binding=runtime_binding)


def compile_transport_evidence(program: UOp, *, transport: str, canonical_identity: str,
                               schedule: Mapping[str, Any], surface: Mapping[str, Any],
                               runtime_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Create compile-only evidence for a non-capture transport.

  The legacy LDS2 route is a compiler input stream assembled by the existing
  AMD instruction generator, so it does not carry the pure-register capture
  attachment.  It still has a final tinygrad PROGRAM, source, binary, launch
  geometry, and an exact candidate identity.  Keep that evidence in a generic
  transport schema instead of weakening the pure-register gate or inventing a
  capture record.
  """
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("compiled PROGRAM UOp is required")
  if not isinstance(transport, str) or not transport.strip(): raise ValueError("transport is required")
  if not isinstance(canonical_identity, str) or len(canonical_identity) != 64:
    raise ValueError("canonical candidate identity must be a SHA-256 string")
  if not isinstance(schedule, Mapping) or not isinstance(surface, Mapping):
    raise TypeError("schedule and surface must be mappings")
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), None)
  binary = next((u.arg for u in program.src if u.op is Ops.BINARY and isinstance(u.arg, bytes)), None)
  if not source: raise ValueError("compiled PROGRAM has no final source")
  if not binary: raise ValueError("compiled PROGRAM has no final binary")
  launch = program.arg
  errors = []
  local_size = getattr(launch, "local_size", None)
  if not isinstance(local_size, tuple) or prod(local_size) != int(schedule.get("threads", 0)):
    errors.append("compiled local size does not match schedule threads")
  if int(schedule.get("lds_bytes", 0)) < 0: errors.append("schedule LDS bytes must be non-negative")
  row = {"schema": "prefill-transport-compile.v1", "transport": transport,
         "canonical_identity": canonical_identity, "binary_sha256": hashlib.sha256(binary).hexdigest(),
         "passed": not errors, "errors": errors,
         "program": {"name": getattr(launch, "name", None),
                     "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                     "binary_sha256": hashlib.sha256(binary).hexdigest(),
                     "target": next((u.arg for u in program.src if u.op is Ops.DEVICE), None),
                     "abi": "amdgpu_kernel"},
         "schedule": dict(schedule), "surface": dict(surface),
         "capture": {"mode": "compile_only", "dispatch_permitted": False,
                     "resource_authority": "compiled_program_descriptor"}}
  if runtime_binding is not None: row["runtime_binding"] = dict(runtime_binding)
  return row


def prepare_executable_artifact(program: UOp, *, pipeline: Mapping[str, Any], wait: Mapping[str, Any],
                                abi_contract: Mapping[str, Any], surface: Mapping[str, Any],
                                runtime_binding: Mapping[str, Any] | None = None,
                                device: str = "AMD") -> tuple[dict[str, Any], ExecutableHandle]:
  """Return passing evidence plus a non-dispatching runtime handle.

  The handle is created only after compile evidence passes.  Calling the
  handle is still the explicit dispatch operation.
  """
  evidence = compile_evidence(program, pipeline=pipeline, wait=wait, abi_contract=abi_contract,
                              surface=surface, runtime_binding=runtime_binding)
  if evidence.get("passed") is not True:
    raise ValueError("compiler capture evidence did not pass: " + "; ".join(evidence.get("errors", ())))
  return evidence, prepare_executable(program, evidence, device=device)


__all__ = ["capture_record", "compile_evidence", "compile_transport_evidence", "prepare_executable_artifact"]
