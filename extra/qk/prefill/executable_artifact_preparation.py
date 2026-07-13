"""Join compiler-owned capture evidence to a real tinygrad PROGRAM.

This is the narrow route seam between compile-only proof and the existing
runtime bridge.  It does not lower, allocate, or dispatch by itself.
"""
from __future__ import annotations

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


__all__ = ["capture_record", "compile_evidence", "prepare_executable_artifact"]
