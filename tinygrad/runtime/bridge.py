"""Small, default-closed bridge from validated compile artifacts to runtimes.

Construction only resolves the runtime.  Dispatch remains the explicit call on
the returned handle.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from tinygrad.engine.realize import get_runtime
from tinygrad.uop.ops import Ops, ProgramInfo, UOp


@dataclass(frozen=True)
class CompileArtifact:
  """The already-gated part of a compile result needed by this bridge."""
  binary: bytes
  binary_sha256: str
  validated: bool = True


@dataclass
class ExecutableHandle:
  """An in-process runtime; calling it is the sole dispatch operation."""
  runtime: Any
  program: UOp
  artifact: CompileArtifact

  def __call__(self, *args, **kwargs):
    return self.runtime(*args, **kwargs)

  def dispatch(self, *buffers, vals=(), wait=True):
    """Explicitly dispatch using the launch geometry from this PROGRAM.

    Callers supply only the compiled program's buffer arguments.  Geometry is
    never reconstructed from a candidate report or left at the runtime's
    default `(1, 1, 1)` launch.  This method is intentionally named separately
    from construction so preparing an executable cannot dispatch by accident.
    """
    if self.program.op is not Ops.PROGRAM or not isinstance(self.program.arg, ProgramInfo):
      raise ValueError("executable program metadata is unavailable")
    return self.runtime(*buffers, global_size=self.program.arg.global_size,
                        local_size=self.program.arg.local_size, vals=vals, wait=wait)

  def close(self) -> None:
    close = getattr(self.runtime, "close", None)
    if close is not None: close()


def _artifact(value: CompileArtifact | Mapping[str, Any]) -> CompileArtifact:
  if isinstance(value, CompileArtifact): return value
  try: return CompileArtifact(binary=value["binary"], binary_sha256=value["binary_sha256"], validated=value["validated"])
  except (KeyError, TypeError) as e: raise ValueError("incomplete compile artifact") from e


def build_executable(artifact: CompileArtifact | Mapping[str, Any], program: UOp | ProgramInfo,
                     device: str = "AMD") -> ExecutableHandle:
  """Resolve a checked artifact and its original program without dispatching.

  The artifact must explicitly be validated.  A PROGRAM UOp is required so
  that the exact binary consumed by tinygrad can be compared with the artifact.
  """
  artifact = _artifact(artifact)
  if not artifact.validated: raise ValueError("compile artifact is not validated")
  if not isinstance(artifact.binary, bytes): raise ValueError("artifact binary must be bytes")
  actual = hashlib.sha256(artifact.binary).hexdigest()
  if actual != artifact.binary_sha256: raise ValueError("compile artifact binary hash mismatch")

  if isinstance(program, ProgramInfo):
    raise ValueError("original program must be the compiled PROGRAM UOp")
  if program.op is not Ops.PROGRAM or not isinstance(program.arg, ProgramInfo) or len(program.src) < 5:
    raise ValueError("program must be a compiled PROGRAM UOp")
  if program.src[4].arg != artifact.binary:
    raise ValueError("compile artifact binary does not match program")

  runtime = get_runtime(device, program, cache=False)
  loaded = getattr(runtime, "lib", None)
  if loaded is None or not isinstance(loaded, bytes) or hashlib.sha256(loaded).hexdigest() != artifact.binary_sha256:
    close = getattr(runtime, "close", None)
    if close is not None: close()
    raise ValueError("runtime binary hash mismatch")
  return ExecutableHandle(runtime, program, artifact)


def prepare_executable(program: UOp, compile_evidence: Mapping[str, Any], device: str = "AMD") -> ExecutableHandle:
  """Promote a gated compile-only evidence record to a runtime handle.

  The original compiled PROGRAM remains the launch-ABI authority.  The
  evidence record only admits it; it never supplies a substitute binary or
  launch contract.  Constructing the handle may allocate the code object on a
  live device, but it never calls the program and therefore never dispatches.
  """
  if not isinstance(compile_evidence, Mapping) or compile_evidence.get("passed") is not True:
    raise ValueError("passing compile evidence is required")
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("compiled PROGRAM UOp is required")
  binary = next((u.arg for u in program.src if u.op is Ops.BINARY and isinstance(u.arg, bytes)), None)
  if not isinstance(binary, bytes) or not binary: raise ValueError("compiled PROGRAM has no final binary")
  expected = compile_evidence.get("binary_sha256")
  if not isinstance(expected, str) or hashlib.sha256(binary).hexdigest() != expected:
    raise ValueError("compile evidence binary identity does not match PROGRAM")
  return build_executable(CompileArtifact(binary, expected, validated=True), program, device=device)
