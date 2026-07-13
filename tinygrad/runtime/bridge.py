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
