import hashlib
from unittest.mock import patch

import pytest

from tinygrad.runtime.bridge import CompileArtifact, build_executable, prepare_executable
from tinygrad.uop.ops import Ops, ProgramInfo, UOp


def program(binary):
  return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU"), UOp(Ops.LINEAR), UOp(Ops.SOURCE), UOp(Ops.BINARY, arg=binary)), arg=ProgramInfo())


def artifact(binary=b"ok", **kw):
  return CompileArtifact(binary, hashlib.sha256(binary).hexdigest(), **kw)


def test_missing_or_mismatched_artifact_fails_closed_without_runtime():
  with patch("tinygrad.runtime.bridge.get_runtime") as runtime:
    with pytest.raises(ValueError): build_executable({}, program(b"ok"))
    with pytest.raises(ValueError): build_executable(artifact(b"bad"), program(b"ok"))
    runtime.assert_not_called()


def test_mock_runtime_is_executable_but_not_dispatched_on_construction():
  class Runtime:
    lib = b"ok"
    def __init__(self): self.calls = 0; self.kwargs = None
    def __call__(self, *args, **kwargs): self.calls += 1; self.kwargs = kwargs; return "done"
  rt = Runtime()
  prg = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU"), UOp(Ops.LINEAR),
                              UOp(Ops.SOURCE), UOp(Ops.BINARY, arg=b"ok")),
            arg=ProgramInfo(global_size=(8, 1, 1), local_size=(64, 1, 1)))
  with patch("tinygrad.runtime.bridge.get_runtime", return_value=rt) as resolver:
    handle = build_executable(artifact(), prg, device="CPU")
  resolver.assert_called_once()
  assert rt.calls == 0
  # P1-2: __call__ is a pure alias for dispatch and therefore uses the PROGRAM
  # launch geometry; there is no alternate default-geometry dispatch path.
  assert handle() == "done"
  assert rt.calls == 1
  assert rt.kwargs == {"global_size": (8, 1, 1), "local_size": (64, 1, 1), "vals": (), "wait": True}
  assert handle.dispatch() == "done"
  assert rt.kwargs["global_size"] == (8, 1, 1)


def test_explicit_dispatch_uses_program_launch_geometry():
  class Runtime:
    lib = b"ok"
    def __init__(self): self.args = None
    def __call__(self, *args, **kwargs): self.args = (args, kwargs); return "launched"
  rt = Runtime()
  prg = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU"), UOp(Ops.LINEAR),
                              UOp(Ops.SOURCE), UOp(Ops.BINARY, arg=b"ok")),
            arg=ProgramInfo(global_size=(32, 4, 1), local_size=(256, 1, 1)))
  with patch("tinygrad.runtime.bridge.get_runtime", return_value=rt):
    handle = build_executable(artifact(), prg, device="CPU")
  assert handle.dispatch("a", "b", "c") == "launched"
  assert rt.args == (("a", "b", "c"), {"global_size": (32, 4, 1), "local_size": (256, 1, 1), "vals": (), "wait": True})


def test_prepare_executable_joins_compile_evidence_to_program_without_dispatch():
  class Runtime:
    lib = b"ok"
    def __call__(self, *args, **kwargs): raise AssertionError("dispatch must be explicit")
  with patch("tinygrad.runtime.bridge.get_runtime", return_value=Runtime()):
    handle = prepare_executable(program(b"ok"), {"passed": True, "binary_sha256": hashlib.sha256(b"ok").hexdigest()}, device="CPU")
  assert handle.artifact.binary == b"ok"
  with patch("tinygrad.runtime.bridge.get_runtime") as resolver:
    with pytest.raises(ValueError, match="identity"):
      prepare_executable(program(b"ok"), {"passed": True, "binary_sha256": "f" * 64}, device="CPU")
  resolver.assert_not_called()
