import hashlib
from unittest.mock import patch

import pytest

from tinygrad.runtime.bridge import CompileArtifact, build_executable
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
    def __init__(self): self.calls = 0
    def __call__(self, *args, **kwargs): self.calls += 1; return "done"
  rt = Runtime()
  with patch("tinygrad.runtime.bridge.get_runtime", return_value=rt) as resolver:
    handle = build_executable(artifact(), program(b"ok"), device="CPU")
  resolver.assert_called_once()
  assert rt.calls == 0
  assert handle() == "done"
  assert rt.calls == 1
