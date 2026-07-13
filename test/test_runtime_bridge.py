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


def test_nonempty_aux_program_still_constructs_with_aux_stripped_for_runtime():
  # Regression: get_runtime unpacks ProgramInfo.aux positionally, but the AMD
  # runtime constructor takes no positional aux.  A program carrying a non-empty
  # aux (e.g. the generated ISA source) must still construct a handle, and the
  # program handed to get_runtime must have aux == () so no positional aux leaks.
  class Runtime:
    lib = b"ok"
    def __call__(self, *args, **kwargs): raise AssertionError("dispatch must be explicit")
  seen = {}
  def fake_get_runtime(device, prg, cache=True): seen["aux"] = prg.arg.aux; return Runtime()
  prg = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU"), UOp(Ops.LINEAR),
                              UOp(Ops.SOURCE), UOp(Ops.BINARY, arg=b"ok")),
            arg=ProgramInfo(global_size=(8, 1, 1), local_size=(64, 1, 1), aux=("large-isa-source",)))
  with patch("tinygrad.runtime.bridge.get_runtime", side_effect=fake_get_runtime):
    handle = build_executable(artifact(), prg, device="CPU")
  assert seen["aux"] == ()
  # The handle keeps the original program (aux preserved) for launch metadata.
  assert handle.program.arg.aux == ("large-isa-source",)


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
