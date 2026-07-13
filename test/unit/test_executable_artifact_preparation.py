import pytest

from extra.qk.prefill.executable_artifact_preparation import capture_record
from tinygrad.codegen import _CompileCaptureAttachment
from tinygrad.uop.ops import Ops, ProgramInfo, UOp


def _program(aux=()):
  return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU"),
    UOp(Ops.LINEAR), UOp(Ops.SOURCE), UOp(Ops.BINARY, arg=b"binary")), arg=ProgramInfo(aux=aux))


def test_capture_record_uses_compiler_owned_attachment():
  record = {"binary": b"binary", "candidate_identity": "a" * 64}
  assert capture_record(_program((_CompileCaptureAttachment(record),))) is record


def test_capture_record_rejects_missing_or_non_program_input():
  with pytest.raises(ValueError, match="capture"):
    capture_record(_program())
  with pytest.raises(ValueError, match="PROGRAM"):
    capture_record(UOp(Ops.SINK))
