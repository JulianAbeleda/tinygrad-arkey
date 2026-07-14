import pytest

from extra.qk.prefill.executable_artifact_preparation import capture_record, compile_transport_evidence
from tinygrad.codegen import _CompileCaptureAttachment
from tinygrad.uop.ops import Ops, ProgramInfo, UOp


def _program(aux=(), local_size=None):
  return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU"),
    UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg="source"), UOp(Ops.BINARY, arg=b"binary")),
    arg=ProgramInfo(local_size=local_size, aux=aux))


def test_capture_record_uses_compiler_owned_attachment():
  record = {"binary": b"binary", "candidate_identity": "a" * 64}
  assert capture_record(_program((_CompileCaptureAttachment(record),))) is record


def test_capture_record_rejects_missing_or_non_program_input():
  with pytest.raises(ValueError, match="capture"):
    capture_record(_program())
  with pytest.raises(ValueError, match="PROGRAM"):
    capture_record(UOp(Ops.SINK))


def test_transport_evidence_accepts_multidimensional_local_size_with_exact_thread_product():
  evidence = compile_transport_evidence(_program(local_size=(32, 4, 2)), transport="lds",
    canonical_identity="a" * 64, schedule={"threads": 256, "lds_bytes": 40960}, surface={})
  assert evidence["passed"] is True
  mismatch = compile_transport_evidence(_program(local_size=(32, 4, 1)), transport="lds",
    canonical_identity="a" * 64, schedule={"threads": 256, "lds_bytes": 40960}, surface={})
  assert mismatch["passed"] is False
