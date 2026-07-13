from types import SimpleNamespace

from tinygrad.uop.ops import Ops, ProgramInfo, UOp
from extra.qk.prefill.transport_execution_authority import validate_transport


def _program(*, capture=None, local=False, source="v_wmma_f32_16x16x16_f16"):
  sink = UOp(Ops.SINK, src=(UOp(Ops.DEFINE_LOCAL) ,)) if local else UOp(Ops.SINK)
  class Attachment:
    def __init__(self, record): self.record = record
  info = ProgramInfo(aux=() if capture is None else (Attachment(capture),))
  return UOp(Ops.PROGRAM, src=(sink,), arg=info)


def test_direct_transport_uses_shared_validator_and_requires_final_capture():
  payload = {"schedule": {"residency": {"resident": ["stage_ab_register"]}}}
  missing = validate_transport(payload, _program())
  assert not missing.passed and "capture" in missing.errors[0]
  capture = {"descriptor": {"resources": {"lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0}},
             "allocator": {"authority": "final_regalloc"}}
  assert validate_transport(payload, _program(capture=capture)).passed


def test_direct_transport_rejects_local_allocation():
  payload = {"schedule": {"residency": {"resident": ["stage_ab_register"]}}}
  row = validate_transport(payload, _program(local=True, capture={"descriptor": {"resources": {"lds_bytes": 0,
    "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0}}, "allocator": {"authority": "final_regalloc"}}))
  assert not row.passed and any("DEFINE_LOCAL" in x for x in row.errors)


def test_lds_transport_delegates_to_shared_authority():
  called = []
  def validator(payload, program):
    called.append(True)
    from extra.qk.prefill.transport_execution_authority import TransportValidation
    return TransportValidation("lds", True, (), {"storage": "lds"})
  row = validate_transport({"schedule": {"residency": {"resident": ["accumulator"]}}}, _program(), lds_validator=validator)
  assert row.passed and called == [True]
