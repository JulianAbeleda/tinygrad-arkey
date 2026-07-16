import pytest
from types import SimpleNamespace

from extra.qk.prefill.execution_bridge_contracts import TransportPlan
from tinygrad.uop.ops import Ops, ProgramInfo, UOp
from extra.qk.prefill.transport_execution_authority import (
  _TRANSPORT_VALIDATORS, TransportValidation, register_transport_validator, validate_transport)


def _program(*, capture=None, local=False, source="v_wmma_f32_16x16x16_f16"):
  sink = UOp(Ops.SINK, src=(UOp(Ops.DEFINE_LOCAL) ,)) if local else UOp(Ops.SINK)
  class Attachment:
    def __init__(self, record): self.record = record
  info = ProgramInfo(aux=() if capture is None else (Attachment(capture),))
  return UOp(Ops.PROGRAM, src=(sink,), arg=info)


def _plan(transport):
  return TransportPlan(transport=transport, schedule_digest="a" * 64)


def test_direct_transport_uses_shared_validator_and_requires_final_capture():
  payload = {"schedule": {"residency": {"resident": ["stage_ab_register"]}}}
  missing = validate_transport(payload, _program(), plan=_plan("direct_l2"))
  assert not missing.passed and "capture" in missing.errors[0]
  capture = {"descriptor": {"resources": {"lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0}},
             "allocator": {"authority": "final_regalloc"}}
  assert validate_transport(payload, _program(capture=capture), plan=_plan("direct_l2")).passed


def test_direct_transport_rejects_local_allocation():
  payload = {"schedule": {"residency": {"resident": ["stage_ab_register"]}}}
  row = validate_transport(payload, _program(local=True, capture={"descriptor": {"resources": {"lds_bytes": 0,
    "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0}}, "allocator": {"authority": "final_regalloc"}}),
    plan=_plan("direct_l2"))
  assert not row.passed and any("DEFINE_LOCAL" in x for x in row.errors)


def test_lds_transport_dispatches_through_the_explicit_registry():
  called = []
  def validator(payload, program):
    called.append(True)
    return TransportValidation("lds", True, (), {"storage": "lds"})
  saved = _TRANSPORT_VALIDATORS.get("lds")
  register_transport_validator("lds", validator)
  try:
    row = validate_transport({"schedule": {"residency": {"resident": ["accumulator"]}}}, _program(), plan=_plan("lds"))
  finally:
    if saved is None: _TRANSPORT_VALIDATORS.pop("lds", None)
    else: _TRANSPORT_VALIDATORS["lds"] = saved
  assert row.passed and called == [True]


def test_unknown_transport_is_rejected_fail_closed():
  # An unregistered transport must never silently default to LDS.
  with pytest.raises(ValueError, match="unknown transport"):
    validate_transport({"schedule": {}}, _program(), plan=_plan("nvlink_smem"))
