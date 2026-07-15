import pytest

from extra.qk.mmq_role_adapter import (
  MMQ_FFN_GATE_UP_ROUTE_ID, MMQRoleAdmissionError,
  admit_ffn_gate_up_mmq_candidate, canonical_mmq_candidate_identity,
)
from extra.qk.mmq_capability import MMQRequest


def _candidate():
  return {"candidate_id": "research.mmq.ffn_gate_up.v1",
          "route_id": MMQ_FFN_GATE_UP_ROUTE_ID, "role": "ffn_gate_up",
          "quant_format": "Q4_K", "activation_format": "Q8_1",
          "evidence": {f"M{i}": True for i in range(1, 8)}}


def test_ffn_gate_up_mmq_candidate_is_admitted_with_canonical_identity():
  payload = _candidate()
  identity = canonical_mmq_candidate_identity(payload)
  admitted = admit_ffn_gate_up_mmq_candidate(payload, identity)
  assert admitted.canonical_identity == identity
  assert admitted.payload["role"] == "ffn_gate_up"
  assert admitted.payload["route_id"] == MMQ_FFN_GATE_UP_ROUTE_ID


@pytest.mark.parametrize("change", [
  lambda p: p["evidence"].pop("M6"),
  lambda p: p.update(role="ffn_down"),
  lambda p: p.update(evidence={f"M{i}": True for i in range(1, 7)} | {"M7": False}),
])
def test_ffn_gate_up_mmq_admission_blocks_without_contract_or_identity(change):
  payload = _candidate(); change(payload)
  with pytest.raises(MMQRoleAdmissionError):
    admit_ffn_gate_up_mmq_candidate(payload, canonical_mmq_candidate_identity(payload))


def test_ffn_gate_up_mmq_admission_blocks_identity_drift():
  with pytest.raises(MMQRoleAdmissionError, match="identity"):
    admit_ffn_gate_up_mmq_candidate(_candidate(), "0" * 64)

def test_role_is_request_data_not_an_admission_special_case():
  payload = _candidate(); payload["role"] = "ffn_down"
  identity = canonical_mmq_candidate_identity(payload)
  admitted = __import__("extra.qk.mmq_role_adapter", fromlist=["admit_mmq_candidate"]).admit_mmq_candidate(
    payload, identity, request=MMQRequest("ffn_down"))
  assert admitted.payload["role"] == "ffn_down"
