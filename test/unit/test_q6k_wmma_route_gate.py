import pytest

from extra.qk.q6k_wmma_route_gate import (Q6K_WMMA_CANDIDATE_ID, Q6K_WMMA_EVIDENCE_ID,
  Q6K_WMMA_FIXTURE_SHAPE, Q6K_WMMA_LIVE_BYTES, Q6K_WMMA_REQUIRED_LDS_BYTES, Q6K_WMMA_ROLE_SHAPE_BOUNDARY,
  Q6K_WMMA_TAILS, admit_q6k_wmma)


VALID = dict(phase="prefill", quant="Q6_K", role="ffn_down", shape=Q6K_WMMA_FIXTURE_SHAPE,
             tails=Q6K_WMMA_TAILS, backend="AMD", arch="gfx1100", wave_size=32,
             lds_bytes=Q6K_WMMA_REQUIRED_LDS_BYTES, candidate_id=Q6K_WMMA_CANDIDATE_ID,
             evidence_id=Q6K_WMMA_EVIDENCE_ID, evidence_valid=True,
             live_byte_budget=Q6K_WMMA_LIVE_BYTES, enabled=True)


def decision(**changes):
  return admit_q6k_wmma(**(VALID | changes))


def test_q6_wmma_exact_structural_contract_admits():
  assert Q6K_WMMA_ROLE_SHAPE_BOUNDARY == {"ffn_down": Q6K_WMMA_FIXTURE_SHAPE}
  d = decision()
  assert d.admitted and d.candidate_id == Q6K_WMMA_CANDIDATE_ID
  assert d.required_live_bytes == d.live_byte_budget == Q6K_WMMA_LIVE_BYTES


def test_profile_and_path_are_optional_provenance_not_eligibility():
  assert decision(model_profile="unrelated-name", model_path="/models/anything.gguf").admitted
  assert decision(model_profile=None, model_path=None).admitted


def test_q6_wmma_gate_is_default_off(monkeypatch):
  monkeypatch.delenv("PREFILL_Q6K_WMMA", raising=False)
  d = decision(enabled=None)
  assert not d.admitted and d.provenance == "rollback" and d.rollback_route == "direct_packed"


def test_legacy_caller_without_structural_facts_fails_closed():
  d = admit_q6k_wmma(role="ffn_down", shape=Q6K_WMMA_FIXTURE_SHAPE, model_profile="old-profile", enabled=True)
  assert not d.admitted and d.route == "direct_packed"


@pytest.mark.parametrize(("field", "value"), [
  ("phase", "decode"), ("quant", "Q4_K"), ("role", "ffn_gate_up"),
  ("shape", {"M": 512, "N": 5120, "K": 17408}), ("tails", {"M": 0, "N": 0, "K": 64}),
  ("backend", "CUDA"), ("arch", "gfx1101"), ("wave_size", 64),
  ("lds_bytes", Q6K_WMMA_REQUIRED_LDS_BYTES - 1), ("candidate_id", "other"),
  ("evidence_id", "unmeasured"), ("evidence_valid", False),
])
def test_q6_wmma_fails_closed_outside_capability(field, value):
  d = decision(**{field: value})
  assert not d.admitted and d.route == d.rollback_route == "direct_packed"


@pytest.mark.parametrize("budget", [None, -1, 1.5, True, Q6K_WMMA_LIVE_BYTES - 1])
def test_q6_wmma_requires_byte_exact_live_budget(budget):
  assert not decision(live_byte_budget=budget).admitted


def test_real_non_fitting_selected_inventory_is_rejected_from_its_facts():
  # This is an actual loaded inventory row, not a model-name/profile assertion.
  inventory = dict(role="ffn_down", shape={"M": 512, "N": 5120, "K": 17408}, quant="Q6_K")
  d = decision(**inventory, live_byte_budget=24_000_000_000)
  assert not d.admitted
  assert "M/N/K are outside candidate capability" in d.errors
