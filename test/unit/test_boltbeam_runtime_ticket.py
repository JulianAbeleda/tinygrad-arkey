import pytest
from extra.llm_research.boltbeam_runtime_ticket import BoltbeamKernelTicket, require_promoted_ticket

def test_ticket_serializes_and_validates():
  t = BoltbeamKernelTicket("a" * 64, "b" * 64, "q8_provider", "sm120", "rev1")
  assert t.to_dict()["route_hash"] == "b" * 64
  with pytest.raises(ValueError): BoltbeamKernelTicket("bad", "b" * 64, "x", "y", "z")

def test_ticket_enforcement(monkeypatch):
  monkeypatch.setenv("TINYGRAD_BOLTBEAM_ENFORCE", "0")
  require_promoted_ticket(None, "r", "p")
  monkeypatch.setenv("TINYGRAD_BOLTBEAM_ENFORCE", "1")
  with pytest.raises(ValueError): require_promoted_ticket(None, "r", "p")
  require_promoted_ticket(BoltbeamKernelTicket("a" * 64, "b" * 64, "x", "y", "z"), "r", "p")

def test_ticket_enforcement_is_default_on(monkeypatch):
  monkeypatch.delenv("TINYGRAD_BOLTBEAM_ENFORCE", raising=False)
  with pytest.raises(ValueError): require_promoted_ticket(None, "r", "p")
