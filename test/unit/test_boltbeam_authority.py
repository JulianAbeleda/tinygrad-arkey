import pytest
from extra.llm_research.boltbeam_authority import load_promoted_routes, ticket_for_authority

def test_authority_ledger_has_promoted_routes():
  routes = load_promoted_routes()
  assert len(routes) == 26
  assert routes["q6_ffn_down_streamk_destination.v1"]["state"] == "exact"

def test_ticket_for_authority_is_fail_closed():
  ticket = ticket_for_authority("decode_shared_q8_attention", "shared_q8_provider", "a" * 64, "sm120")
  assert len(ticket.route_hash) == 64
  with pytest.raises(ValueError): ticket_for_authority("missing", "x", "a" * 64, "sm120")
