import pytest
from extra.llm_research.boltbeam_authority import load_promoted_routes, ticket_for_authority, tickets_for_candidate

def test_authority_ledger_has_promoted_routes():
  routes = load_promoted_routes()
  assert len(routes) == 31
  assert routes["q6_ffn_down_streamk_destination.v1"]["state"] == "exact"

def test_ticket_for_authority_is_fail_closed():
  ticket = ticket_for_authority("decode_shared_q8_attention", "shared_q8_provider", "a" * 64, "sm120")
  assert len(ticket.route_hash) == 64
  with pytest.raises(ValueError): ticket_for_authority("missing", "x", "a" * 64, "sm120")

def test_composite_candidate_authority():
  bundle = tickets_for_candidate({"family": "q6_ffn_down.v1", "rows_per_block": 1}, (
    ("decode_q6k_ffn_down_fp16_geometry", "q6_ffn_down_fp16"),
    ("decode_q6k_ffn_down_packed_lanemap", "q6_ffn_down_packed_lanemap")))
  assert len(bundle.tickets) == 2
  assert len({ticket.candidate_hash for ticket in bundle.tickets}) == 1
