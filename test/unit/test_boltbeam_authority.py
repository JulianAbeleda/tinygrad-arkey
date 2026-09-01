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

def test_bound_reduction_and_cache_authorities_exist():
  routes = load_promoted_routes()
  assert "q4_ffn_down_resadd" in routes["decode_ffn_down_resadd"]["components"]
  assert "finite_fp32_argmax" in routes["decode_native_argmax"]["components"]
  assert "qk_norm_rope_cache_sink" in routes["decode_producer_kv_cache_sink"]["components"]

def test_shared_q8_composite_authority():
  bundle = tickets_for_candidate({"family":"shared_q8_consumer.v1","quant":"q4"}, (
    ("decode_shared_q8_attention","shared_q8_q4_consumer"),
    ("decode_q4_direct_shared_q8_attention","shared_q8_q4_direct")))
  assert len(bundle.tickets) == 2
