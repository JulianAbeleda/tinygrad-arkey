import ast
from pathlib import Path
import pytest
from extra.llm_research.boltbeam_authority import (BOLTBEAM_SOURCE_LEDGER, DEFAULT_LEDGER, load_promoted_routes,
                                                   ticket_for_authority, tickets_for_candidate)

def test_authority_ledger_has_promoted_routes():
  routes = load_promoted_routes()
  assert len(routes) == 34
  assert routes["q6_ffn_down_streamk_destination.v1"]["state"] == "exact"

def test_bundled_authority_matches_boltbeam_source():
  if BOLTBEAM_SOURCE_LEDGER.exists(): assert DEFAULT_LEDGER.read_bytes() == BOLTBEAM_SOURCE_LEDGER.read_bytes()

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
  routes = load_promoted_routes()
  assert "shared_q8_q4q4_qkv_full" in routes["decode_shared_q8_q4q4_qkv_full"]["components"]

def test_attention_and_norm_authorities_exist():
  routes = load_promoted_routes()
  assert set(routes["custom_kernel_prefill_attention"]["components"]) == {"flash_prefill_score", "flash_prefill_combine"}
  assert "decode_rmsnorm" in routes["decode_rmsnorm_native_lowering"]["components"]

def test_generated_kernel_programs_carry_boltbeam_tickets():
  misses = []
  for path in (Path(__file__).parents[2] / "tinygrad" / "llm").glob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
      if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "KernelProgram": continue
      provenance = ast.unparse(node.args[2]) if len(node.args) > 2 else ""
      if "RESEARCH_ONLY" in provenance: continue
      if not any(keyword.arg == "boltbeam_ticket" for keyword in node.keywords): misses.append(f"{path.name}:{node.lineno}")
  assert misses == []
