import pytest
from extra.llm_research.boltbeam_kernel_provider import candidate_hash, generate_route_bound_candidate
from extra.llm_research.boltbeam_authority import lower_authorized_candidate

def _candidate(family, parameters, route, component):
  return {"schema":"boltbeam.route_bound_candidate.v1","target":"nvidia_sm120","family":family,"parameters":parameters,
          "authorities":[{"route_id":route,"component":component}]}

@pytest.mark.parametrize("candidate", [
  _candidate("q4_gate_up.v1", {"vector_loads":False}, "decode_q4k_gate_up_four_warp_vector", "q4_gate_up_four_warp"),
  _candidate("q4_kv_pair.v1", {"rows":1024,"k":4096}, "decode_q4k_kv_pair", "q4_kv_pair"),
  _candidate("q6_v.v1", {}, "decode_q6k_v_four_warp_fp16_geometry", "q6_v_four_warp_fp16"),
  _candidate("q6_ffn_down.v1", {"rows_per_block":1,"packed_lanemap":True,"unroll_blocks":None,
    "split_weight_stream":False}, "decode_q6k_ffn_down_fp16_geometry", "q6_ffn_down_fp16"),
])
def test_route_bound_packed_family_generates_registered_emitter(candidate):
  generated = generate_route_bound_candidate(candidate, candidate_hash(candidate))
  assert generated.kind == "uop" and callable(generated.artifact)

def test_route_bound_provider_rejects_identity_drift():
  candidate = _candidate("q6_v.v1", {}, "decode_q6k_v_four_warp_fp16_geometry", "q6_v_four_warp_fp16")
  with pytest.raises(ValueError): generate_route_bound_candidate(candidate, "0" * 64)

def test_lowering_returns_emitter_and_matching_ticket():
  emitter,ticket = lower_authorized_candidate({"family":"q6_v.v1"},
    (("decode_q6k_v_four_warp_fp16_geometry","q6_v_four_warp_fp16"),))
  assert callable(emitter) and ticket.tickets[0].component == "q6_v_four_warp_fp16"
