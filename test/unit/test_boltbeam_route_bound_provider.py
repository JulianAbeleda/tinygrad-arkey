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
  _candidate("q4_ffn_down.v1", {"block_count":12,"resadd":True,"load_style":"vector"},
    "decode_q4k_ffn_down_fp16_geometry", "q4_ffn_down_fp16"),
  _candidate("q4_w1w3.v1", {"rows":12288,"k":4096,"load_style":"vector","store_fp16":False},
    "decode_q4k_w1w3_fusion", "q4_w1w3_fused"),
  _candidate("finite_argmax.v1", {"n":151936,"threads":1024,"host_mirror":False},
    "decode_native_argmax", "finite_fp32_argmax"),
  _candidate("kv_rope_store.v1", {"Hkv":8,"Hd":128,"max_context":4096,"vparts":1},
    "decode_kv_store_fusion", "kv_rope_store"),
  _candidate("q4_k_four_warp.v1", {"rows":1024,"k":4096}, "decode_q4k_k_four_warp", "q4_k_four_warp"),
  _candidate("shared_q8_provider.v1", {"k":4096,"source_dtype":"fp16"},
    "decode_shared_q8_attention", "shared_q8_provider"),
  _candidate("decode_rmsnorm.v1", {"rows":1,"dim":4096,"eps":1e-6,"warps_per_row":16,
    "x_dtype":"dtypes.float","weight_dtype":"dtypes.half","out_dtype":"dtypes.float","x_rank":1},
    "decode_rmsnorm_native_lowering", "decode_rmsnorm"),
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

def test_symbolic_lowering_binding_is_outside_candidate_identity():
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  candidate={"family":"shared_q8_attention_consumer.v1","rows":1024,"variant":"q4_cooperative",
             "direct_output":True,"block_count_binding":"cooperative_blocks"}
  authorities=(("decode_q4_direct_shared_q8_attention","shared_q8_q4_direct"),)
  first,ticket1=lower_authorized_candidate(candidate,authorities,lowering_bindings={"cooperative_blocks":UOp.const(dtypes.weakint,4)})
  second,ticket2=lower_authorized_candidate(candidate,authorities,lowering_bindings={"cooperative_blocks":UOp.const(dtypes.weakint,8)})
  assert callable(first) and callable(second) and ticket1.tickets[0].candidate_hash == ticket2.tickets[0].candidate_hash

def test_multi_output_symbolic_provider():
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  emitter,ticket=lower_authorized_candidate({"family":"shared_q8_multi_output.v1","variant":"q4kv_pair",
    "rows":1024,"block_count_binding":"cooperative_blocks"},
    (("decode_shared_q8_q4kv_pair","shared_q8_q4_kv_pair"),),
    lowering_bindings={"cooperative_blocks":UOp.const(dtypes.weakint,4)})
  assert callable(emitter) and ticket.tickets[0].component == "shared_q8_q4_kv_pair"
