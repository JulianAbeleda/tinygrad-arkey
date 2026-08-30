from extra.llm_research.prefill.nv_llama_packed_q6k_down_pp512_binding import K,M,N,supports

def test_exact_q6_down_admission_is_fail_closed():
  base=dict(model_family="qwen3_8b",role="ffn_down",weight_type="Q6_K",m=M,n=N,k=K,device="NV")
  assert supports(**base)
  for mutation in ({"role":"attn_v"},{"weight_type":"Q4_K"},{"m":256},{"n":12288},{"k":4096},{"device":"AMD"}):assert not supports(**(base|mutation))
