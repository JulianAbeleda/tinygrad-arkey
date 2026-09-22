from extra.llm_research.prefill.nv_compiler_q4k_down_pp512_binding import *

def test_down_admission_is_exact_and_type12_only():
  kw = dict(model_family="qwen3_8b", role="ffn_down", weight_type="Q4_K",
            m=512, n=4096, k=12288, device="NV")
  assert supports(**kw)
  assert not supports(**kw, ggml_type=14)
  assert not supports(**{**kw, "role":"attn_k"})
  assert not supports(**{**kw, "n":12288, "k":4096})

def test_down_census_is_18():
  capture_for().prepare_records(18)
