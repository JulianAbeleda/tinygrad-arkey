import pytest

from tinygrad.llm.boltbeam_authority import build_registered_llm_emitter



def test_existing_q4_q6_q8_families_are_closed_registered_adapters():
  families = [
    ("q8_1_provider.v1", {"k": 4096, "source_dtype": "fp16"}),
    ("q4_g3_gemv.v1", {"rows": 4096, "k": 4096, "lanes": 32, "load_style": "vector", "epilogue": ""}),
    ("q4_w1w3.v1", {"rows": 12288, "k": 4096, "load_style": "vector", "store_fp16": True}),
    ("q4_gate_up.v1", {"vector_loads": True}),
    ("q4_ffn_down.v1", {"block_count": 3, "resadd": True, "load_style": "vector"}),
    ("q4_kv_pair.v1", {"rows": 1024, "k": 4096}),
    ("q6_ffn_down.v1", {"rows_per_block": 1, "packed_lanemap": True, "unroll_blocks": 4, "split_weight_stream": False}),
    ("q6_v.v1", {}),
    ("shared_q8_consumer.v1", {"rows": 4096, "block_count": 16, "direct_output": True, "residual_add": False, "quant": "q4"}),
  ]
  for family, params in families:
    assert callable(build_registered_llm_emitter(family, params)), family
  with pytest.raises(ValueError, match="no registered"):
    build_registered_llm_emitter("unknown.v1", {})
  with pytest.raises(ValueError, match="parameters are invalid"):
    build_registered_llm_emitter("q4_ffn_down.v1", {"block_count":12,"resadd":True,"load_style":"vector"})
