import pytest


def test_q4k_int8_wmma_prefill_spec_names_generated_route():
  from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_prefill
  spec = describe_q4k_int8_wmma_prefill(64, 256, 16, role="ffn_gate_up")
  assert spec.kernel_name == "prefill_q4k_q8_1_wmma_generated_gemm_ffn_gate_up_64_256_16"
  assert spec.groups == 8
  assert spec.k_blocks == 1
  assert spec.to_json()["implementation"] == "group_tensor_matmul_v0"


def test_q4k_int8_wmma_prefill_spec_rejects_unaligned_shapes():
  from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_prefill
  with pytest.raises(ValueError, match="Q4_K block elems"):
    describe_q4k_int8_wmma_prefill(64, 128, 16)
  with pytest.raises(ValueError, match="WMMA tile"):
    describe_q4k_int8_wmma_prefill(63, 256, 16)
