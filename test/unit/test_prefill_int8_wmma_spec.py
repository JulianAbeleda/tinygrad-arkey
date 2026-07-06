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


def test_q4k_int8_wmma_tiled_prefill_spec_covers_14b_roles():
  from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill
  for role, n, k in (("attn_qo", 5120, 5120), ("attn_kv", 1024, 5120),
                     ("ffn_gate_up", 17408, 5120), ("ffn_down", 5120, 17408)):
    spec = describe_q4k_int8_wmma_tiled_prefill(n, k, 512, role=role, m_tile=16, n_tile=16, group_tile=1)
    assert spec.implementation == "direct_tiled_wmma_v0"
    assert spec.kernel_name.startswith(f"prefill_q4k_q8_1_wmma_tiled_generated_gemm_{role}_")
    assert spec.live_raw_elems == 16 * 16
    assert spec.forbidden_full_raw_elems == spec.groups * spec.m * spec.n


def test_q4k_wmma_full_role_lowering_spec_counts_14b_work_without_full_raw():
  from extra.qk.q4k_wmma_tile_lowering import describe_q4k_wmma_full_role_lowering
  spec = describe_q4k_wmma_full_role_lowering(5120, 5120, 512, role="attn_qo",
                                             m_tile=16, n_tile=16, group_tile=1)
  assert spec.grid_m == 32
  assert spec.grid_n == 320
  assert spec.output_tiles == 10240
  assert spec.raw_tile_steps == 10240 * 160
  assert spec.wmma_fragments_per_raw_tile == 2
  assert spec.live_raw_elems == 256
  assert spec.forbidden_full_raw_elems == 160 * 512 * 5120
  assert spec.requires_scheduler_owned_loop
  assert spec.to_json()["lowering"]["wmma_surface"] == "tc_matcher_tile"
