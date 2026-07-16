import numpy as np

from extra.qk.q4k_q8_fixture import ACTIVATION_LAYOUT_MMQ_DS4, make_finite_q4k_bytes, make_q8_activation_inputs
from extra.qk.mmq_llama_oracle import (
  LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID, LlamaMMQOracleGeometry, llama_mma_writeback_coverage,
  llama_mma_writeback_owners, llama_mma_sum_slot_mapping, llama_mmq_source_policy, run_llama_mmq_coop_tile_oracle,
)
from extra.qk.mmq_q4k_q8_reference import (
  Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_ds4_tile_reference,
)


def test_llama_mmq_oracle_writeback_owners_follow_8_wave_16x16_stripes():
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=128, n=128, k=256, m_tile=128, n_tile=128,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  owners = llama_mma_writeback_owners(spec)

  assert len(owners) == 64
  assert owners[0]["warp_id"] == 0
  assert owners[0]["m_range"] == [0, 16]
  assert owners[0]["n_range"] == [0, 16]
  assert owners[-1]["warp_id"] == 7
  assert owners[-1]["m_range"] == [112, 128]
  assert owners[-1]["n_range"] == [112, 128]


def test_llama_mmq_writeback_coverage_has_no_missing_or_duplicate_stores():
  cases = [
    (16, 16, 1),
    (32, 16, 2),
    (32, 32, 4),
    (128, 128, 64),
  ]
  for m, n, expected_fragments in cases:
    spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=256, m_tile=m, n_tile=n,
                                      activation_layout=Q8_1_MMQ_DS4_LAYOUT)
    coverage = llama_mma_writeback_coverage(spec)

    assert coverage["owner_fragment_count"] == expected_fragments
    assert coverage["covered_output_count"] == m * n
    assert coverage["expected_output_count"] == m * n
    assert coverage["duplicate_store_count"] == 0
    assert coverage["missing_store_count"] == 0


def test_llama_mmq_sum_slot_mapping_probe_covers_16x16_and_32x32_tiles_without_dispatch_claim():
  for m, n in [(16, 16), (32, 32)]:
    spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=256, m_tile=m, n_tile=n,
                                      activation_layout=Q8_1_MMQ_DS4_LAYOUT)
    probe = llama_mma_sum_slot_mapping(spec)

    assert probe["schema"] == "llama-mmq-asm-sum-slot-mapping-probe.v1"
    assert probe["candidate_id"] == "llama_mmq_r4_sum_slot_mapping_probe"
    assert probe["probe_kind"] == "sum_slot_accumulator_mapping"
    assert probe["status"] == "static_mapping_pass"
    assert probe["research_only"] is True
    assert probe["production_dispatch_changed"] is False
    assert probe["default_route"] == "direct_packed"
    assert probe["tile_c_thread_elems"] == 8
    assert probe["slots_per_thread"] == 8 * (n // 16)
    assert probe["mapped_output_count"] == m * n
    assert probe["expected_output_count"] == m * n
    assert probe["duplicate_store_count"] == 0
    assert probe["missing_store_count"] == 0
    assert len(probe["slots"]) == m * n
    assert probe["tinygrad_asm_surface"]["representable_static_identity"] is True
    assert probe["tinygrad_asm_surface"]["runtime_kernel_probe_status"] == "blocked_missing_physical_slot_introspection"
    assert "physical VGPR" in probe["tinygrad_asm_surface"]["exact_missing_primitive_or_api"]


def test_llama_mmq_sum_slot_mapping_probe_matches_writeback_owner_fragments():
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=32, n=32, k=256, m_tile=32, n_tile=32,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  probe = llama_mma_sum_slot_mapping(spec)
  coverage = llama_mma_writeback_coverage(spec)

  probe_points = {(row["m"], row["n"]) for row in probe["slots"]}
  owner_points = {
    (m, n)
    for owner in coverage["owners"]
    for m in range(owner["m_range"][0], owner["m_range"][1])
    for n in range(owner["n_range"][0], owner["n_range"][1])
  }
  assert probe_points == owner_points
  assert [row["sum_slot"] for row in probe["slots"][:4]] == [0, 1, 2, 3]
  assert [row["thread"]["lane_id"] for row in probe["slots"][:4]] == [0, 0, 0, 0]
  assert probe["slots"][8]["sum_slot"] == 0
  assert probe["slots"][8]["thread"]["lane_id"] == 1


def test_llama_mmq_oracle_matches_ds4_reference_for_bounded_tile():
  m, n, k = 16, 16, 256
  q4k = make_finite_q4k_bytes(n, k, seed=20260721)
  activation = make_q8_activation_inputs(m, k, seed=20260722, activation_layout=ACTIVATION_LAYOUT_MMQ_DS4)
  assert activation.ds4_activation is not None
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)

  result = run_llama_mmq_coop_tile_oracle(q4k, activation.ds4_activation, spec)
  ref = q4k_q8_1_mmq_ds4_tile_reference(q4k, activation.ds4_activation, spec)

  assert result.backend_id == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID
  assert result.source_policy["vendored_cuda"] is False
  assert result.to_json()["oracle_only"] is True
  np.testing.assert_allclose(result.output, ref, rtol=0, atol=0)


def test_llama_mmq_oracle_rejects_invalid_geometry():
  geom = LlamaMMQOracleGeometry(mmq_y=64, nwarps=8, tile_c_i=16)

  try:
    geom.validate()
  except ValueError as exc:
    assert "nwarps*tile_c_i" in str(exc)
  else:
    raise AssertionError("invalid geometry should fail")


def test_llama_mmq_source_policy_points_to_clone_not_vendored_cuda():
  policy = llama_mmq_source_policy()

  assert policy["source_clone"].endswith("ggml/src/ggml-cuda/mmq.cuh")
  assert policy["vendored_cuda"] is False
  assert "mul_mat_q_process_tile" in policy["anchors"]
