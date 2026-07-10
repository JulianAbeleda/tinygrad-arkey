import pytest

from extra.qk.mmq_bounded_harness import (
  ACTIVATION_LAYOUT_MMQ_DS4, ACTIVATION_LAYOUT_ROW_MAJOR, CANDIDATE_ROUTE_ID, COMPARATOR_ID, K, M, N, ROLE,
  STAGED_DS4_BACKEND_ID, BoundedMMQConfig, candidate_metadata, run_bounded_harness,
)


def test_mmq_bounded_harness_metadata_names_required_14b_candidate_surface():
  cfg = BoundedMMQConfig(m_tile=8, n_tile=8, k_groups=8)
  meta = candidate_metadata(cfg)

  assert meta["role"] == ROLE == "ffn_gate_up"
  assert (meta["M"], meta["N"], meta["K"]) == (M, N, K) == (512, 17408, 5120)
  assert meta["quant"] == "Q4_K"
  assert meta["activation"] == "Q8_1"
  assert meta["candidate_route_id"] == CANDIDATE_ROUTE_ID == "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
  assert meta["comparator_id"] == COMPARATOR_ID == "direct_packed"
  assert meta["rollback"] == "direct_packed"
  assert meta["primitive_class"] == "compiler_primitive_spec_owned__hand_mmq_backend_atom"
  assert meta["activation_layout"] == ACTIVATION_LAYOUT_ROW_MAJOR


def test_mmq_bounded_harness_reference_only_runs_without_atom_or_gpu_route_binding():
  report = run_bounded_harness(BoundedMMQConfig(m_tile=4, n_tile=5, k_groups=8, rounds=1, backend="reference"))

  assert report["schema"] == "q4k-q8-1-mmq-bounded-harness.v1"
  assert report["status"] == "PASS"
  assert report["metadata"]["backend"] == "reference"
  assert report["metadata"]["activation_layout"] == "row_major_q8_1"
  assert report["metadata"]["candidate_route_id"] == CANDIDATE_ROUTE_ID
  assert report["activation_layout"] == "row_major_q8_1"
  assert report["activation_layout_source"] == "current_row_major_q8_1_reference_pack"
  assert report["q8_values_shape"] == [4, 256]
  assert report["q8_scales_shape"] == [4, 8]
  assert report["q8_sums_shape"] is None
  assert report["llama_mmq_geometry"] == {"mmq_x": 128, "mmq_y": 128, "iter_k": 256, "nwarps": 8}
  assert report["uses_precomputed_activation_sums"] is False
  assert report["timing"]["comparator_id"] == "direct_packed"
  assert report["timing"]["comparator_status"] == "named_not_measured"
  assert report["artifacts"]["emitted_binary_hash"] is None


def test_mmq_bounded_harness_reference_runs_mmq_ds4_activation_layout():
  report = run_bounded_harness(BoundedMMQConfig(m_tile=4, n_tile=5, k_groups=8, rounds=1, backend="reference",
                                               activation_layout=ACTIVATION_LAYOUT_MMQ_DS4))

  assert report["status"] == "PASS"
  assert report["metadata"]["activation_layout"] == "mmq_ds4"
  assert report["activation_layout"] == "mmq_ds4"
  assert report["activation_layout_source"] == "l0_l1_q8_1_mmq_ds4_reference_pack"
  assert report["q8_values_shape"] == [2, 4, 128]
  assert report["q8_scales_shape"] == [2, 4, 4]
  assert report["q8_sums_shape"] == [2, 4, 4]
  assert report["uses_precomputed_activation_sums"] is True
  assert report["timing"]["comparator_id"] == "direct_packed"
  assert report["timing"]["comparator_status"] == "named_not_measured"


def test_mmq_bounded_harness_multi_tile_reference_surface_is_bounded():
  report = run_bounded_harness(BoundedMMQConfig(m_tile=4, n_tile=5, k_groups=8, m_tiles=2, n_tiles=3, rounds=1))

  assert report["status"] == "PASS"
  assert report["correctness"]["tiles"] == 6
  assert report["metadata"]["bounded_shape"] == {"M": 8, "N": 15, "K": 256}


def test_mmq_bounded_harness_atom_backend_runs_bounded_correctness():
  report = run_bounded_harness(BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, m_tiles=2, n_tiles=2, backend="atom"))

  assert report["status"] == "PASS"
  assert report["metadata"]["backend"] == "atom"
  assert report["correctness"]["max_abs"] == 0.0
  assert report["correctness"]["tiles"] == 4
  assert report["artifacts"]["atom_source_hash"]
  assert report["blockers"] == ["atom backend is reference-backed; AMD GPU atom body is not implemented"]


def test_mmq_bounded_harness_amd_backend_is_selectable_metadata_only():
  cfg = BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend="amd")
  meta = candidate_metadata(cfg)

  assert meta["backend"] == "amd"
  assert meta["candidate_route_id"] == CANDIDATE_ROUTE_ID


def test_mmq_bounded_harness_amd_warp_backend_is_selectable_metadata_only():
  cfg = BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend="amd_warp")
  meta = candidate_metadata(cfg)

  assert meta["backend"] == "amd_warp"
  assert meta["candidate_route_id"] == CANDIDATE_ROUTE_ID


def test_mmq_bounded_harness_staged_ds4_backend_reports_fail_loud_probe_contract():
  cfg = BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend=STAGED_DS4_BACKEND_ID, rounds=1)
  report = run_bounded_harness(cfg)

  assert report["status"] == "PASS"
  assert report["metadata"]["backend"] == STAGED_DS4_BACKEND_ID
  assert report["metadata"]["backend_atom_id"] == STAGED_DS4_BACKEND_ID
  assert report["metadata"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert report["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert report["uses_precomputed_activation_sums"] is True
  assert report["q8_values_shape"] == [2, 4, 128]
  assert report["q8_scales_shape"] == [2, 4, 4]
  assert report["q8_sums_shape"] == [2, 4, 4]
  assert report["llama_mmq_geometry"] == {"mmq_x": 128, "mmq_y": 128, "iter_k": 256, "nwarps": 8}
  assert report["artifacts"]["atom_source_hash"]
  assert report["artifacts"]["staged_ds4_atom_source_hash"]
  assert report["artifacts"]["emitted_binary_hash"] is None
  assert "no production dispatch or route promotion is claimed" in report["blockers"]


def test_mmq_bounded_harness_staged_ds4_backend_metadata_only_is_not_default_route():
  meta = candidate_metadata(BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend=STAGED_DS4_BACKEND_ID))

  assert meta["backend"] == STAGED_DS4_BACKEND_ID
  assert meta["backend_atom_id"] == STAGED_DS4_BACKEND_ID
  assert meta["candidate_route_id"] == CANDIDATE_ROUTE_ID
  assert meta["comparator_id"] == COMPARATOR_ID


@pytest.mark.parametrize("backend", ["direct_packed", "amd_warp_batched", "amd_dot4_batched", "amd_dot4x4_batched"])
def test_mmq_bounded_harness_comparator_and_batched_backends_are_selectable_metadata_only(backend):
  cfg = BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend=backend, measure_direct_packed=True)
  meta = candidate_metadata(cfg)

  assert meta["backend"] == backend
  assert meta["comparator_id"] == "direct_packed"
  assert meta["candidate_route_id"] == CANDIDATE_ROUTE_ID


def test_mmq_bounded_harness_rejects_unbounded_shape():
  with pytest.raises(ValueError, match="exceeds role shape"):
    BoundedMMQConfig(m_tile=M + 1).validate()


def test_mmq_bounded_harness_rejects_unknown_activation_layout():
  cfg = BoundedMMQConfig(activation_layout="blocked")

  with pytest.raises(ValueError, match="unknown activation_layout"):
    cfg.validate()
