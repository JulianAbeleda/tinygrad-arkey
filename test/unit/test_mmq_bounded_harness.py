import pytest

import extra.qk.mmq_bounded_harness as harness
from extra.qk.mmq_bounded_harness import (
  ACTIVATION_LAYOUT_MMQ_DS4, ACTIVATION_LAYOUT_ROW_MAJOR, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID,
  AMD_DS4_COOP_TILE_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, CANDIDATE_ROUTE_ID, COMPARATOR_ID, K, M, N, ROLE,
  LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID, STAGED_DS4_BACKEND_ID, BoundedMMQConfig, MMQAtomUnavailableError,
  bounded_candidate_id, build_bounded_candidate_result, candidate_metadata,
  coop_tile_blocked_translation_evidence, run_bounded_harness,
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


def test_mmq_bounded_harness_amd_ds4_dot4x4_backend_metadata_only_is_not_default_route():
  meta = candidate_metadata(BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend=AMD_DS4_DOT4X4_BACKEND_ID))

  assert meta["backend"] == AMD_DS4_DOT4X4_BACKEND_ID
  assert meta["backend_atom_id"] == AMD_DS4_DOT4X4_BACKEND_ID
  assert meta["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert meta["candidate_route_id"] == CANDIDATE_ROUTE_ID
  assert meta["comparator_id"] == COMPARATOR_ID


def test_mmq_bounded_harness_amd_ds4_coop_tile_runs_bounded_correctness():
  cfg = BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, backend=AMD_DS4_COOP_TILE_BACKEND_ID,
                         measure_direct_packed=True, rounds=1)
  meta = candidate_metadata(cfg)
  evidence = coop_tile_blocked_translation_evidence(cfg)
  report = run_bounded_harness(cfg)

  assert meta["backend"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert meta["backend_atom_id"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert meta["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert meta["bounded_shape"] == {"M": 16, "N": 16, "K": 256}
  assert meta["comparator_id"] == COMPARATOR_ID
  assert evidence["status"] == "bounded_numeric_pass"
  assert evidence["bounded_only"] is True
  assert evidence["production_dispatch_changed"] is False
  assert evidence["default_route"] == "direct_packed"
  assert evidence["coop_tile_atom_source_hash"]
  assert {"M": 8, "N": 8, "K": 256} in evidence["attempted_shapes"]
  assert {"M": 16, "N": 16, "K": 256} in evidence["attempted_shapes"]
  assert {"M": 16, "N": 16, "K": 512} in evidence["attempted_shapes"]

  assert "store_owner metadata is not attached" in evidence["exact_blocker"]

  assert report["status"] == "PASS"
  assert report["metadata"]["backend"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert report["metadata"]["backend_atom_id"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert report["metadata"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert report["activation_layout_source"] == "amd_ds4_coop_tile_gpu_local_owner_carrier"
  assert report["correctness"]["tiles"] == 1
  assert report["correctness"]["max_abs"] <= report["correctness"]["atol"]
  assert report["artifacts"]["atom_source_hash"]
  assert report["artifacts"]["amd_ds4_coop_tile_atom_source_hash"]
  assert report["artifacts"]["emitted_binary_hash"]
  assert report["blockers"] == []


def test_mmq_bounded_harness_llama_coop_tile_oracle_runs_without_route_promotion():
  cfg = BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, backend=LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID,
                         measure_direct_packed=True)
  report = run_bounded_harness(cfg)

  assert report["status"] == "PASS"
  assert report["metadata"]["backend"] == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID
  assert report["metadata"]["backend_atom_id"] == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID
  assert report["metadata"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert report["activation_layout_source"] == "llama_mmq_coop_tile_oracle_carrier"
  assert report["correctness"]["max_abs"] == 0.0
  assert report["artifacts"]["llama_mmq_oracle_source_hash"]
  assert report["artifacts"]["llama_mmq_oracle_source_policy"]["vendored_cuda"] is False
  assert report["artifacts"]["llama_mmq_oracle_tiles"][0]["oracle_only"] is True
  assert report["artifacts"]["llama_mmq_oracle_tiles"][0]["writeback_owner_count"] == 1
  assert report["metadata"]["rollback"] == COMPARATOR_ID


def test_mmq_bounded_harness_amd_ds4_dot4x4_backend_runs_bounded_correctness(monkeypatch):
  seen = []

  def fake_ds4_dot4x4_runner(q4k_bytes, ds4):
    seen.append(ds4)
    m, k = ds4.spec.m, ds4.spec.k
    n = q4k_bytes.shape[0]
    spec = harness.describe_q4k_q8_1_mmq_tile(role=ROLE, m=m, n=n, k=k, m_tile=m, n_tile=n,
                                              k_groups=k // harness.Q8_1_BLOCK_ELEMS,
                                              activation_layout=harness.Q8_1_MMQ_DS4_LAYOUT)
    return harness.q4k_q8_1_mmq_ds4_tile_reference(q4k_bytes, ds4, spec)

  monkeypatch.setattr(harness, "_run_amd_ds4_dot4x4", fake_ds4_dot4x4_runner)

  report = run_bounded_harness(BoundedMMQConfig(m_tile=4, n_tile=5, k_groups=8, backend=AMD_DS4_DOT4X4_BACKEND_ID,
                                               rounds=1))

  assert len(seen) == 1
  assert report["status"] == "PASS"
  assert report["metadata"]["backend"] == AMD_DS4_DOT4X4_BACKEND_ID
  assert report["metadata"]["backend_atom_id"] == AMD_DS4_DOT4X4_BACKEND_ID
  assert report["metadata"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert report["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert report["activation_layout_source"] == "amd_ds4_dot4x4_gpu_direct_carrier"
  assert report["uses_precomputed_activation_sums"] is True
  assert report["q8_values_shape"] == [2, 4, 128]
  assert report["q8_scales_shape"] == [2, 4, 4]
  assert report["q8_sums_shape"] == [2, 4, 4]
  assert report["correctness"]["tiles"] == 1
  assert report["correctness"]["max_abs"] == 0.0
  assert report["artifacts"]["atom_source_hash"]
  assert report["artifacts"]["amd_ds4_dot4x4_atom_source_hash"]
  assert report["blockers"] == []


def test_mmq_bounded_harness_amd_ds4_warp_backend_metadata_only_is_not_default_route():
  meta = candidate_metadata(BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend=AMD_DS4_WARP_BACKEND_ID))

  assert meta["backend"] == AMD_DS4_WARP_BACKEND_ID
  assert meta["backend_atom_id"] == AMD_DS4_WARP_BACKEND_ID
  assert meta["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert meta["candidate_route_id"] == CANDIDATE_ROUTE_ID
  assert meta["comparator_id"] == COMPARATOR_ID


def test_mmq_bounded_harness_amd_ds4_lds_skeleton_backend_runs_bounded_correctness(monkeypatch):
  seen = []

  def fake_ds4_lds_runner(q4k_bytes, ds4):
    seen.append(ds4)
    m, k = ds4.spec.m, ds4.spec.k
    n = q4k_bytes.shape[0]
    spec = harness.describe_q4k_q8_1_mmq_tile(role=ROLE, m=m, n=n, k=k, m_tile=m, n_tile=n,
                                              k_groups=k // harness.Q8_1_BLOCK_ELEMS,
                                              activation_layout=harness.Q8_1_MMQ_DS4_LAYOUT)
    return harness.q4k_q8_1_mmq_ds4_tile_reference(q4k_bytes, ds4, spec)

  monkeypatch.setattr(harness, "_run_amd_ds4_lds_skeleton", fake_ds4_lds_runner)

  report = run_bounded_harness(BoundedMMQConfig(m_tile=4, n_tile=5, k_groups=8,
                                               backend=AMD_DS4_LDS_SKELETON_BACKEND_ID, rounds=1))

  assert len(seen) == 1
  assert report["status"] == "PASS"
  assert report["metadata"]["backend"] == AMD_DS4_LDS_SKELETON_BACKEND_ID
  assert report["metadata"]["backend_atom_id"] == AMD_DS4_LDS_SKELETON_BACKEND_ID
  assert report["metadata"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert report["activation_layout_source"] == "amd_ds4_lds_skeleton_gpu_local_carrier"
  assert report["correctness"]["max_abs"] == 0.0
  assert report["artifacts"]["amd_ds4_lds_skeleton_atom_source_hash"]
  assert report["timing"]["comparator_id"] == COMPARATOR_ID
  assert report["blockers"] == []


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


def test_mmq_bounded_harness_amd_ds4_dot4x4_requires_m_multiple_of_4():
  with pytest.raises(ValueError, match="multiple of 4"):
    BoundedMMQConfig(m_tile=2, m_tiles=1, n_tile=4, k_groups=8, backend=AMD_DS4_DOT4X4_BACKEND_ID).validate()


def test_mmq_bounded_candidate_result_oracle_only_exports_numeric_artifact():
  cfg = BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, backend=LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID)
  result = build_bounded_candidate_result(cfg)

  assert result["schema"] == "q4k-q8-1-mmq-bounded-candidate-result.v1"
  assert result["candidate_id"] == bounded_candidate_id(cfg)
  assert result["shape"] == {"M": 16, "N": 16, "K": 256}
  assert result["status"] == "oracle_only"
  assert result["numeric_status"] == "PASS"
  assert result["oracle_only"] is True
  assert result["production_dispatch_changed"] is False
  assert result["default_route"] == COMPARATOR_ID
  assert "no emitted AMD GPU candidate is claimed" in result["exact_blocker"]
  assert result["amd_isa_proof_bundle"]["schema"] == "tinygrad.amd_isa_proof_manifest.v1"
  assert result["amd_isa_proof_bundle"]["candidate_id"] == result["candidate_id"]
  assert result["amd_isa_proof_bundle"]["kernel_name"] == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID
  assert result["amd_isa_proof_bundle"]["rows"] == []
  assert result["harness_report"]["artifacts"]["llama_mmq_oracle_tiles"][0]["oracle_only"] is True


def test_mmq_bounded_candidate_result_marks_reference_backed_atom_as_missing_emitted_candidate():
  cfg = BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, backend="atom")
  result = build_bounded_candidate_result(cfg)

  assert result["candidate_id"] == bounded_candidate_id(cfg)
  assert result["shape"] == {"M": 4, "N": 4, "K": 256}
  assert result["status"] == "blocked_emitted_candidate_missing"
  assert result["numeric_status"] == "PASS"
  assert result["oracle_only"] is False
  assert result["production_dispatch_changed"] is False
  assert result["harness_report"]["artifacts"]["emitted_binary_hash"] is None
  assert result["exact_blocker"] == "atom backend is reference-backed; AMD GPU atom body is not implemented"


def test_mmq_bounded_candidate_result_records_coop_numeric_pass():
  cfg = BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, backend=AMD_DS4_COOP_TILE_BACKEND_ID)
  result = build_bounded_candidate_result(cfg)

  assert result["status"] == "PASS"
  assert result["numeric_status"] == "PASS"
  assert result["shape"] == {"M": 16, "N": 16, "K": 256}
  assert result["harness_report"]["status"] == "PASS"
  assert result["harness_report"]["artifacts"]["amd_ds4_coop_tile_atom_source_hash"]
  assert result["harness_report"]["artifacts"]["emitted_binary_hash"]
  assert result["exact_blocker"] is None
