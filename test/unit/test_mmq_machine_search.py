from extra.qk.mmq_bounded_harness import (
  ACTIVATION_LAYOUT_MMQ_DS4, AMD_DS4_COOP_TILE_BACKEND_ID, AMD_DS4_DOT4X4_BACKEND_ID,
  AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID, FULL_GRID_BACKEND_ID,
  LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID,
  BoundedMMQConfig,
)
from extra.qk.mmq_machine_search import (
  build_r4_evidence_artifacts, build_r5_geometry_search_report, build_r6_route_gate_status,
  build_r7_reduction_status, build_search_report, build_full_gpu_probe_candidate, evaluate_candidate_promotion,
  R5_GEOMETRY_CANDIDATES, build_r6_role_shape_integration_artifact, build_full_grid_k_tiled_dispatch_plan,
)
from extra.qk.mmq_machine_search import build_boltbeam_oracle_trace


def test_mmq_promotion_adapter_fails_closed_on_missing_or_invalid_evidence():
  missing = evaluate_candidate_promotion()
  assert missing["verdict"] == "BLOCKED_FAIL_CLOSED"
  assert missing["promotion_eligible"] is False
  assert "missing owner coverage evidence" in missing["blockers"]
  invalid = evaluate_candidate_promotion(owner_coverage={"schema": "wrong"})
  assert invalid["verdict"] == "BLOCKED_FAIL_CLOSED"
  assert any("invalid owner coverage evidence" in blocker for blocker in invalid["blockers"])


def test_mmq_promotion_adapter_rejects_forged_all_true_cooperative_evidence():
  """PASS must not be reachable from summary booleans without provenance."""
  owner = build_r4_evidence_artifacts()["owner_coverage"]
  forged = {
    "status": "PASS", "bounded_only": True,
    "production_dispatch_changed": False, "default_route": "direct_packed",
  }
  result = evaluate_candidate_promotion(
    owner_coverage=owner, cooperative_tile=forged,
    q4_q8_staging=True, resource_scratch=True, distinct_binary=True,
    correctness=True, same_session_timing=True, no_fallback=True,
  )
  assert result["promotion_eligible"] is False
  assert result["verdict"] == "BLOCKED_FAIL_CLOSED"


def test_mmq_machine_search_only_marks_completed_components_searchable():
  report = build_search_report()

  assert report["schema"] == "q4k-q8-1-mmq-machine-search.v1"
  assert report["production_dispatch_changed"] is False
  assert report["default_route"] == "direct_packed"
  assert report["llama_kernel_source_policy"]["mode"] == "point_to_local_clone_do_not_vendor"
  assert report["llama_kernel_source_policy"]["handcoded_translation"] is True
  assert report["llama_kernel_source_policy"]["reduction_model"] == "unconverted_parts_point_to_clone_converted_parts_become_bounded_atoms"
  assert "minimized hand-coded tinygrad translation" in report["llama_kernel_source_policy"]["atom_definition"]
  source_paths = {row["path"] for row in report["llama_kernel_source_policy"]["sources"]}
  assert "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh" in source_paths
  assert "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh" in source_paths
  assert report["promotion_verdict"] == "BLOCKED_UNTIL_COOPERATIVE_TILE_WIN"
  assert report["r5_geometry_search_status"] == {
    "status": "ready_for_bounded_geometry_search",
    "reason": "R4 lowered owner trace, staging evidence, and R5 bounded coop numeric correctness pass; R6 remains blocked until R5 reports a same-session coop speed win",
    "required_r4_evidence": ["owner_coverage:PASS", "staging_sum_slots:PASS", "gpu_owner_trace:PASS"],
  }
  assert report["r5_geometry_search"]["promotion_verdict"] == "NO_PROMOTION_WITHOUT_BOUNDED_COOP_WIN"
  assert report["r6_route_gate_status"]["status"] == "BLOCKED_NO_BOUNDED_COOP_WIN"
  assert report["r7_reduction_status"]["status"] == "BLOCKED_REMAINING_SOURCE_CLONE_ROWS"
  assert report["searchable_components"] == [
    "DS4 layout",
    "DS4 reference correctness",
    "Q4_K x DS4 formula",
    "Q4_K tile loader",
    "sudot4 primitive availability",
    "direct DS4 GPU atom",
    "R3 LDS skeleton atom",
    "R4 cooperative multi-wave output ownership",
    "R4 llama cooperative tile oracle",
    "R5 bounded cooperative numeric atom",
  ]
  blocked_components = [row for row in report["done_components"] if row["status"] == "blocked_translation"]
  assert blocked_components == []
  r4_component = next(row for row in report["done_components"] if row["component"] == "R4 cooperative multi-wave output ownership")
  assert r4_component["implementation"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert "lowered AMD ISA proof manifest" in r4_component["evidence"]
  assert {row["component"] for row in report["done_components"] if row["status"] == "done"} == set(report["searchable_components"])

  rows = {row["candidate_id"]: row for row in report["searchable_candidates"]}
  assert set(rows) == {
    "direct_packed_comparator",
    "ds4_reference_formula",
    "amd_ds4_warp_direct",
    "staged_ds4_reference_probe",
    "amd_ds4_dot4x4_packed",
    "amd_ds4_lds_skeleton",
    "llama_mmq_coop_tile_oracle",
    "amd_ds4_coop_tile_bounded",
  }
  assert rows["amd_ds4_warp_direct"]["backend"] == AMD_DS4_WARP_BACKEND_ID
  assert rows["amd_ds4_warp_direct"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert rows["amd_ds4_warp_direct"]["promotion_eligible"] is False
  assert rows["amd_ds4_dot4x4_packed"]["backend"] == AMD_DS4_DOT4X4_BACKEND_ID
  assert rows["amd_ds4_dot4x4_packed"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert rows["amd_ds4_dot4x4_packed"]["status"] == "searchable"
  assert rows["amd_ds4_dot4x4_packed"]["promotion_eligible"] is False
  assert rows["amd_ds4_dot4x4_packed"]["bounded_config"] == {
    "m_tile": 4,
    "n_tile": 5,
    "k_groups": 8,
    "m_tiles": 1,
    "n_tiles": 1,
  }
  assert rows["amd_ds4_lds_skeleton"]["backend"] == AMD_DS4_LDS_SKELETON_BACKEND_ID
  assert rows["amd_ds4_lds_skeleton"]["activation_layout"] == ACTIVATION_LAYOUT_MMQ_DS4
  assert rows["amd_ds4_lds_skeleton"]["status"] == "evidence_only"
  assert rows["amd_ds4_lds_skeleton"]["promotion_eligible"] is False
  assert rows["amd_ds4_lds_skeleton"]["metadata"]["backend_atom_id"] == AMD_DS4_LDS_SKELETON_BACKEND_ID
  assert rows["llama_mmq_coop_tile_oracle"]["backend"] == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID
  assert rows["llama_mmq_coop_tile_oracle"]["status"] == "oracle_only"
  assert rows["llama_mmq_coop_tile_oracle"]["promotion_eligible"] is False
  assert rows["amd_ds4_coop_tile_bounded"]["backend"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert rows["amd_ds4_coop_tile_bounded"]["status"] == "searchable"
  assert rows["amd_ds4_coop_tile_bounded"]["promotion_eligible"] is False

  blocked = {row["candidate_id"]: row for row in report["blocked_candidates"]}
  assert "amd_ds4_dot4x4_packed" not in blocked
  assert "cooperative_multi_wave_tile" not in blocked
  assert blocked["full_14b_prefill_route"]["status"] == "blocked"

  r4 = report["r4_evidence_artifacts"]
  assert r4["owner_coverage"]["schema"] == "tinygrad.mmq_owner_coverage.v1"
  assert r4["owner_coverage"]["candidate_id"] == "cooperative_multi_wave_tile"
  assert r4["owner_coverage"]["backend"] == "lowered_amd_isa_fragmented_store_owner_manifest"
  assert r4["owner_coverage"]["status"] == "PASS"
  assert r4["owner_coverage"]["observed_stores"]["stores"][0]["owner"]["gpu_execution_trace"] is False
  assert r4["owner_coverage"]["production_dispatch_changed"] is False
  assert r4["gpu_owner_trace"]["status"] == "PASS"
  assert r4["gpu_owner_trace"]["store_rows"] == 256
  assert r4["gpu_owner_trace"]["unique_store_owners"] == 256
  assert r4["gpu_owner_trace"]["fragment_count"] == 8
  assert r4["gpu_owner_trace"]["gated_store_rows"] == 256
  assert r4["staging_sum_slots"]["schema"] == "tinygrad.mmq_staging_evidence.v1"
  assert r4["staging_sum_slots"]["candidate_id"] == "cooperative_multi_wave_tile"
  assert r4["staging_sum_slots"]["backend"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert r4["staging_sum_slots"]["status"] == "PASS"
  assert r4["staging_sum_slots"]["production_dispatch_changed"] is False


def test_mmq_r4_evidence_artifacts_are_transfer_shaped_and_non_promoting():
  artifacts = build_r4_evidence_artifacts()

  assert set(artifacts) == {"owner_coverage", "gpu_owner_trace", "staging_sum_slots"}
  for artifact in (artifacts["owner_coverage"], artifacts["staging_sum_slots"]):
    assert artifact["candidate_id"] == "cooperative_multi_wave_tile"
    assert artifact["shape"] == {"M": 16, "N": 16, "K": 256}
    assert artifact["production_dispatch_changed"] is False
  assert artifacts["owner_coverage"]["evidence_kind"] == "owner_coverage"
  assert artifacts["owner_coverage"]["status"] == "PASS"
  assert artifacts["gpu_owner_trace"]["status"] == "PASS"
  assert artifacts["gpu_owner_trace"]["production_dispatch_changed"] is False
  assert artifacts["staging_sum_slots"]["evidence_kind"] == "staging_sum_slots"
  assert artifacts["staging_sum_slots"]["status"] == "PASS"


def test_full_gpu_probe_joins_candidate_identity_but_stays_non_promotable():
  r4 = build_r4_evidence_artifacts()
  probe = {
    "protocol": "tinygrad.mmq_llama_five_buffer_gpu_harness.v1",
    "shape": [128, 128, 256], "passed": True,
    "verdict": "MMQ_LLAMA_FIVE_BUFFER_GPU_PASS",
    "evidence": {
      "source_sha256": "a" * 64, "binary_sha256": "b" * 64,
      "resources": {"vgpr": 256, "scratch_bytes": 0, "lds_bytes": 57856},
      "comparison": {"status": "pass", "mismatch_count": 0},
    },
  }
  candidate = build_full_gpu_probe_candidate(probe, r4_evidence=r4)
  assert candidate["candidate_id"] == "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
  assert candidate["route_id"] == "prefill_q4k_q8_1_hybrid_mmq_atom"
  assert candidate["role"] == "ffn_gate_up"
  assert candidate["quant_format"] == "Q4_K" and candidate["activation_format"] == "Q8_1"
  assert candidate["evidence"] == {"M1": True, "M2": True, "M3": True, "M4": True,
                                    "M5": True, "M6": False, "M7": False}
  assert len(candidate["canonical_identity"]) == 64
  assert candidate["owner_coverage"] is r4["owner_coverage"]
  assert candidate["staging_sum_slots"] is r4["staging_sum_slots"]
  assert candidate["complete_atom"] is False
  assert candidate["promotion_eligible"] is False
  assert candidate["default_route"] == "direct_packed"
  assert candidate["production_dispatch_changed"] is False


def test_search_report_wires_optional_full_gpu_probe_without_promotion():
  probe = {
    "shape": [128, 128, 256], "passed": True,
    "evidence": {
      "source_sha256": "a" * 64, "binary_sha256": "b" * 64,
      "resources": {"vgpr": 256, "scratch_bytes": 0},
      "comparison": {"status": "pass", "mismatch_count": 0},
    },
  }
  base = build_search_report()
  assert base["full_gpu_probe_candidate"] is None
  report = build_search_report(full_gpu_probe=probe)
  candidate = report["full_gpu_probe_candidate"]
  assert candidate is not None
  assert candidate["candidate_id"] == "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
  assert candidate["evidence"]["M1"] is True and candidate["evidence"]["M5"] is True
  assert candidate["evidence"]["M6"] is False and candidate["evidence"]["M7"] is False
  assert candidate["promotion_eligible"] is False and candidate["complete_atom"] is False
  assert report["promotion_verdict"] == "BLOCKED_UNTIL_COOPERATIVE_TILE_WIN"


def test_mmq_r5_geometry_search_ranks_non_promotable_existing_atoms_with_fake_runner():
  def fake_runner(config: BoundedMMQConfig):
    direct = 10.0
    own = {
      AMD_DS4_WARP_BACKEND_ID: 8.0,
      AMD_DS4_DOT4X4_BACKEND_ID: 6.0,
      AMD_DS4_LDS_SKELETON_BACKEND_ID: 20.0,
      FULL_GRID_BACKEND_ID: 5.0,
      LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID: 1.0,
    }[config.backend]
    return {
      "status": "PASS",
      "correctness": {"max_abs": 0.0, "atol": 0.001, "tiles": 1},
      "timing": {
        "min_ms": own,
        "median_ms": own,
        "comparator_status": "measured",
        "direct_packed": {"status": "PASS", "min_ms": direct, "samples_ms": [direct]},
      },
      "artifacts": {},
      "blockers": [],
    }

  report = build_r5_geometry_search_report(run=True, runner=fake_runner)

  assert report["schema"] == "q4k-q8-1-mmq-r5-geometry-search.v1"
  assert report["status"] == "PASS_NON_PROMOTABLE"
  assert report["promotion_eligible"] is False
  assert report["promotion_verdict"] == "R5_COOP_WIN_READY_FOR_R6"
  assert report["emitted_backend_win"] is True
  assert report["role_shape_integration"] is False
  assert report["best_candidate_id"] == "r5_llama_coop_oracle_16x16"
  assert report["ranking"][0]["speedup_vs_direct_packed"] == 10.0
  assert report["ranking"][0]["promotion_eligible"] is False
  assert "role/shape integration" in report["exact_blocker"]


def test_mmq_r5_includes_distinct_full_grid_candidate_and_keeps_r6_fail_closed():
  report = build_r5_geometry_search_report(run=False)
  full = next(row for row in report["ranking"] if row["backend"] == FULL_GRID_BACKEND_ID)
  assert full["candidate_id"] == "r5_full_grid_128x128"
  assert full["shape"] == {"M": 128, "N": 128, "K": 256}
  assert full["promotion_eligible"] is False

  synthetic = {**report, "status": "PASS_NON_PROMOTABLE", "emitted_backend_win": True,
               "promotion_verdict": "R5_COOP_WIN_READY_FOR_R6",
               "role_shape_integration": False}
  r6 = build_r6_route_gate_status(synthetic)
  assert r6["status"] == "BLOCKED_ROLE_SHAPE_INTEGRATION"
  assert r6["production_dispatch_changed"] is False
  assert r6["required_evidence"]["ffn_gate_up_only"] is False
  assert r6["required_evidence"]["negative_role_tests"] is False
  assert r6["required_evidence"]["no_hidden_direct_packed_fallback"] is False
  assert r6["role_shape_integration"]["status"] == "BLOCKED"
  assert r6["role_shape_integration"]["target"] == {"role": "ffn_gate_up", "M": 512, "N": 17408, "K": 5120}
  # An emitted full-grid win must not imply any of the one-role route gates:
  # integration, negative-role coverage, and fallback exclusion are separate
  # evidence obligations and remain fail-closed until actually measured.
  assert r6["required_evidence"] == {
    "bounded_coop_candidate_win": False,
    "ffn_gate_up_only": False,
    "negative_role_tests": False,
    "no_hidden_direct_packed_fallback": False,
  }


def test_mmq_r5_full_grid_win_is_ranked_as_emitted_but_not_promoted():
  own_ms = {
    "r5_ds4_warp_4x5": 8.0, "r5_ds4_dot4x4_8x7": 7.0,
    "r5_ds4_lds_skeleton_4x5": 9.0, "r5_ds4_coop_tile_16x16": 6.0,
    "r5_llama_coop_oracle_16x16": 2.0, "r5_full_grid_128x128": 1.0,
  }
  def fake_runner(config: BoundedMMQConfig):
    own = own_ms[next(c.candidate_id for c in R5_GEOMETRY_CANDIDATES if c.backend == config.backend)]
    return {"status": "PASS", "correctness": {"status": "PASS"},
            "timing": {"min_ms": own, "direct_packed": {"min_ms": 10.0}}}

  report = build_r5_geometry_search_report(run=True, runner=fake_runner)
  assert report["best_candidate_id"] == "r5_full_grid_128x128"
  assert report["emitted_backend_win"] is True
  assert report["promotion_verdict"] == "R5_COOP_WIN_READY_FOR_R6"
  assert report["promotion_eligible"] is False
  assert report["role_shape_integration"] is False
  r6 = build_r6_route_gate_status(report)
  assert r6["status"] == "BLOCKED_ROLE_SHAPE_INTEGRATION"

  artifact = build_r6_role_shape_integration_artifact(report)
  assert artifact["shape_matches"] is False
  assert "128x128x256" in artifact["exact_blocker"]
  assert artifact["tile_plan"]["launch_count"] == 4 * 136 * 20
  assert artifact["tile_plan"]["requires_k_epoch_accumulate"] is True
  assert artifact["tile_plan"]["monolithic_k512_compile"]["status"] == "BLOCKED"
  assert "vgpr lease" in artifact["tile_plan"]["monolithic_k512_compile"]["exact_blocker"]
  assert artifact["tile_plan"]["per_store_accumulate_sink_probe"]["status"] == "BLOCKED_TIMEOUT"
  assert artifact["tile_plan"]["k_tiled_accumulate_probe"]["status"] == "PASS_BOUNDED"
  assert artifact["tile_plan"]["k_tiled_accumulate_probe"]["mismatch_count"] == 0


def test_mmq_r5_emitted_win_survives_oracle_speed_rank():
  own_ms = {c.candidate_id: 8.0 for c in R5_GEOMETRY_CANDIDATES}
  own_ms.update({"r5_full_grid_128x128": 0.5, "r5_llama_coop_oracle_16x16": 0.1})
  def fake_runner(config: BoundedMMQConfig):
    own = own_ms[next(c.candidate_id for c in R5_GEOMETRY_CANDIDATES if c.backend == config.backend)]
    return {"status": "PASS", "correctness": {"status": "PASS"},
            "timing": {"min_ms": own, "direct_packed": {"min_ms": 10.0}}}

  report = build_r5_geometry_search_report(run=True, runner=fake_runner)
  assert report["best_candidate_id"] == "r5_llama_coop_oracle_16x16"
  assert report["emitted_backend_win"] is True
  assert report["promotion_verdict"] == "R5_COOP_WIN_READY_FOR_R6"
  assert build_r6_route_gate_status(report)["status"] == "BLOCKED_ROLE_SHAPE_INTEGRATION"


def test_mmq_full_grid_tile_plan_rejects_unaligned_shapes_fail_closed():
  blocked = build_full_grid_k_tiled_dispatch_plan({"M": 512, "N": 17408, "K": 5121})
  assert blocked["status"] == "BLOCKED"
  assert "multiples" in blocked["exact_blocker"]


def test_mmq_r6_and_r7_statuses_fail_closed_until_coop_win():
  r5 = build_r5_geometry_search_report(run=False)
  r6 = build_r6_route_gate_status(r5)
  r7 = build_r7_reduction_status()

  assert r6["status"] == "BLOCKED_NO_BOUNDED_COOP_WIN"
  assert r6["required_evidence"]["bounded_coop_candidate_win"] is False
  assert r6["production_dispatch_changed"] is False
  assert "illegal until R5" in r6["exact_blocker"]
  assert r7["status"] == "BLOCKED_REMAINING_SOURCE_CLONE_ROWS"
  assert len(r7["remaining_rows"]) >= 3
  assert all(row["status"] for row in r7["remaining_rows"])
  assert all(row.get("blocking_evidence") for row in r7["remaining_rows"])


def test_mmq_machine_search_runner_receives_bounded_configs():
  seen = []

  def fake_runner(config: BoundedMMQConfig):
    seen.append(config)
    return {
      "status": "PASS",
      "correctness": {"max_abs": 0.0, "atol": 0.001, "tiles": 1},
      "timing": {"samples_ms": [1.0], "min_ms": 1.0, "median_ms": 1.0},
      "artifacts": {"atom_source_hash": "fake", "q4k_tile_loader_source_hash": "loader"},
      "blockers": [],
    }

  report = build_search_report(run=True, warmups=2, rounds=3, runner=fake_runner)

  assert len(seen) == len(report["searchable_candidates"])
  assert all(cfg.warmups == 2 and cfg.rounds == 3 for cfg in seen)
  assert any(cfg.backend == AMD_DS4_WARP_BACKEND_ID and cfg.activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 for cfg in seen)
  assert any(cfg.backend == AMD_DS4_DOT4X4_BACKEND_ID and cfg.activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 for cfg in seen)
  assert any(cfg.backend == AMD_DS4_LDS_SKELETON_BACKEND_ID and cfg.activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 for cfg in seen)
  assert any(cfg.backend == AMD_DS4_COOP_TILE_BACKEND_ID and cfg.activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 for cfg in seen)
  assert any(cfg.backend == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID and cfg.activation_layout == ACTIVATION_LAYOUT_MMQ_DS4 for cfg in seen)
  assert all(row["run"]["status"] == "PASS" for row in report["searchable_candidates"])
  assert all(row["run"]["artifacts"]["q4k_tile_loader_source_hash"] == "loader" for row in report["searchable_candidates"])


def test_mmq_boltbeam_oracle_trace_preserves_route_gates_and_owner_contract():
  trace = build_boltbeam_oracle_trace(context=512)

  assert trace["schema"] == "boltbeam.hw_trace.v1"
  assert trace["metadata"]["production_dispatch_changed"] is False
  assert trace["metadata"]["default_route"] == "direct_packed"
  row = trace["rows"][0]
  assert row["scope"] == "kernel"
  assert row["role"] == "ffn_gate_up"
  assert row["tile_oracle"]["backend_id"] == LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID
  assert row["tile_oracle"]["target_backend_atom_id"] == AMD_DS4_COOP_TILE_BACKEND_ID
  assert row["tile_oracle"]["geometry"]["nwarps"] == 8
  assert row["tile_oracle"]["writeback_owner_count"] == 64
  assert row["resource_constraints"]["duplicate_store_count"]["eq"] == 0
  assert row["resource_constraints"]["production_dispatch_changed"]["eq"] is False
