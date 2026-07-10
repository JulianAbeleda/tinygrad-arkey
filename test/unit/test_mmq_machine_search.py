from extra.qk.mmq_bounded_harness import (
  ACTIVATION_LAYOUT_MMQ_DS4, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID,
  BoundedMMQConfig,
)
from extra.qk.mmq_machine_search import build_search_report


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
  assert report["promotion_verdict"] == "BLOCKED_UNTIL_COOPERATIVE_TILE_PASS"
  assert report["searchable_components"] == [
    "DS4 layout",
    "DS4 reference correctness",
    "Q4_K x DS4 formula",
    "Q4_K tile loader",
    "sudot4 primitive availability",
    "direct DS4 GPU atom",
    "R3 LDS skeleton atom",
  ]
  assert [row["status"] for row in report["done_components"]] == ["done"] * 7
  assert {row["component"] for row in report["done_components"]} == set(report["searchable_components"])

  rows = {row["candidate_id"]: row for row in report["searchable_candidates"]}
  assert set(rows) == {
    "direct_packed_comparator",
    "ds4_reference_formula",
    "amd_ds4_warp_direct",
    "staged_ds4_reference_probe",
    "amd_ds4_dot4x4_packed",
    "amd_ds4_lds_skeleton",
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

  blocked = {row["candidate_id"]: row for row in report["blocked_candidates"]}
  assert "amd_ds4_dot4x4_packed" not in blocked
  assert blocked["cooperative_multi_wave_tile"]["status"] == "blocked"
  assert blocked["full_14b_prefill_route"]["status"] == "blocked"


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
  assert all(row["run"]["status"] == "PASS" for row in report["searchable_candidates"])
  assert all(row["run"]["artifacts"]["q4k_tile_loader_source_hash"] == "loader" for row in report["searchable_candidates"])
