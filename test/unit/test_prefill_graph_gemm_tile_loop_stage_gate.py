from extra.qk.prefill_graph_gemm_tile_loop_stage_gate import build_report


def test_tile_loop_stage_gate_default_report_is_conservative_without_amd_run():
  report = build_report(run_amd=False, artifact=False)
  assert report["schema"] == "prefill-graph-gemm-tile-loop-stage-gate.v1"
  assert report["route_id"] == "generated_fp16_shaped_wmma_tile_loop_local_stage_probe"
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_TILE_LOOP_LOCAL_STAGE_BLOCKED"
  assert report["api"]["ops_stage_available"] is True
  assert report["api"]["bufferize_opts_local_removable_false"] is True
  assert report["api"]["stage_shape_is_lane_only"] is True
  assert report["api"]["pm_add_buffers_local_available"] is True
  assert report["api"]["shaped_wmma_helper_available"] is True
  assert report["required_evidence"]["emitted_amd_source_has_tile_sized_local"] is False
  assert report["required_evidence"]["emitted_amd_source_avoids_whole_gemm_sized_local"] is False
  assert report["remaining_blocker"] == "generated tile-loop WMMA LOCAL staging has not been proven on AMD with bounded tile-shaped LDS"
