from extra.qk.prefill_graph_gemm_coop_partition_gate import build_report


def test_coop_partition_gate_default_report_is_conservative_without_amd_run():
  report = build_report(run_amd=False, artifact=False)
  assert report["schema"] == "prefill-graph-gemm-coop-partition-gate.v1"
  assert report["route_id"] == "generated_fp16_shaped_wmma_coop_b_tile_partition_probe"
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_COOP_PARTITION_PROBE_BLOCKED"
  assert report["api"]["cooperative_stage_lanemap_available"] is True
  assert report["api"]["b_tile_lanemap_valid"] is True
  assert report["api"]["b_tile_unique_elements"] == 256
  assert report["api"]["producer_lanes"] == 16
  assert report["api"]["consumer_lanes"] == 32
  assert report["required_evidence"]["emitted_amd_source_has_unique_256_half_local"] is False
  assert report["required_evidence"]["custom_probe_output_matches_direct"] is False
