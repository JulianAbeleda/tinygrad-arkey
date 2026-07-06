from extra.qk.prefill_graph_gemm_coop_route_contract_gate import build_report


def test_coop_route_contract_gate_blocks_until_route_bound_case_exists():
  report = build_report(artifact=False)
  assert report["schema"] == "prefill-graph-gemm-coop-route-contract-gate.v1"
  assert report["route_id"] == "prefill_v2_scheduler_matmul_default"
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_COOP_ROUTE_CONTRACT_BLOCKED"
  assert report["required_evidence"]["custom_coop_partition_probe_pass"] is True
  assert report["required_evidence"]["medium_gate_has_b_tile_operand_stage"] is True
  assert report["required_evidence"]["medium_gate_defines_route_bound_coop_partition_case"] is True
  assert report["required_evidence"]["medium_gate_route_bound_coop_partition_executes"] is False
  assert report["required_evidence"]["route_bound_coop_partition_beats_baseline"] is False
