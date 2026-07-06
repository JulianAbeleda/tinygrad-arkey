from extra.qk.prefill_graph_gemm_route_bound_stage_gate import build_report


def test_route_bound_stage_gate_default_report_is_conservative_without_amd_run():
  report = build_report(run_amd=False)
  assert report["schema"] == "prefill-graph-gemm-route-bound-stage-gate.v1"
  assert report["route_id"] == "prefill_v2_scheduler_matmul_default"
  assert report["shape"] == {"m": 512, "n": 512, "k": 512}
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_ROUTE_BOUND_LOCAL_STAGE_MISSING"
  assert report["evidence"]["route_bound_executes"] is False
  assert report["evidence"]["route_bound_numeric_ok"] is False
  assert report["evidence"]["route_bound_local_stage_present"] is False
  assert report["remaining_blocker"] == "default fp16 prefill route emits WMMA but not generated LOCAL operand staging"
