from extra.qk.prefill_graph_gemm_fp16_stage_gate import build_report


def test_fp16_stage_gate_default_report_is_conservative_without_amd_run():
  report = build_report(run_amd=False)
  assert report["schema"] == "prefill-graph-gemm-fp16-single-operand-stage-gate.v1"
  assert report["route_id"] == "generated_fp16_shaped_wmma_local_stage_probe"
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_FP16_SINGLE_OPERAND_STAGE_BLOCKED_IMPLEMENTATION_MISSING"
  assert report["api"]["ops_stage_available"] is True
  assert report["api"]["bufferize_opts_local_removable_false"] is True
  assert report["api"]["pm_add_buffers_local_available"] is True
  assert report["api"]["shaped_wmma_helper_available"] is True
  assert report["required_evidence"]["emitted_amd_source_has_fp16_wmma"] is False
  assert report["required_evidence"]["custom_probe_has_no_raw_ops_ins_marker"] is False
  assert report["remaining_blocker"] == "generated fp16 single-operand WMMA LOCAL staging probe not implemented or not run"


def test_fp16_stage_gate_both_operands_default_report_is_conservative_without_amd_run():
  report = build_report(run_amd=False, both_operands=True)
  assert report["schema"] == "prefill-graph-gemm-fp16-single-operand-stage-gate.v1"
  assert report["operand_mode"] == "both_operands"
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_FP16_BOTH_OPERANDS_STAGE_BLOCKED_IMPLEMENTATION_MISSING"
  assert report["api"]["ops_stage_available"] is True
  assert report["required_evidence"]["emitted_amd_source_has_expected_local_buffer_count"] is False
  assert report["remaining_blocker"] == "generated fp16 both-operand WMMA LOCAL staging probe not implemented or not run"
