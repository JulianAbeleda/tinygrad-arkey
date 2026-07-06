from extra.qk import prefill_graph_gemm_medium_stage_gate as gate


def test_medium_stage_gate_static_report_is_skipped():
  report = gate.build_report(run_amd=False, artifact=False)
  assert report["schema"] == "prefill-graph-gemm-medium-stage-gate.v1"
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_MEDIUM_LOCAL_STAGE_BLOCKED"
  assert report["evidence"]["run_amd"] is False
  assert "skipped" in report["cases"]


def test_medium_stage_gate_classifies_forced_stage_blocker(monkeypatch):
  results = [
    {"status": "ok", "tflops": 40.0},
    {"status": "WRONG rr=1.2e+00", "tflops": 0.0},
    {"status": "CompileError", "tflops": 0.0},
    {"status": "ok", "tflops": 42.5},
    {"status": "CompileError", "tflops": 0.0},
  ]

  def fake_run_config(*args, **kwargs):
    return results.pop(0)

  monkeypatch.setattr(gate, "_run_config", fake_run_config)
  report = gate.build_report(run_amd=True, artifact=False)
  assert report["evidence"]["baseline_table_local_ok"] is True
  assert report["evidence"]["pre_wmma_forced_local_ok"] is False
  assert report["evidence"]["post_local_stage_ok"] is False
  assert report["evidence"]["post_tile_b_stage_ok"] is True
  assert report["evidence"]["scalar_post_local_stage_ok"] is False
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_MEDIUM_LOCAL_STAGE_PASS"
