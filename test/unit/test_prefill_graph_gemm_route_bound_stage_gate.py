from extra.qk.prefill_graph_gemm_route_bound_stage_gate import build_report
from tinygrad.codegen.opt import KernelOptError, postrange


def test_route_bound_stage_gate_default_report_is_conservative_without_amd_run():
  report = build_report(run_amd=False)
  assert report["schema"] == "prefill-graph-gemm-route-bound-stage-gate.v1"
  assert report["route_id"] == "prefill_v2_scheduler_matmul_default"
  assert report["shape"] == {"m": 512, "n": 512, "k": 512}
  assert report["verdict"] == "PREFILL_GRAPH_GEMM_ROUTE_BOUND_LOCAL_STAGE_MISSING"
  assert report["evidence"]["route_bound_executes"] is False
  assert report["evidence"]["route_bound_numeric_ok"] is False
  assert report["evidence"]["route_bound_local_stage_present"] is False
  assert report["evidence"]["pin_clock"] is False
  assert report["remaining_blocker"] == "default fp16 prefill route emits WMMA but not generated LOCAL operand staging"


def test_tc_local_stage_rejects_unvalidated_b_operand_modes(monkeypatch):
  monkeypatch.setattr(postrange, "_tc_local_stage_mode", lambda: "b")
  try:
    postrange._tc_local_stage_wmma_sources([], ())
  except KernelOptError as e:
    assert "supports only a/off" in str(e)
  else:
    raise AssertionError("expected KernelOptError")


def test_tc_local_stage_off_is_noop(monkeypatch):
  srcs = []
  monkeypatch.setattr(postrange, "_tc_local_stage_mode", lambda: "off")
  assert postrange._tc_local_stage_wmma_sources(srcs, ()) is srcs


def test_tc_local_stage_disabled_bypasses_unvalidated_mode(monkeypatch):
  srcs = []
  monkeypatch.setattr(postrange, "_tc_local_stage_mode", lambda: "b")
  assert postrange._tc_local_stage_wmma_sources(srcs, (), enabled=False) is srcs
