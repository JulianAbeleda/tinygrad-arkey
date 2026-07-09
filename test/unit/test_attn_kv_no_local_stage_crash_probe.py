from extra.qk.prefill.attn_kv_no_local_stage_crash_probe import build_report


def test_crash_probe_scopes_cases_without_execution():
  report = build_report(execute=False)

  assert report["schema"] == "s10-attn-kv-no-local-stage-crash-probe.v1"
  assert report["verdict"] == "S10_ATTN_KV_NO_LOCAL_STAGE_CRASH_PROBE_SCOPED"
  assert [c["name"] for c in report["cases"]] == [
    "attn_kv_generated_no_local_stage",
    "attn_kv_generated_no_local_stage_hip",
    "attn_kv_raw_fallback_safety",
    "attn_qo_generated_pipe_control",
  ]
  assert all(c["status"] == "not_run" for c in report["cases"])


def test_crash_probe_classifies_signal_139_with_fallback_pass():
  def fake_runner(case):
    if case["name"] == "attn_kv_generated_no_local_stage":
      return {"returncode": -11, "stdout_tail": [], "stderr_tail": [], "payload": None}
    return {"returncode": 0, "stdout_tail": [], "stderr_tail": [], "payload": {
      "passed": True, "finite": True, "nonzero": True, "rel_rmse": 0.0,
      "route_transport": "ordinary_generated_matmul",
      "warmstart_key_present_after_route": True,
      "warmstart_stats": {"apply": 1},
    }}

  report = build_report(execute=True, runner=fake_runner)

  assert report["verdict"] == "S10_ATTN_KV_NO_LOCAL_STAGE_ISOLATED_SIGNAL_139"
  cases = {c["name"]: c for c in report["cases"]}
  assert cases["attn_kv_generated_no_local_stage"]["status"] == "signal_139"
  assert cases["attn_kv_raw_fallback_safety"]["status"] == "correctness_pass"


def test_crash_probe_classifies_hip_backend_blocked():
  def fake_runner(case):
    if case["name"] == "attn_kv_generated_no_local_stage_hip":
      return {"returncode": 1, "stdout_tail": [], "stderr_tail": ["comgr fail"], "payload": None}
    return {"returncode": 0, "stdout_tail": [], "stderr_tail": [], "payload": {
      "passed": True, "finite": True, "nonzero": True, "rel_rmse": 0.0,
      "route_transport": "ordinary_generated_matmul",
      "warmstart_key_present_after_route": True,
      "warmstart_stats": {"apply": 1},
    }}

  report = build_report(execute=True, runner=fake_runner)

  assert report["verdict"] == "S10_ATTN_KV_NO_LOCAL_STAGE_HIP_BACKEND_BLOCKED"


def test_crash_probe_classifies_no_local_stage_pass():
  def fake_runner(case):
    return {"returncode": 0, "stdout_tail": [], "stderr_tail": [], "payload": {
      "passed": True, "finite": True, "nonzero": True, "rel_rmse": 0.0,
      "route_transport": "ordinary_generated_matmul",
      "warmstart_key_present_after_route": True,
      "warmstart_stats": {"apply": 1},
    }}

  report = build_report(execute=True, runner=fake_runner)

  assert report["verdict"] == "S10_ATTN_KV_NO_LOCAL_STAGE_ISOLATED_PASS"
