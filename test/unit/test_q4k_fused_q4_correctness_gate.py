from extra.qk.q4k_fused_q4_correctness_gate import _telemetry


def test_fused_gate_telemetry_records_first_compiler_failure_and_no_fallback():
  report = _telemetry(lambda: (_ for _ in ()).throw(RuntimeError("compiler: bad intrinsic")))
  assert report["ok"] is False
  assert report["first_compiler_failure"] == "RuntimeError: compiler: bad intrinsic"
  assert report["fallback"] is False
