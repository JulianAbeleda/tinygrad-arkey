from extra.qk.prefill.hybrid_route_quality_gate import HYBRID_ROUTE, compare_results


def _side(route: bool, tokens=(7, 11, 13)):
  return {"effective_routes": [HYBRID_ROUTE] if route else ["prefill_v2_scheduler_matmul_default"],
          "outputs": [{"case": i, "token": [token]} for i, token in enumerate(tokens)]}


def test_hybrid_quality_gate_accepts_bound_whole_model_greedy_parity():
  report = compare_results(_side(False), _side(True), baseline_healthy=True, candidate_healthy=True)
  assert report["status"] == "PASS"
  assert report["passed"] is True
  assert report["route_bound"] is True
  assert report["case_count"] == 3


def test_hybrid_quality_gate_rejects_token_drift_or_unhealthy_device():
  drift = compare_results(_side(False), _side(True, tokens=(7, 12, 13)),
                          baseline_healthy=True, candidate_healthy=True)
  unhealthy = compare_results(_side(False), _side(True), baseline_healthy=True, candidate_healthy=False)
  assert drift["status"] == "FAIL" and drift["value"] == 0.0
  assert unhealthy["status"] == "FAIL"
