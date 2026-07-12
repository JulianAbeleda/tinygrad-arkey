from extra.qk import prefill_whole_synced as whole


def _report(route_id=whole.PREFILL_WMMA_PIPE_ROUTE, **route_overrides):
  route = {
    "prefill_route_family": route_id,
    "prefill_route_pure": True,
    "prefill_route_rolled_back": False,
    "prefill_route_provenance": "tinygrad_scheduler_generated",
  }
  route.update(route_overrides)
  return {
    "graph_gemm": True,
    "prefill_v2": "1",
    "prefill_route": "fp16",
    "prefill_chunked": "0",
    "logits_only": True,
    "route_attribution": route,
  }


def _effective(route_id):
  return [{
    "family": "prefill_gemm",
    "effective_route": route_id,
    "provenance": "tinygrad_scheduler_generated",
    "pure": True,
    "rolled_back_to_oracle": False,
  }]


def test_prefill_role_routes_names_pipe_only_roles():
  assert whole._prefill_role_routes(whole.PREFILL_WMMA_PIPE_ROUTE) == {
    "attn_qo": "pipe",
    "attn_kv": "pipe",
    "ffn_down": "pipe",
    "ffn_gate_up": "pipe",
  }


def test_prefill_role_routes_names_composed_lds_dbuf_role():
  assert whole._prefill_role_routes(whole.PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE) == {
    "attn_qo": "pipe",
    "attn_kv": "generated_pipe_no_local_stage",
    "ffn_down": "pipe",
    "ffn_gate_up": "lds_dbuf",
  }


def test_prefill_role_routes_names_decoupled_lds_dbuf_role():
  assert whole._prefill_role_routes(whole.PREFILL_WMMA_LDS_DBUF_MIXED_ROUTE) == {
    "attn_qo": "raw_pipe_oracle",
    "attn_kv": "raw_pipe_oracle",
    "ffn_down": "raw_pipe_oracle",
    "ffn_gate_up": "lds_dbuf",
  }


def test_route_binding_gate_accepts_existing_pipe_route(monkeypatch):
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(whole.PREFILL_WMMA_PIPE_ROUTE))
  gate = whole.route_binding_gate(_report(), whole.PREFILL_WMMA_PIPE_ROUTE, env={})
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_PASS"
  assert gate["failures"] == []


def test_route_binding_gate_accepts_composed_route_when_effective(monkeypatch):
  route_id = whole.PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(route_id))
  gate = whole.route_binding_gate(_report(route_id, prefill_route_pure=False,
                                          prefill_route_provenance="compiler_primitive_spec_owned"), route_id, env={
    "PREFILL_WMMA_LDS_PRIMITIVE": "1",
    "PREFILL_DBUF": "1",
  })
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_PASS"
  assert gate["failures"] == []
  assert gate["lds_dbuf_requested"] is True


def test_route_binding_gate_accepts_decoupled_lds_route_when_effective(monkeypatch):
  route_id = whole.PREFILL_WMMA_LDS_DBUF_MIXED_ROUTE
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(route_id))
  gate = whole.route_binding_gate(_report(route_id, prefill_route_pure=False,
                                          prefill_route_provenance="compiler_primitive_spec_owned"), route_id, env={
    "PREFILL_WMMA_LDS_PRIMITIVE": "1",
    "PREFILL_DBUF": "1",
  })
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_PASS"
  assert gate["failures"] == []
  assert gate["lds_dbuf_requested"] is True


def test_route_binding_gate_rejects_composed_requirement_before_lane_a(monkeypatch):
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(whole.PREFILL_WMMA_PIPE_ROUTE))
  gate = whole.route_binding_gate(_report(), whole.PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE, env={
    "PREFILL_WMMA_LDS_PRIMITIVE": "1",
    "PREFILL_DBUF": "1",
  })
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_FAIL"
  assert any("required_route" in failure for failure in gate["failures"])
  assert any("pipe-only" in failure for failure in gate["failures"])


def test_route_binding_gate_marks_dbuf_flags_on_pipe_only_route(monkeypatch):
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(whole.PREFILL_WMMA_PIPE_ROUTE))
  gate = whole.route_binding_gate(_report(), env={"PREFILL_DBUF": "1"})
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_FAIL"
  assert gate["lds_dbuf_requested"] is True
  assert any("pipe-only" in failure for failure in gate["failures"])


def test_measurement_regime_names_generated_vs_hand_regimes():
  gen = whole.measurement_regime({"route_attribution": {"prefill_route_provenance": "tinygrad_scheduler_generated"}})
  hand = whole.measurement_regime({"route_attribution": {"prefill_route_provenance": "external_handwritten_kernel"}})
  assert gen["regime_id"] == "generated_pure"
  assert gen["authoritative_for_generated_promotion"] is True
  assert hand["regime_id"] == "hand_external_reference"
  assert hand["authoritative_for_generated_promotion"] is False


def test_reproducibility_band_flags_single_sample_and_computes_cv():
  single = whole.reproducibility_band({"0": [4354.14]})
  assert single["single_sample"] is True
  multi = whole.reproducibility_band({"0": [100.0, 102.0, 98.0]})
  assert multi["single_sample"] is False
  assert multi["worst_cv"] > 0.0
  assert multi["per_chunk"]["0"]["n"] == 3


def test_authority_completeness_gate_refuses_without_checklist_fields():
  bare = whole.authority_completeness_gate({"reproducibility_band": {"single_sample": True}})
  assert bare["ok"] is False
  assert set(bare["missing"]) == {
    "comparator_id", "reproducibility_band", "candidate_id", "primitive_class", "threshold",
    "ledger", "quality_gate_pass",
  }


def test_authority_completeness_gate_passes_with_full_checklist():
  full = whole.authority_completeness_gate({
    "comparator_id": "baseline-after-s10", "candidate_id": "cand-1", "primitive_class": "generated_pure",
    "threshold": {"pp512_min": 1629}, "ledger": "docs/ledger.md",
    "reproducibility_band": {"single_sample": False, "worst_cv": 0.01},
    "quality_gate": {"status": "PASS"},
  })
  assert full["ok"] is True
  assert full["missing"] == []
