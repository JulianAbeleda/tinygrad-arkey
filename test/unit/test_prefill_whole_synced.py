from extra.qk import prefill_whole_synced as whole


def _report(route_id=whole.PATH1_MVP_ROUTE, **route_overrides):
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


def _path1_report(**route_overrides):
  return _report(whole.PATH1_MVP_ROUTE, **route_overrides)


def _effective(route_id):
  return [{
    "family": "prefill_gemm",
    "effective_route": route_id,
    "provenance": "tinygrad_scheduler_generated",
    "pure": True,
    "rolled_back_to_oracle": False,
  }]


def test_path1_mvp_env_sets_mixed_generated_pipe_flags():
  env = {}
  whole.apply_path1_mvp_env(env)
  assert env == whole.PATH1_MVP_ENV


def test_path1_mvp_gate_accepts_generated_pipe_route():
  gate = whole.path1_mvp_gate(_path1_report())
  assert gate["verdict"] == "PATH1_MIXED_PREFILL_MVP_PASS"
  assert gate["failures"] == []


def test_path1_mvp_gate_rejects_raw_oracle_route():
  gate = whole.path1_mvp_gate(_path1_report(
    prefill_route_family="prefill_pipe_role_selective_generated",
    prefill_route_pure=False,
    prefill_route_rolled_back=True,
    prefill_route_provenance="external_handwritten_kernel",
  ))
  assert gate["verdict"] == "PATH1_MIXED_PREFILL_MVP_FAIL"
  assert any("prefill_route_family" in failure for failure in gate["failures"])
  assert any("prefill_route_pure" in failure for failure in gate["failures"])
  assert any("prefill_route_rolled_back" in failure for failure in gate["failures"])


def test_prefill_role_routes_names_pipe_only_roles():
  assert whole._prefill_role_routes(whole.PATH1_MVP_ROUTE) == {
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
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(whole.PATH1_MVP_ROUTE))
  gate = whole.route_binding_gate(_path1_report(), whole.PATH1_MVP_ROUTE, env={})
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
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(whole.PATH1_MVP_ROUTE))
  gate = whole.route_binding_gate(_path1_report(), whole.PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE, env={
    "PREFILL_WMMA_LDS_PRIMITIVE": "1",
    "PREFILL_DBUF": "1",
  })
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_FAIL"
  assert any("required_route" in failure for failure in gate["failures"])
  assert any("pipe-only" in failure for failure in gate["failures"])


def test_route_binding_gate_marks_dbuf_flags_on_pipe_only_route(monkeypatch):
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: _effective(whole.PATH1_MVP_ROUTE))
  gate = whole.route_binding_gate(_path1_report(), env={"PREFILL_DBUF": "1"})
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_FAIL"
  assert gate["lds_dbuf_requested"] is True
  assert any("pipe-only" in failure for failure in gate["failures"])
