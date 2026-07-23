from extra.qk import prefill_whole_synced as whole
from tinygrad.helpers import ProfileRangeEvent


def _report(**overrides):
  route={"prefill_route_family":whole.PREFILL_PROMOTED_CANDIDATE_ROUTE,"prefill_route_pure":True,
         "prefill_route_rolled_back":False,"prefill_route_provenance":"tinygrad_scheduler_generated"}
  route.update(overrides)
  return {"graph_gemm":True,"prefill_v2":"1","prefill_route":"fp16","logits_only":True,"route_attribution":route}


def _effective(route_id=whole.PREFILL_PROMOTED_CANDIDATE_ROUTE):
  return [{"family":"prefill_gemm","effective_route":route_id,"provenance":"tinygrad_scheduler_generated",
           "pure":True,"rolled_back_to_oracle":False}]


def test_profile_range_summary_aggregates_without_debug_sync():
  events=[ProfileRangeEvent("AMD","rmsnorm",1000,1250),ProfileRangeEvent("AMD","rmsnorm",2000,2100),
          ProfileRangeEvent("AMD","rope",3000,4000)]
  summary=whole.profile_range_summary(events)
  assert summary["kernel_count"] == 3 and summary["by_name"]["rmsnorm"]["device_ms"] == 0.35
  assert summary["by_name"]["rope"]["device_ms"] == 1.0


def test_prefill_role_routes_names_generated_candidates_only():
  assert whole._prefill_role_routes(whole.PREFILL_PROMOTED_CANDIDATE_ROUTE) == {
    role:"generated_lds_buffer2" for role in whole.PREFILL_GENERATED_DENSE_ROLES}
  assert whole._prefill_role_routes("ordinary") == {}


def test_shared_attention_attribution_reports_only_bound_8b_one_buffer_identity():
  lin = type("Linear", (), {"_prefill_full_kernel_candidate_identity": "a" * 64,
                             "_prefill_full_kernel_candidate_one_buffer": True})()
  model = type("Model", (), {"config": type("Config", (), {"n_heads": 32, "prefill_tc_attn": True, "prefill_v2": True})(),
                              "blk": [type("Block", (), {"attn_output": lin})()]})()
  attr = whole.shared_attention_attribution(model)
  assert attr["model_forward_attn_qo_identity"] == "a" * 64
  assert attr["model_forward_attn_qo_one_buffer"] is True
  model.config.n_heads = 40
  assert whole.shared_attention_attribution(model)["model_forward_attn_qo_identity"] is None


def test_route_binding_gate_accepts_promoted_generated_route(monkeypatch):
  monkeypatch.setattr(whole,"effective_routes",lambda env=None:_effective())
  report=_report(); report["candidate_set_route_census"]={
    "schema":"prefill-candidate-set-route-census.v1","passed":True,
    "policy_roles":sorted(whole.PREFILL_GENERATED_DENSE_ROLES),"missing":[],"unexpected":[],"identity_mismatches":[]}
  gate=whole.route_binding_gate(report,whole.PREFILL_PROMOTED_CANDIDATE_ROUTE,env={})
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_PASS" and gate["binding_regime"] == "generated_pure"


def test_candidate_set_route_binding_requires_actual_census(monkeypatch):
  monkeypatch.setattr(whole,"effective_routes",lambda env=None:_effective())
  env={"BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON":"{}"}
  missing=whole.route_binding_gate(_report(),env=env)
  assert missing["verdict"] == "PREFILL_ROUTE_BINDING_FAIL" and "census is missing" in missing["failures"][0]
  failed_report=_report(); failed_report["candidate_set_route_census"]={
    "schema":"prefill-candidate-set-route-census.v1","passed":False,
    "missing":[{"role":"attn_kv","shape":{"m":512,"n":1024,"k":4096},"canonical_identity":"a"*64}],
    "unexpected":[],"identity_mismatches":[]}
  assert whole.route_binding_gate(failed_report,env=env)["verdict"] == "PREFILL_ROUTE_BINDING_FAIL"


def test_generated_candidate_rejects_partial_role_ownership(monkeypatch):
  monkeypatch.setattr(whole,"effective_routes",lambda env=None:_effective())
  report=_report(); report["candidate_set_route_census"]={
    "schema":"prefill-candidate-set-route-census.v1","passed":True,"policy_roles":["ffn_gate_up"],
    "missing":[],"unexpected":[],"identity_mismatches":[]}
  gate=whole.route_binding_gate(report,env={"BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON":"{}"})
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_FAIL"
  assert any("complete dense-role ownership" in failure for failure in gate["failures"])


def test_reproducibility_band_and_authority_completeness():
  single=whole.reproducibility_band({"0":[4354.14]}); multi=whole.reproducibility_band({"0":[100.0,102.0,98.0]})
  assert single["single_sample"] is True and multi["single_sample"] is False and multi["worst_cv"] > 0
  bare=whole.authority_completeness_gate({"reproducibility_band":{"single_sample":True}})
  assert bare["ok"] is False
  full=whole.authority_completeness_gate({"comparator_id":"baseline","candidate_id":"cand-1","primitive_class":"generated_pure",
    "threshold":{"pp512_min":3300},"ledger":"docs/prefill-lessons-ledger.md",
    "reproducibility_band":{"single_sample":False,"worst_cv":0.01},"quality_gate":{"status":"PASS"}})
  assert full["ok"] is True


def test_candidate_compiler_state_scope_restores_typed_state():
  import tinygrad.codegen.opt.postrange as pr
  names=("_WARMSTART_OPTS","_WARMSTART_CANDIDATE_CONTEXTS"); saved={name:getattr(pr,name,None) for name in names}
  try:
    original=({"old":"opts"},{"old":"ctx"})
    for name,value in zip(names,original): setattr(pr,name,value)
    with whole._scoped_candidate_compiler_state():
      for name in names: setattr(pr,name,{"new"})
    assert tuple(getattr(pr,name) for name in names) == original
  finally:
    for name,value in saved.items(): setattr(pr,name,value)


def test_route_binding_gate_can_require_q4k_substrate(monkeypatch):
  effective = _effective() + [{"family":"prefill_q4k", "effective_route":"prefill_q4k_int8_wmma_tiled_research",
    "provenance":"machine_authored_generated", "pure":True, "rolled_back_to_oracle":False}]
  monkeypatch.setattr(whole, "effective_routes", lambda env=None: effective)
  report = _report(prefill_q4k_route_family="prefill_q4k_int8_wmma_tiled_research",
    prefill_q4k_route_pure=True, prefill_q4k_route_rolled_back=False,
    prefill_q4k_route_provenance="machine_authored_generated")
  gate = whole.route_binding_gate(report, "prefill_q4k_int8_wmma_tiled_research", env={})
  assert gate["verdict"] == "PREFILL_ROUTE_BINDING_PASS" and gate["selected_family"] == "prefill_q4k"
