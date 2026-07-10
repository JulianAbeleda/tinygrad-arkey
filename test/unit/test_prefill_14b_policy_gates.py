import json

from tinygrad.llm import route_policy

from extra.qk.model_profiles import qwen3_14b_q4k_m_gfx1100_profile
import extra.qk.prefill_14b_model_authority_gate as model_gate
import extra.qk.prefill_14b_q6_decision_gate as q6_gate

PROFILE_14B = qwen3_14b_q4k_m_gfx1100_profile()
ATTN_QO_14B = PROFILE_14B.role_shape("attn_qo")
ATTN_KV_14B = PROFILE_14B.role_shape("attn_kv")
FFN_DOWN_14B = PROFILE_14B.role_shape("ffn_down")
FFN_GATE_UP_14B = PROFILE_14B.role_shape("ffn_gate_up")


def _policy_shape(row):
  return {"rows": row.N, "cols": row.K}


def test_prefill_14b_model_authority_gate_blocks_without_route_policy_selection():
  route_policy.set_qk_route_policy(None)
  report = model_gate.build()
  assert report["schema"] == "prefill_14b_model_authority_gate.v1"
  assert report["classified_blocker"] is True
  assert report["verdict"] == "PREFILL_14B_MODEL_AUTHORITY_BLOCKED"
  assert report["policy_evidence"]["policy_loaded"] is False
  assert report["policy_evidence"]["policy_selectable_count"] == 0
  assert report["route_matrix"], "route rows should be present"
  assert len(report["route_matrix"]) == len(report["target_routes"])

  # The registry and gate should stay stable when run from a clean process.
  assert "prefill_q4k_int8_wmma_generated_research" in report["policy_evidence"]["candidate_routes_present"]


def test_prefill_14b_model_authority_gate_loaded_policy_branch_is_safe():
  route_policy.set_qk_route_policy({
    "selected": {
      "prefill_q4k_int8_wmma_tiled_research": {
        "selected_route": "prefill_q4k_int8_wmma_tiled_research",
        "shape": _policy_shape(ATTN_KV_14B),
      }
    }
  })
  try:
    report = model_gate.build()
  finally:
    route_policy.set_qk_route_policy(None)
  assert report["policy_evidence"]["policy_loaded"] is True
  assert report["policy_evidence"]["policy_selected_routes"] == ["prefill_q4k_int8_wmma_tiled_research"]
  assert report["classified_blocker"] is True


def test_prefill_14b_model_authority_gate_accepts_hybrid_atom_one_role_scope(tmp_path):
  policy_path = tmp_path / "hybrid_atom_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "model_id": PROFILE_14B.id,
    "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [{
      "role": FFN_GATE_UP_14B.role,
      "shape": _policy_shape(FFN_GATE_UP_14B),
      "quant": "Q4_K",
      "selected_route": "prefill_14b_q4k_q8_1_hybrid_mmq_atom",
      "route_params": {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "1", "PREFILL_ROUTE_STRICT": "1"},
      "atom_available": True,
    }],
  }))
  policy = route_policy.load_qk_route_policy(str(policy_path))
  route_policy.set_qk_route_policy(policy)
  try:
    report = model_gate.build(
      target_route_ids=("prefill_14b_q4k_q8_1_hybrid_mmq_atom",),
      representative_shapes=((FFN_GATE_UP_14B.role, FFN_GATE_UP_14B.M, FFN_GATE_UP_14B.N, FFN_GATE_UP_14B.K),),
      scope="unit hybrid atom authority scope")
  finally:
    route_policy.set_qk_route_policy(None)

  assert report["verdict"] == "PREFILL_14B_MODEL_AUTHORITY_PASS"
  assert report["classified_blocker"] is False
  assert report["target_routes"] == ("prefill_14b_q4k_q8_1_hybrid_mmq_atom",)
  assert report["policy_evidence"]["candidate_routes_present"] == ["prefill_14b_q4k_q8_1_hybrid_mmq_atom"]
  assert report["policy_evidence"]["policy_selected_roles"] == [{
    "role": "ffn_gate_up",
    "rows": 17408,
    "cols": 5120,
    "selected_route": "prefill_14b_q4k_q8_1_hybrid_mmq_atom",
  }]


def test_qk_route_policy_accepts_prefill_direct_and_tiled_shape_rows(tmp_path):
  policy_path = tmp_path / "prefill_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "model_id": "shape-profile",
    "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [
      {"role": FFN_GATE_UP_14B.role, "shape": _policy_shape(FFN_GATE_UP_14B), "quant": "Q4_K",
       "selected_route": "prefill_q4k_direct_tile4x4_default", "route_params": {}},
      {"role": FFN_DOWN_14B.role, "shape": _policy_shape(FFN_DOWN_14B), "quant": "Q6_K",
       "selected_route": "prefill_q6k_direct_generated", "route_params": {}},
      {"role": FFN_GATE_UP_14B.role, "shape": _policy_shape(FFN_GATE_UP_14B), "quant": "Q4_K",
       "selected_route": "prefill_q4k_int8_wmma_generated_research", "route_params": {}},
      {"role": "attn_q", "shape": _policy_shape(ATTN_QO_14B), "quant": "Q4_K",
       "selected_route": "prefill_q4k_int8_wmma_tiled_research", "route_params": {}},
      {"role": ATTN_KV_14B.role, "shape": _policy_shape(ATTN_KV_14B), "quant": "Q4_K",
       "selected_route": "prefill_q4k_int8_wmma_tiled_research", "route_params": {}},
    ],
  }))
  policy = route_policy.load_qk_route_policy(str(policy_path))
  assert [row["selected_route"] for row in policy["prefill_gen"]] == [
    "prefill_q4k_direct_tile4x4_default",
    "prefill_q6k_direct_generated",
    "prefill_q4k_int8_wmma_generated_research",
    "prefill_q4k_int8_wmma_tiled_research",
    "prefill_q4k_int8_wmma_tiled_research",
  ]
  route_policy.set_qk_route_policy(policy)
  try:
    assert route_policy.qk_route_policy_selected("prefill_q4k_direct_tile4x4_default", _policy_shape(FFN_GATE_UP_14B))
    assert route_policy.qk_route_policy_selected("prefill_q6k_direct_generated", _policy_shape(FFN_DOWN_14B))
    assert route_policy.qk_route_policy_selected("prefill_q4k_int8_wmma_generated_research", _policy_shape(FFN_GATE_UP_14B))
    assert route_policy.qk_route_policy_selected("prefill_q4k_int8_wmma_tiled_research", _policy_shape(ATTN_QO_14B))
    assert route_policy.qk_route_policy_selected("prefill_q4k_int8_wmma_tiled_research", _policy_shape(ATTN_KV_14B))
    assert not route_policy.qk_route_policy_selected("prefill_q4k_int8_wmma_tiled_research", {"rows": 4096, "cols": 4096})
    assert route_policy.qk_route_policy_selects_prefill_generated(ATTN_KV_14B.N, ATTN_KV_14B.K)
  finally:
    route_policy.set_qk_route_policy(None)


def test_prefill_14b_hybrid_mmq_atom_policy_is_one_role_opt_in(tmp_path):
  policy_path = tmp_path / "m7_prefill_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "model_id": PROFILE_14B.id,
    "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [
      {"role": FFN_GATE_UP_14B.role, "shape": _policy_shape(FFN_GATE_UP_14B), "quant": "Q4_K",
       "selected_route": "prefill_14b_q4k_q8_1_hybrid_mmq_atom",
       "route_params": {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "1", "PREFILL_ROUTE_STRICT": "1"},
       "atom_available": True},
      {"role": ATTN_QO_14B.role, "shape": _policy_shape(ATTN_QO_14B), "quant": "Q4_K",
       "selected_route": "prefill_q4k_direct_tile4x4_default", "route_params": {}},
      {"role": ATTN_KV_14B.role, "shape": _policy_shape(ATTN_KV_14B), "quant": "Q4_K",
       "selected_route": "prefill_q4k_direct_tile4x4_default", "route_params": {}},
      {"role": FFN_DOWN_14B.role, "shape": _policy_shape(FFN_DOWN_14B), "quant": "Q6_K",
       "selected_route": "prefill_q6k_direct_generated", "route_params": {}},
    ],
  }))
  policy = route_policy.load_qk_route_policy(str(policy_path))
  assert [row["selected_route"] for row in policy["prefill_mmq_atom"]] == [
    "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
  ]
  assert [row["selected_route"] for row in policy["prefill_gen"]] == [
    "prefill_q4k_direct_tile4x4_default",
    "prefill_q4k_direct_tile4x4_default",
    "prefill_q6k_direct_generated",
  ]
  route_policy.set_qk_route_policy(policy)
  try:
    assert route_policy.qk_route_policy_selected(
      "prefill_14b_q4k_q8_1_hybrid_mmq_atom", _policy_shape(FFN_GATE_UP_14B))
    assert not route_policy.qk_route_policy_selected(
      "prefill_14b_q4k_q8_1_hybrid_mmq_atom", _policy_shape(ATTN_QO_14B))
    assert route_policy.qk_route_policy_selected("prefill_q4k_direct_tile4x4_default", _policy_shape(ATTN_QO_14B))
    assert route_policy.qk_route_policy_selected("prefill_q4k_direct_tile4x4_default", _policy_shape(ATTN_KV_14B))
    assert route_policy.qk_route_policy_selected("prefill_q6k_direct_generated", _policy_shape(FFN_DOWN_14B))
  finally:
    route_policy.set_qk_route_policy(None)


def test_prefill_14b_hybrid_mmq_atom_policy_fails_closed_when_atom_unavailable(tmp_path):
  policy_path = tmp_path / "m7_prefill_policy_unavailable.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "routes": [{
      "role": FFN_GATE_UP_14B.role,
      "shape": _policy_shape(FFN_GATE_UP_14B),
      "quant": "Q4_K",
      "selected_route": "prefill_14b_q4k_q8_1_hybrid_mmq_atom",
      "route_params": {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "1", "PREFILL_ROUTE_STRICT": "1"},
    }],
  }))
  try:
    route_policy.load_qk_route_policy(str(policy_path))
  except ValueError as exc:
    assert "fail-closed" in str(exc)
  else:
    raise AssertionError("hybrid MMQ atom policy must fail closed without atom_available=true")


def test_prefill_14b_hybrid_mmq_atom_policy_rejects_m8_roles_until_expanded(tmp_path):
  policy_path = tmp_path / "m7_prefill_policy_bad_role.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "routes": [{
      "role": ATTN_QO_14B.role,
      "shape": _policy_shape(ATTN_QO_14B),
      "quant": "Q4_K",
      "selected_route": "prefill_14b_q4k_q8_1_hybrid_mmq_atom",
      "route_params": {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "1", "PREFILL_ROUTE_STRICT": "1"},
      "atom_available": True,
    }],
  }))
  try:
    route_policy.load_qk_route_policy(str(policy_path))
  except ValueError as exc:
    assert "only supports role='ffn_gate_up'" in str(exc)
  else:
    raise AssertionError("M7 scaffold must reject M8 roles until explicit expansion rows land")


def test_prefill_14b_q6_decision_gate_sees_direct_policy_selection(tmp_path):
  policy_path = tmp_path / "q6_prefill_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "routes": [{
      "role": FFN_DOWN_14B.role,
      "shape": _policy_shape(FFN_DOWN_14B),
      "quant": "Q6_K",
      "selected_route": "prefill_q6k_direct_generated",
      "route_params": {},
    }],
  }))
  policy = route_policy.load_qk_route_policy(str(policy_path))
  route_policy.set_qk_route_policy(policy)
  try:
    report = q6_gate.build()
  finally:
    route_policy.set_qk_route_policy(None)
  assert report["generated_route_inventory"]["policy_loaded"] is True
  assert report["generated_route_inventory"]["policy_selects_direct"] is True


def test_prefill_14b_q6_decision_gate_blocks_when_no_q6_mmq_route_exists():
  route_policy.set_qk_route_policy(None)
  report = q6_gate.build()
  assert report["schema"] == "prefill_14b_q6_decision_gate.v1"
  assert report["classified_blocker"] is True
  assert report["verdict"] == "PREFILL_14B_Q6_DECISION_BLOCKED_NO_GENERATED_Q6_MMQ"
  assert report["direct_route"]["route_id"] == "prefill_q6k_direct_generated"
  assert report["generated_route_inventory"]["q6_mmq_routes_supported_by_policy"] == []
  assert report["generated_route_inventory"]["policy_loaded"] is False
  assert report["generated_route_inventory"]["policy_selects_direct"] is False
  assert report["blocker"].startswith("No generated Q6_K prefill MMQ")
