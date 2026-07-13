from extra.qk.prefill_graph_gemm_route import candidate_set_role_enabled

def test_promoted_candidate_set_defaults_to_all_dense_roles():
  assert all(candidate_set_role_enabled(role, {}) for role in ("attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"))

def test_candidate_set_roles_are_explicit_and_reversible():
  env = {"BOLTBEAM_FULL_KERNEL_CANDIDATE_ROLES": "attn_qo,ffn_down"}
  assert candidate_set_role_enabled("attn_qo", env)
  assert candidate_set_role_enabled("ffn_down", env)
  assert not candidate_set_role_enabled("ffn_gate_up", env)
