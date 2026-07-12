from extra.qk.prefill_nonlds_role_search_scope import qwen3_8b_nonlds_searches, search_command

def test_nonlds_scope_has_exact_three_roles_and_existing_knobs():
  rows = qwen3_8b_nonlds_searches()
  assert {x.role for x in rows} == {"attn_qo", "ffn_down", "attn_kv"}
  assert all(x.shape[0] == 512 and set(x.knobs) == {"UPCAST_M", "UPCAST_N", "LOCAL", "UNROLL"} for x in rows)
  assert all("prefill_v2_schedule_search.py" in search_command(x) for x in rows)
