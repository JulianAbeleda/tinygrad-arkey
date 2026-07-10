from extra.qk.prefill import s10_hybrid_role_trace as trace


def test_s10_hybrid_role_trace_maps_roles_to_s9_backend_atoms():
  report = trace.build_trace()

  assert report["schema"] == "prefill-s10-hybrid-s9-s10-role-trace.v1"
  assert report["profile"]["id"] == "qwen3_8b_q4k_m_gfx1100"
  assert report["roles"] == ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"]
  assert report["env"] == {"PREFILL_V2": "1", "PREFILL_GRAPH_GEMM": "1"}
  assert report["classification"] == trace.HYBRID_CLASSIFICATION
  assert set(report["forbidden_env"]) == {"PREFILL_WMMA_PIPE_PRIMITIVE", "PREFILL_WMMA_LDS_PRIMITIVE", "PREFILL_DBUF"}
  assert report["acceptance_gate"]["pp512_min_tok_s"] == 4000
  assert report["acceptance_gate"]["primitive_flags_allowed"] is False

  rows = {row["role"]: row for row in report["rows"]}
  assert set(rows) == {"attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"}

  for role in ("attn_qo", "attn_kv", "ffn_down"):
    assert rows[role]["route_family"] == "pipe"
    assert rows[role]["backend_atom"] == "build_gemm_pipe"
    assert rows[role]["classification"] == trace.HYBRID_CLASSIFICATION
    assert "lds_spec_summary" not in rows[role]
    assert "hand_coded_epoch_primitive" not in rows[role]

  gate = rows["ffn_gate_up"]
  assert gate["route_family"] == "lds"
  assert gate["backend_atom"] == "lower_lds2_gemm_kernel/build_gemm_lds2"
  assert gate["lds_spec_summary"]["ownership_classification"] == "compiler_primitive_spec_owned__asm_backend_atom"
  assert gate["lds_spec_summary"]["selection_label"] == "S9_COMPLETE_KEEP_OPT_IN"
  assert gate["lds_spec_summary"]["legality_errors"] == []
  assert gate["hand_coded_epoch_primitive"]["classification"] == "hand_coded_dbuf_epoch_primitive"
  assert gate["hand_coded_epoch_primitive"]["slot_expr"] == "epoch % 2"
  assert gate["hand_coded_epoch_primitive"]["reusable_contract"] == "parameterized_by_role_tile_layout_wait_policy"


def test_s10_hybrid_role_trace_accepts_profile_and_role_selection():
  report = trace.build_trace(profile="qwen3_14b_q4k_m_gfx1100", roles=("ffn_gate_up",))

  assert report["profile"]["id"] == "qwen3_14b_q4k_m_gfx1100"
  assert report["profile"]["size_label"] == "14B"
  assert report["roles"] == ["ffn_gate_up"]
  assert len(report["rows"]) == 1
  assert report["rows"][0]["role"] == "ffn_gate_up"
  assert report["rows"][0]["shape"] == {"m": 512, "n": 17408, "k": 5120}


def test_s10_hybrid_role_trace_writes_artifact(tmp_path):
  out = tmp_path / "trace.json"
  report = trace.main(["--output", str(out), "--profile", "qwen3_14b_q4k_m_gfx1100", "--role", "attn_qo"])

  assert out.exists()
  assert report["rows"][0]["role"] == "attn_qo"
  assert report["rows"][0]["shape"] == {"m": 512, "n": 5120, "k": 5120}
  assert "PREFILL_GRAPH_GEMM=1" in report["acceptance_gate"]["authority_command"]
