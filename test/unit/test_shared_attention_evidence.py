from extra.qk.model_profiles import MODEL_PROFILES
from extra.qk.shared_attention_evidence import (DEFAULT_CONTEXTS, attention_workloads, authority_command,
  dual_wmma_fused_call_report, dual_wmma_fused_call_fixture, fused_wmma_role_report, geometry_candidates)
from pathlib import Path


def test_shared_attention_workloads_cover_both_real_routes_with_one_schema():
  rows = attention_workloads()
  assert {(row.profile_id, row.T, row.KV) for row in rows} == {
    (profile.id, context, context) for profile in MODEL_PROFILES for context in DEFAULT_CONTEXTS}
  assert {(row.Hq, row.Hkv, row.G, row.Hd, row.activation_dtype) for row in rows} == {
    (32, 8, 4, 128, "float16"), (40, 8, 5, 128, "float16")}


def test_geometry_domain_is_shared_and_not_a_route_selector():
  by_profile = {profile.id: [row for row in attention_workloads(contexts=(2048,)) if row.profile_id == profile.id][0]
                for profile in MODEL_PROFILES}
  ids = {profile_id: tuple(candidate.candidate_id for candidate in geometry_candidates(row))
         for profile_id, row in by_profile.items()}
  assert len(set(ids.values())) == 1
  assert ids["qwen3_8b_q4k_m_gfx1100"]


def test_both_routes_use_the_same_pinned_whole_prefill_authority_harness():
  for profile in MODEL_PROFILES:
    argv = authority_command(profile, artifact_path=f"bench/shared-flash/{profile.id}.json")
    assert argv[:3] == ["extra/qk/prefill_whole_synced.py", "--model", argv[2]]
    assert "--model-profile" in argv and profile.id in argv
    assert "--pin-clock" in argv
    assert "--artifact" in argv


def test_model_prefill_has_one_shared_attention_boundary_for_both_weight_routes():
  """The 8B overlay and 14B packed projections must not grow separate attention kernels."""
  source = (Path(__file__).parents[2] / "tinygrad/llm/model.py").read_text()
  assert source.count("from tinygrad.llm.flash_prefill_attention import shared_prefill_attention") == 1
  assert source.count("shared_prefill_attention(q, k, v, mask=mask)") == 1
  # The shared call is downstream of both projection branches and retains the
  # model-owned mask; this guards accidental route-specific attention forks.
  assert "self.config.prefill_tc_attn" in source
  assert "q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)" in source


def test_real_model_context_domain_and_gqa_are_explicitly_represented():
  rows = attention_workloads()
  assert {row.T for row in rows} == {512, 2048, 4096}
  assert {row.KV for row in rows} == {512, 2048, 4096}
  for row in rows:
    assert row.Hq % row.Hkv == 0
    assert row.G == row.Hq // row.Hkv
    assert row.Hd == 128
    assert row.causal

def test_dual_wmma_report_requires_qk_and_pv_in_one_call():
  assert fused_wmma_role_report("CALL fused\n// QK WMMA\n// PV WMMA")["promotable"]

def test_dual_wmma_report_fails_closed_for_missing_role_or_multiple_calls():
  assert not fused_wmma_role_report("CALL fused\n// QK WMMA")["promotable"]
  assert not fused_wmma_role_report("CALL qk\n// QK WMMA\nCALL pv\n// PV WMMA")["promotable"]

def test_strict_dual_wmma_gate_requires_one_call_shaped_fragments_and_residency():
  source = "CALL fused_attention\n// QK WMMA\n// PV WMMA\nSHAPED_WMMA(TILE_GATHER)"
  report = dual_wmma_fused_call_report(source, ((32, 128), (32, 128)))
  assert report["promotable"]
  assert report["full_score_probability_buffers"] is False

def test_strict_dual_wmma_gate_fails_closed_for_unshaped_or_materialized_graphs():
  assert not dual_wmma_fused_call_report("CALL fused\n// QK WMMA\n// PV WMMA")["promotable"]
  assert not dual_wmma_fused_call_report("CALL fused\n// QK WMMA\n// PV WMMA\nSHAPED_WMMA", ((1, 8, 64, 64),))["promotable"]
  assert not dual_wmma_fused_call_report("CALL qk\n// QK WMMA\nCALL pv\n// PV WMMA\nSHAPED_WMMA")["promotable"]

def test_dual_wmma_fused_call_fixture_requires_real_isa_capture():
  fixture = dual_wmma_fused_call_fixture()
  assert fixture["qk_wmma"] and fixture["pv_wmma"]
  assert fixture["single_call"] and fixture["shaped_fragments"]
  assert fixture["isa_captured"] is False
  assert fixture["promotable"] is False

def test_dual_wmma_fused_call_report_accepts_one_call_source_and_isa_fixture():
  fixture = dual_wmma_fused_call_fixture(isa="QK v_wmma_f32_16x16x16_f16\nPV v_wmma_f32_16x16x16_f16")
  assert fixture["isa_captured"] is True
  assert fixture["qk_isa_wmma_instructions"] == fixture["pv_isa_wmma_instructions"] == 1
  assert fixture["promotable"] is True

def test_dual_wmma_report_does_not_credit_unattributed_isa_or_source_lines():
  fixture = dual_wmma_fused_call_fixture(isa="v_wmma_f32_16x16x16_f16")
  assert fixture["isa_captured"] is True
  assert fixture["role_attributed_isa"] is False
  assert fixture["promotable"] is False
  report = dual_wmma_fused_call_report("CALL fused\n// WMMA helper\nSHAPED_WMMA")
  assert report["qk_source_wmma_lines"] == report["pv_source_wmma_lines"] == 0
  assert report["promotable"] is False
