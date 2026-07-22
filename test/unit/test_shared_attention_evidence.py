from extra.qk.model_profiles import MODEL_PROFILES
from extra.qk.shared_attention_evidence import DEFAULT_CONTEXTS, attention_workloads, authority_command, geometry_candidates


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
