import json

from tinygrad.llm import route_policy
from extra.qk.route_manifest import immutable_route_registry


def _policy(tmp_path):
  path = tmp_path / "roles.json"
  path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "routes": [
    {"phase": "prefill", "role": "ffn_gate_up", "quant": "Q4_K", "shape": {"rows": 12288, "cols": 4096},
     "selected_route": "prefill_q4k_int8_wmma_generated_research", "provenance": "q4_gate_up_partial_v1",
     "rollback_route": "direct_packed", "route_params": {}},
    {"phase": "prefill", "role": "ffn_down", "quant": "Q6_K", "shape": {"rows": 4096, "cols": 12288},
     "selected_route": "prefill_q6k_direct_generated", "provenance": "q6_down_partial_v1",
     "rollback_route": "direct_packed", "route_params": {}},
  ]}))
  return route_policy.load_qk_route_policy(str(path), manifest_registry=immutable_route_registry())


def test_partial_fused_candidates_are_admitted_only_for_their_role_and_shape(tmp_path):
  policy = _policy(tmp_path)
  q4 = route_policy.qk_route_policy_role_admission(
      "prefill_q4k_int8_wmma_generated_research", policy=policy, phase="prefill", role="ffn_gate_up",
      shape={"rows": 12288, "cols": 4096}, quant="Q4_K")
  assert q4["admitted"] and q4["provenance"] == "q4_gate_up_partial_v1"
  assert q4["coverage"] == "exact_structural" and q4["rollback_route"] == "direct_packed"
  assert not route_policy.qk_route_policy_role_admission(
      "prefill_q4k_int8_wmma_generated_research", policy=policy, phase="prefill", role="ffn_down",
      shape={"rows": 12288, "cols": 4096}, quant="Q4_K")["admitted"]


def test_role_admission_is_fail_closed_and_does_not_enable_defaults(tmp_path):
  policy = _policy(tmp_path)
  d = route_policy.qk_route_policy_role_admission(
      "prefill_q6k_direct_generated", policy=policy, phase="prefill", role="ffn_down",
      shape={"rows": 4096, "cols": 12288}, quant="Q4_K")
  assert not d["admitted"] and d["route"] == "direct_packed"
  assert d["provenance"] == "rollback" and d["coverage"] == "exact_structural"
  assert not route_policy.has_qk_route_policy()


def test_integrated_loop_route_is_disabled_without_real_amd_gate(tmp_path):
  path = tmp_path / "integrated-loop.json"
  path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "routes": [{
    "role": "ffn_down", "quant": "Q4_K", "shape": {"rows": 4096, "cols": 12288},
    "selected_route": "prefill_q4k_int8_wmma_tiled_research", "implementation": "integrated_loop",
    "provenance": "machine_authored_generated", "rollback_route": "direct_packed", "route_params": {},
  }]}))
  policy = route_policy.load_qk_route_policy(str(path), manifest_registry=immutable_route_registry())
  assert policy["prefill_gen"] == ()
  admission = route_policy.qk_route_policy_role_admission(
      "prefill_q4k_int8_wmma_tiled_research", policy=policy, phase="prefill", role="ffn_down",
      shape={"rows": 4096, "cols": 12288}, quant="Q4_K")
  assert not admission["admitted"]
  assert admission["route"] == "direct_packed"
  assert admission["provenance"] == "rollback"


def test_integrated_loop_requires_all_real_amd_evidence(tmp_path):
  base = {
    "role": "ffn_down", "quant": "Q4_K", "shape": {"rows": 4096, "cols": 12288},
    "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32},
    "selected_route": "prefill_q4k_int8_wmma_tiled_research", "implementation": "integrated_loop",
    "provenance": "machine_authored_generated", "rollback_route": "direct_packed", "route_params": {},
  }
  for evidence in (
    {"target": {"backend": "PYTHON", "arch": "host", "wave_size": 1}, "real_device": True, "fallback_used": False,
     "compile": {"passed": True}, "correctness": {"passed": True}, "instruction": {"passed": True}},
    {"target": base["target"], "real_device": True, "fallback_used": True,
     "compile": {"passed": True}, "correctness": {"passed": True}, "instruction": {"passed": True}},
    {"target": base["target"], "real_device": True, "fallback_used": False,
     "compile": {"passed": True}, "correctness": {"passed": True}, "instruction": {"passed": False}},
  ):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "routes": [{**base, "evidence": evidence}]}))
    assert route_policy.load_qk_route_policy(str(path), manifest_registry=immutable_route_registry())["prefill_gen"] == ()


def test_integrated_loop_evidence_binds_exact_scanned_and_candidate_target(tmp_path):
  target = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}
  row = {
    "phase": "prefill", "role": "ffn_down", "quant": "Q4_K", "shape": {"rows": 4096, "cols": 12288},
    "target": target, "candidate": {"identity": "tile_128", "target": target},
    "selected_route": "prefill_q4k_int8_wmma_tiled_research", "implementation": "integrated_loop",
    "provenance": "machine_authored_generated", "rollback_route": "direct_packed", "route_params": {},
    "evidence": {"target": target, "real_device": True, "fallback_used": False,
                 "compile": {"passed": True}, "correctness": {"passed": True}, "instruction": {"passed": True}},
  }
  path = tmp_path / "bound-gate.json"
  path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "routes": [row]}))
  assert len(route_policy.load_qk_route_policy(str(path), manifest_registry=immutable_route_registry())["prefill_gen"]) == 1


def test_legacy_route_id_artifact_loads_under_canonical_policy_key(tmp_path):
  legacy = "decode_flash_live_split_g4_8b_kvboth"
  canonical = "decode_flash_live_split_g4_kvboth"
  path = tmp_path / "legacy.json"
  path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "routes": [{
    "selected_route": legacy, "shape": {"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128},
    "route_params": {"DECODE_LIVE_SPLIT": "1"},
  }]}))
  policy = route_policy.load_qk_route_policy(str(path), manifest_registry=immutable_route_registry())
  assert set(policy["selected"]) == {canonical}
  assert policy["selected"][canonical]["legacy_selected_route"] == legacy
  assert route_policy.qk_route_policy_selected(legacy, {"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128}, policy=policy)


def test_profile_and_model_filename_are_provenance_only(tmp_path):
  route = {"phase": "prefill", "role": "ffn_gate_up", "quant": "Q4_K",
           "shape": {"rows": 12288, "cols": 4096},
           "selected_route": "prefill_q4k_int8_wmma_generated_research", "route_params": {},
           "rollback_route": "direct_packed"}
  policies = []
  for filename, profile in (("known-14b.json", "qwen3_14b_q4k_m_gfx1100"),
                            ("completely-unknown-model-name.gguf.policy.json", "renamed_evidence_fixture")):
    path = tmp_path / filename
    path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "profile": profile,
                                "model_filename": filename, "routes": [route]}))
    policies.append(route_policy.load_qk_route_policy(str(path), manifest_registry=immutable_route_registry()))
  assert policies[0]["provenance"]["profile"] != policies[1]["provenance"]["profile"]
  facts = dict(phase="prefill", role="ffn_gate_up", quant="Q4_K", shape={"rows": 12288, "cols": 4096})
  assert all(route_policy.qk_route_policy_role_admission(
    "prefill_q4k_int8_wmma_generated_research", policy=policy, **facts)["admitted"] for policy in policies)


def test_partial_shape_is_rejected_with_explicit_rollback(tmp_path):
  admission = route_policy.qk_route_policy_role_admission(
    "prefill_q4k_int8_wmma_generated_research", policy=_policy(tmp_path), phase="prefill",
    role="ffn_gate_up", quant="Q4_K", shape={"rows": 12288})
  assert not admission["admitted"] and admission["route"] == "direct_packed"
  assert any("partial or non-canonical" in reason for reason in admission["errors"])


def test_exact_structural_binding_includes_present_identities(tmp_path):
  path = tmp_path / "identity.json"
  path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "routes": [{
    "phase": "prefill", "role": "ffn_down", "quant": "Q4_K", "shape": {"rows": 4096, "cols": 12288},
    "target": {"backend": "AMD", "arch": "gfx1100"}, "capability": "wmma_i8_v1", "candidate": "tile_128",
    "selected_route": "prefill_q4k_int8_wmma_tiled_research", "route_params": {}, "rollback_route": "direct_packed",
  }]}))
  policy = route_policy.load_qk_route_policy(str(path), manifest_registry=immutable_route_registry())
  facts = dict(phase="prefill", role="ffn_down", quant="Q4_K", shape={"rows": 4096, "cols": 12288},
               target={"backend": "AMD", "arch": "gfx1100"}, capability="wmma_i8_v1", candidate="tile_128")
  assert route_policy.qk_route_policy_role_admission(
    "prefill_q4k_int8_wmma_tiled_research", policy=policy, **facts)["admitted"]
  facts["candidate"] = "tile_64"
  mismatch = route_policy.qk_route_policy_role_admission(
    "prefill_q4k_int8_wmma_tiled_research", policy=policy, **facts)
  assert not mismatch["admitted"] and mismatch["route"] == "direct_packed"
