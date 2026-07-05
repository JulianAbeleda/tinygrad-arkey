import json

import pytest

from extra.audit.pure_machine_search_default_path_census import build_census
from extra.qk.route_manifest import default_purity_report, route_provenance, validate_manifest
from tinygrad.llm.model import (_load_qk_route_policy, _qk_route_policy_selected, _set_qk_route_policy,
                                _qk_route_policy_selects_q4k_g3, _qk_route_policy_selects_q6k_generated)


def _write_q4k_g3_policy(tmp_path, rows):
  policy_path = tmp_path / "route_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "model_id": "qwen8g3",
    "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [dict({
      "role": role, "shape": {"rows": rows_n, "cols": cols_k}, "quant": "Q4_K",
      "selected_route": "decode_q4k_g3_generated", "status": "promoted",
      "provenance": "machine_authored_generated", "route_family": "q4k_g3_lanemap",
      "route_params": {"BUBBLEBEAM_FUTURESIGHT": "1"}, "rollback": {"BUBBLEBEAM_FUTURESIGHT": "0"},
    }) for role, rows_n, cols_k in rows],
  }))
  return policy_path


def test_qk_route_manifest_purity_debt_is_explicit():
  assert validate_manifest() == []
  report = default_purity_report()
  assert report["verdict"] == "TINYGRAD_DEFAULT_PURITY_FAIL"
  assert route_provenance("decode_q4k_g3_generated") == "machine_authored_generated"
  assert route_provenance("decode_flash_block_tile_g5_konly") == "machine_authored_generated"
  assert route_provenance("decode_flash_live_split_g4_8b_kvboth") == "machine_authored_generated"
  # TG-P3: Q6_K default is now the generated route; the hand template is rollback-only, no longer transitional debt.
  assert route_provenance("decode_q6k_coop_generated") == "machine_authored_generated"
  assert route_provenance("decode_q6k_coop_shipped") == "rollback_oracle"
  # TG-P4: prefill default is now the spec-generated schedule; the legacy fixed emit rollback is removed.
  assert route_provenance("prefill_pipe_role_selective_generated") == "machine_authored_generated"
  assert set(report["transitional_default_routes"]) == {"prefill_q4k_direct_tile4x4_default"}
  assert set(report["forbidden_default_routes"]) == set()


def test_default_path_census_uses_manifest_provenance():
  census = build_census()
  assert census["verdict"] == "PMS_R0_PASS_CENSUS_PINNED"
  assert census["strict_default_purity_verdict"] == "TINYGRAD_DEFAULT_PURITY_FAIL"
  by_route = {row["route_id"]: row for row in census["default_route_table"]}
  assert by_route["decode_q4k_g3_generated"]["final_default_allowed"] is True
  assert by_route["decode_flash_block_tile_g5_konly"]["final_default_allowed"] is True
  assert by_route["decode_flash_live_split_g4_8b_kvboth"]["final_default_allowed"] is True
  # TG-P3: the generated Q6_K route is the default; the hand template is no longer on the default path.
  assert by_route["decode_q6k_coop_generated"]["provenance"] == "machine_authored_generated"
  assert by_route["decode_q6k_coop_generated"]["final_default_allowed"] is True
  assert "decode_q6k_coop_shipped" not in by_route
  assert "decode_attention_owned_two_kernel" not in by_route
  # TG-P4: generated prefill schedule is the default; legacy fixed emit is off the default path.
  assert by_route["prefill_pipe_role_selective_generated"]["provenance"] == "machine_authored_generated"
  assert by_route["prefill_pipe_role_selective_generated"]["final_default_allowed"] is True
  assert "prefill_pipe_role_selective_default" not in by_route
  assert by_route["prefill_q4k_direct_tile4x4_default"]["provenance"] == "hand_authored_uop_template"
  assert by_route["prefill_q4k_direct_tile4x4_default"]["final_default_allowed"] is False


def test_qk_route_policy_selects_g5_by_shape(tmp_path):
  policy_path = tmp_path / "route_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "model_id": "qwen14ish",
    "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [{
      "role": "flash_attention_v_access",
      "shape": {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128},
      "quant": "Q4_K",
      "selected_route": "decode_flash_block_tile_g5_konly",
      "status": "promoted",
      "route_params": {"DECODE_LIVE_SPLIT": "1"},
      "rollback": {"DECODE_LIVE_SPLIT": "0"},
    }],
  }))
  policy = _load_qk_route_policy(str(policy_path))
  _set_qk_route_policy(policy)
  try:
    assert _qk_route_policy_selected("decode_flash_block_tile_g5_konly", {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128})
    assert not _qk_route_policy_selected("decode_flash_block_tile_g5_konly", {"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128})
  finally:
    _set_qk_route_policy(None)


def test_qk_route_policy_selects_8b_live_split_by_shape(tmp_path):
  policy_path = tmp_path / "route_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "model_id": "qwen8ish",
    "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [{
      "role": "flash_attention_v_access",
      "shape": {"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128},
      "quant": "fp16",
      "selected_route": "decode_flash_live_split_g4_8b_kvboth",
      "status": "promoted",
      "route_params": {"DECODE_LIVE_SPLIT": "1"},
      "rollback": {"DECODE_LIVE_SPLIT": "0"},
    }],
  }))
  policy = _load_qk_route_policy(str(policy_path))
  _set_qk_route_policy(policy)
  try:
    assert _qk_route_policy_selected("decode_flash_live_split_g4_8b_kvboth", {"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128})
    assert not _qk_route_policy_selected("decode_flash_live_split_g4_8b_kvboth", {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128})
  finally:
    _set_qk_route_policy(None)


def test_qk_route_policy_selects_q4k_g3_per_tensor(tmp_path):
  policy_path = _write_q4k_g3_policy(tmp_path, [
    ("ffn_gate_up", 12288, 4096), ("ffn_down", 4096, 12288), ("attn_qo", 4096, 4096)])
  policy = _load_qk_route_policy(str(policy_path))
  assert len(policy["q4k_g3"]) == 3
  assert policy["selected"]["decode_q4k_g3_generated"]["selected_route"] == "decode_q4k_g3_generated"
  _set_qk_route_policy(policy)
  try:
    # per-tensor: each selected tensor's real GEMV dims (out=rows, in=cols) bind G3
    assert _qk_route_policy_selects_q4k_g3(12288, 4096)   # ffn_gate_up
    assert _qk_route_policy_selects_q4k_g3(4096, 12288)   # ffn_down
    assert _qk_route_policy_selects_q4k_g3(4096, 4096)    # attn_qo
    # a shape the policy did NOT select is not bound (e.g. attn_kv 1024x4096)
    assert not _qk_route_policy_selects_q4k_g3(1024, 4096)
  finally:
    _set_qk_route_policy(None)


def test_qk_route_policy_selects_nothing_when_absent():
  _set_qk_route_policy(None)
  assert not _qk_route_policy_selects_q4k_g3(12288, 4096)
  assert not _qk_route_policy_selects_q6k_generated(4096, 12288)


def _write_q6k_gen_policy(tmp_path, rows):
  policy_path = tmp_path / "q6k_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1", "model_id": "qwen8q6", "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [{"role": role, "shape": {"rows": rows_n, "cols": cols_k}, "quant": "Q6_K",
                "selected_route": "decode_q6k_coop_generated", "status": "promoted",
                "provenance": "machine_authored_generated", "route_family": "q6k_route",
                "route_params": {"DECODE_Q6K_GENERATED": "1"}, "rollback": {"DECODE_Q6K_GENERATED": "0"}}
               for role, rows_n, cols_k in rows]}))
  return policy_path


def test_qk_route_policy_selects_q6k_generated_per_tensor(tmp_path):
  policy_path = _write_q6k_gen_policy(tmp_path, [("ffn_down", 4096, 12288), ("lm_head", 151936, 4096)])
  policy = _load_qk_route_policy(str(policy_path))
  assert len(policy["q6k_gen"]) == 2
  _set_qk_route_policy(policy)
  try:
    assert _qk_route_policy_selects_q6k_generated(4096, 12288)     # ffn_down
    assert _qk_route_policy_selects_q6k_generated(151936, 4096)    # lm_head
    assert not _qk_route_policy_selects_q6k_generated(1024, 4096)  # not selected
    assert not _qk_route_policy_selects_q4k_g3(4096, 12288)        # G3 helper stays independent
  finally:
    _set_qk_route_policy(None)


def test_qk_route_policy_rejects_unsupported_q6k_params(tmp_path):
  policy_path = tmp_path / "bad_q6k.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1", "routes": [{
      "selected_route": "decode_q6k_coop_generated", "shape": {"rows": 4096, "cols": 12288},
      "route_params": {"BUBBLEBEAM_FUTURESIGHT": "1"}}]}))
  with pytest.raises(ValueError, match="unsupported params"):
    _load_qk_route_policy(str(policy_path))


def test_qk_route_policy_rejects_unsupported_g3_params(tmp_path):
  policy_path = tmp_path / "bad_params.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1", "routes": [{
      "selected_route": "decode_q4k_g3_generated", "shape": {"rows": 4096, "cols": 4096},
      "route_params": {"DECODE_Q4K_G3_ANYSHAPE": "1"}}]}))
  with pytest.raises(ValueError, match="unsupported params"):
    _load_qk_route_policy(str(policy_path))


def test_qk_route_policy_rejects_malformed_g3_shape(tmp_path):
  policy_path = tmp_path / "bad_shape.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1", "routes": [{
      "selected_route": "decode_q4k_g3_generated", "shape": {"Hq": 40, "Hkv": 8},
      "route_params": {"BUBBLEBEAM_FUTURESIGHT": "1"}}]}))
  with pytest.raises(ValueError, match="malformed shape"):
    _load_qk_route_policy(str(policy_path))


def test_qk_route_policy_rejects_unsupported_route_id(tmp_path):
  policy_path = tmp_path / "bad_route.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1", "routes": [{
      "selected_route": "decode_q4k_nonexistent", "shape": {"rows": 4096, "cols": 4096}}]}))
  with pytest.raises(ValueError, match="unsupported route"):
    _load_qk_route_policy(str(policy_path))
