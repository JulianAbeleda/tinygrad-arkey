import json

from extra.pure_machine_search_default_path_census import build_census
from extra.qk_route_manifest import default_purity_report, route_provenance, validate_manifest
from tinygrad.llm.model import _load_qk_route_policy, _qk_route_policy_selected, _set_qk_route_policy


def test_qk_route_manifest_purity_debt_is_explicit():
  assert validate_manifest() == []
  report = default_purity_report()
  assert report["verdict"] == "TINYGRAD_DEFAULT_PURITY_FAIL"
  assert route_provenance("decode_q4k_g3_generated") == "machine_authored_generated"
  assert route_provenance("decode_flash_block_tile_g5_konly") == "machine_authored_generated"
  assert set(report["transitional_default_routes"]) == {"decode_q6k_coop_shipped"}
  assert set(report["forbidden_default_routes"]) == {
    "decode_attention_owned_two_kernel",
    "prefill_pipe_role_selective_default",
  }


def test_default_path_census_uses_manifest_provenance():
  census = build_census()
  assert census["verdict"] == "PMS_R0_PASS_CENSUS_PINNED"
  assert census["strict_default_purity_verdict"] == "TINYGRAD_DEFAULT_PURITY_FAIL"
  by_route = {row["route_id"]: row for row in census["default_route_table"]}
  assert by_route["decode_q4k_g3_generated"]["final_default_allowed"] is True
  assert by_route["decode_flash_block_tile_g5_konly"]["final_default_allowed"] is True
  assert by_route["decode_q6k_coop_shipped"]["provenance"] == "hand_authored_uop_template"
  assert by_route["decode_attention_owned_two_kernel"]["provenance"] == "external_handwritten_kernel"
  assert by_route["prefill_pipe_role_selective_default"]["provenance"] == "external_handwritten_kernel"


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
      "route_params": {"DECODE_FLASH_BLOCK_TILE_G5": "1", "DECODE_FLASH_BLOCK_TILE_G5_KONLY": "1"},
      "rollback": {"DECODE_FLASH_BLOCK_TILE_G5": "0", "DECODE_FLASH_BLOCK_TILE_G5_KONLY": "0"},
    }],
  }))
  policy = _load_qk_route_policy(str(policy_path))
  _set_qk_route_policy(policy)
  try:
    assert _qk_route_policy_selected("decode_flash_block_tile_g5_konly", {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128})
    assert not _qk_route_policy_selected("decode_flash_block_tile_g5_konly", {"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128})
  finally:
    _set_qk_route_policy(None)
