import json, subprocess, sys

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
  assert report["verdict"] == "TINYGRAD_DEFAULT_PURITY_PASS"
  assert route_provenance("decode_q4k_g3_generated") == "machine_authored_generated"
  assert route_provenance("decode_flash_block_tile_g5_konly") == "machine_authored_generated"
  assert route_provenance("decode_flash_live_split_g4_8b_kvboth") == "machine_authored_generated"
  # TG-P3: Q6_K default is now the generated route; no manifest hand-kernel rollback remains.
  assert route_provenance("decode_q6k_coop_generated") == "machine_authored_generated"
  assert route_provenance("prefill_q6k_direct_generated") == "machine_authored_generated"
  assert route_provenance("prefill_pipe_role_selective_generated") == "external_handwritten_kernel"
  assert route_provenance("prefill_wmma_pipe_primitive_generated") == "tinygrad_scheduler_generated"
  assert route_provenance("prefill_wmma_pipe_lds_dbuf_primitive_generated") == "compiler_primitive_spec_owned"
  assert route_provenance("prefill_v2_scheduler_matmul_default") == "tinygrad_scheduler_generated"
  assert set(report["transitional_default_routes"]) == set()
  assert set(report["forbidden_default_routes"]) == set()


def test_prefill_hybrid_and_pure_route_identities():
  # Ground-truth mapping (verified by running the model 2026-07-10): the HYBRID is the 4k route that keeps the fast
  # hand backend atom, selected by PREFILL_GRAPH_GEMM=1 with NO primitive flags. The PURE machine-search transport is
  # the ~1.3k route selected by adding the primitive flags. Do not invert these.
  from extra.qk.route_manifest import route_env
  HYBRID = "prefill_pipe_role_selective_generated"          # ~4413 pp512, fast hand backend atom + machine schedule
  PURE = "prefill_wmma_pipe_lds_dbuf_primitive_generated"   # ~1332 pp512, fully generated transport
  hybrid_env = route_env(HYBRID)
  pure_env = route_env(PURE)
  assert hybrid_env == {"PREFILL_GRAPH_GEMM": "1"}, "hybrid selects on GRAPH_GEMM alone (no primitive flags)"
  # The pure transport turns ON the primitive flags that the hybrid must NOT set.
  assert set(hybrid_env.items()) < set(pure_env.items())
  assert {"PREFILL_WMMA_PIPE_PRIMITIVE", "PREFILL_WMMA_LDS_PRIMITIVE", "PREFILL_DBUF"} <= set(pure_env)


def test_14b_hybrid_mmq_atom_manifest_is_research_only_and_ordered_for_m8():
  from extra.qk.route_manifest import ROUTES, default_routes, route_env, rollback_env
  route_id = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
  row = ROUTES[route_id]

  assert route_id not in default_routes()
  assert row["status"] == "research"
  assert row["roles"] == ["ffn_gate_up"]
  assert set(row["excluded_roles"]) == {"attn_qo", "attn_kv", "ffn_down"}
  assert row["route_classification"] == "compiler_primitive_spec_owned__hand_mmq_backend_atom"
  assert row["baseline_route_id"] == "prefill_q4k_direct_tile4x4_default"
  assert route_env(route_id) == {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "1", "PREFILL_ROUTE_STRICT": "1"}
  assert rollback_env(route_id) == {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "0", "PREFILL_ROUTE": "direct_packed"}
  assert [(item["role"], item["quant"]) for item in row["m8_expansion_order"]] == [
    ("ffn_gate_up", "Q4_K"),
    ("attn_qo", "Q4_K"),
    ("attn_kv", "Q4_K"),
    ("ffn_down", "Q6_K"),
  ]


def test_default_path_census_uses_manifest_provenance():
  census = build_census()
  assert census["verdict"] == "PMS_R0_PASS_CENSUS_PINNED"
  assert census["strict_default_purity_verdict"] == "TINYGRAD_DEFAULT_PURITY_PASS"
  assert census["missing_from_census"] == []
  assert all(row["in_manifest"] for row in census["rows"])
  by_route = {row["route_id"]: row for row in census["default_route_table"]}
  assert by_route["decode_q4k_g3_generated"]["final_default_allowed"] is True
  assert by_route["decode_flash_block_tile_g5_konly"]["final_default_allowed"] is True
  assert by_route["decode_flash_live_split_g4_8b_kvboth"]["final_default_allowed"] is True
  # TG-P3: the generated Q6_K route is the default and the census only carries current manifest route rows.
  assert by_route["decode_q6k_coop_generated"]["provenance"] == "machine_authored_generated"
  assert by_route["decode_q6k_coop_generated"]["final_default_allowed"] is True
  assert by_route["prefill_q6k_direct_generated"]["provenance"] == "machine_authored_generated"
  assert by_route["prefill_q6k_direct_generated"]["final_default_allowed"] is True
  assert by_route["prefill_v2_scheduler_matmul_default"]["provenance"] == "tinygrad_scheduler_generated"
  assert by_route["prefill_v2_scheduler_matmul_default"]["final_default_allowed"] is True
  assert "prefill_pipe_role_selective_generated" not in by_route
  assert "prefill_pipe_role_selective_default" not in by_route
  assert by_route["prefill_q4k_direct_tile4x4_default"]["provenance"] == "machine_authored_generated"
  assert by_route["prefill_q4k_direct_tile4x4_default"]["final_default_allowed"] is True


def test_route_policy_import_does_not_eagerly_import_qk_manifest():
  code = "import sys; import tinygrad.llm.route_policy; print(any(k.startswith('extra.qk') for k in sys.modules))"
  out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
  assert out == "False"


def test_qk_route_policy_supported_ids_include_manifest_defaults():
  from extra.qk.route_manifest import default_routes
  from tinygrad.llm import route_policy
  assert set(default_routes()) <= route_policy._supported_qk_route_ids()


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


def test_qk_route_policy_accepts_pure_prefill_scheduler_route(tmp_path):
  policy_path = tmp_path / "prefill_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "model_id": "qwen8ish",
    "architecture_class": "dense_decoder",
    "authorized": True,
    "routes": [{
      "role": "ffn_down",
      "shape": {"rows": 4096, "cols": 12288},
      "quant": "Q4_K",
      "selected_route": "prefill_v2_scheduler_matmul_default",
      "status": "promoted",
      "route_params": {},
      "rollback": {"PREFILL_GRAPH_GEMM": "1"},
    }],
  }))
  policy = _load_qk_route_policy(str(policy_path))
  assert policy["prefill_gen"][0]["selected_route"] == "prefill_v2_scheduler_matmul_default"


def test_qk_route_policy_rejects_raw_prefill_graph_gemm_route(tmp_path):
  policy_path = tmp_path / "raw_prefill_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "routes": [{
      "selected_route": "prefill_pipe_role_selective_generated",
      "shape": {"rows": 4096, "cols": 12288},
    }],
  }))
  with pytest.raises(ValueError, match="unsupported route"):
    _load_qk_route_policy(str(policy_path))


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
