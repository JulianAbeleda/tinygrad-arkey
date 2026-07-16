import pytest
from extra.qk.role_shape_route_validation import run

def test_rejects_wrong_model_artifacts():
  with pytest.raises(ValueError): run("/tmp/Qwen3-8B-Q4_K_M.gguf")

def test_has_exact_four_roles_and_route_contract(monkeypatch):
  monkeypatch.setattr("extra.qk.role_shape_route_validation._one", lambda shape, role, pp, seed: {"role": role, "shape": {}, "modes": {"wmma_tiled": {"status":"PASS"}, "direct_packed": {"status":"PASS"}}})
  out = run("/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
  assert out["route"] == "prefill_q4k_int8_wmma_tiled_research"
  assert [r["role"] for r in out["rows"]] == ["attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"]
