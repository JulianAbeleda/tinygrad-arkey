import pytest
from extra.qk.role_baseline_14b import identity, measure

def test_identity_is_role_shape_and_mode_specific():
  a = identity("ffn_down", 512, 5120, 6144, "direct_packed", 512)
  b = identity("ffn_down", 512, 5120, 6144, "wmma_tiled", 512)
  assert a != b and a.startswith("generated:14b-q4km:ffn_down:direct_packed:")

def test_rejects_unrelated_artifacts():
  with pytest.raises(ValueError): measure("/tmp/Qwen3-8B-Q4_K_M.gguf")
  with pytest.raises(ValueError): measure("/tmp/Qwen3-14B-Q4_K_M-q4_k_gemv.gguf")

def test_contract_has_explicit_roles_shapes_and_both_modes():
  out = measure("/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
  assert out["pp"] == 512
  assert {(r["role"], r["mode"]) for r in out["rows"]} == {
    (role, mode) for role in ("attn_qo", "attn_kv", "ffn_gate_up", "ffn_down")
    for mode in ("direct_packed", "wmma_tiled")}
  assert all(all(r[k] > 0 for k in ("M", "N", "K")) for r in out["rows"])
