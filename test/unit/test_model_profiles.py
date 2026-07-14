from types import SimpleNamespace

import pytest

from extra.qk.model_profiles import (
  QWEN3_14B_Q4_K_M_GFX1100,
  QWEN3_8B_Q4_K_M_GFX1100,
  AttentionShape,
  LinearRoleShape,
  attention_shape,
  prefill_role_shapes,
  profile_by_id,
  profile_from_model_path,
  profile_from_transformer_config,
)


def test_qwen3_8b_q4k_profile_carries_prefill_role_shapes():
  profile = QWEN3_8B_Q4_K_M_GFX1100
  assert profile.id == "qwen3_8b_q4k_m_gfx1100"
  assert [(shape.role, shape.M, shape.N, shape.K) for shape in prefill_role_shapes(profile)] == [
    ("attn_kv", 512, 1024, 4096),
    ("attn_qo", 512, 4096, 4096),
    ("ffn_down", 512, 4096, 12288),
    ("ffn_gate_up", 512, 12288, 4096),
  ]
  assert attention_shape(profile) == AttentionShape(B=1, Hq=32, Hkv=8, Hd=128)
  assert profile.role_shape("ffn_gate_up").mnk == (512, 12288, 4096)


def test_qwen3_14b_q4k_profile_carries_tiled_wmma_role_shapes():
  profile = QWEN3_14B_Q4_K_M_GFX1100
  assert [(shape.role, shape.M, shape.N, shape.K) for shape in prefill_role_shapes(profile)] == [
    ("attn_kv", 512, 1024, 5120),
    ("attn_qo", 512, 5120, 5120),
    ("ffn_down", 512, 5120, 17408),
    ("ffn_gate_up", 512, 17408, 5120),
  ]
  assert attention_shape(profile) == AttentionShape(B=1, Hq=40, Hkv=8, Hd=128)
  assert profile.role_shape("attn_qo").tensor_patterns == ("blk.*.attn_q.weight", "blk.*.attn_output.weight")


def test_profile_lookup_by_id_and_transformer_config():
  assert profile_by_id("14b") is QWEN3_14B_Q4_K_M_GFX1100
  assert profile_by_id("qwen3_14b_q4k_m_gfx1100") is QWEN3_14B_Q4_K_M_GFX1100
  assert profile_by_id("qwen3_14b_q4_k_m_gfx1100") is QWEN3_14B_Q4_K_M_GFX1100
  config = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
  }
  assert profile_from_transformer_config(config, quant="Q4_K_M", device_profile="gfx1100") is QWEN3_8B_Q4_K_M_GFX1100

  obj_config = SimpleNamespace(model_type="qwen3", hidden_size=5120, intermediate_size=17408,
                               num_attention_heads=40, num_key_value_heads=8, head_dim=128)
  assert profile_from_transformer_config(obj_config, quant="Q4_K_M",
                                         device_profile="gfx1100") is QWEN3_14B_Q4_K_M_GFX1100
  local_config = SimpleNamespace(family="qwen3", dim=4096, hidden_dim=12288, n_heads=32, n_kv_heads=8, head_dim=128)
  assert profile_from_transformer_config(local_config, quant="Q4_K_M",
                                         device_profile="gfx1100") is QWEN3_8B_Q4_K_M_GFX1100


def test_profile_lookup_from_model_path_uses_central_model_facts():
  assert profile_from_model_path("/models/Qwen3-14B-Q4_K_M.gguf") is QWEN3_14B_Q4_K_M_GFX1100
  assert profile_from_model_path("/models/Qwen3-8B-Q4_K_M.gguf") is QWEN3_8B_Q4_K_M_GFX1100
  with pytest.raises(KeyError, match="no model profile"):
    profile_from_model_path("/models/Qwen3-32B-Q4_K_M.gguf")


def test_profile_lookup_blocks_unknown_shape_and_role():
  with pytest.raises(KeyError, match="no model profile"):
    profile_from_transformer_config({"model_type": "qwen3", "hidden_size": 8192, "intermediate_size": 28672,
                                     "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128},
                                    quant="Q4_K_M", device_profile="gfx1100")
  with pytest.raises(KeyError, match="no prefill role"):
    QWEN3_8B_Q4_K_M_GFX1100.role_shape("lm_head")


def test_profile_rows_are_json_serializable_data_only():
  row = QWEN3_14B_Q4_K_M_GFX1100.to_json()
  assert row["roles"][-1] == {
    "role": "ffn_gate_up",
    "phase": "prefill",
    "quant": "Q4_K_M",
    "M": 512,
    "N": 17408,
    "K": 5120,
    "tensor_patterns": ["blk.*.ffn_gate.weight", "blk.*.ffn_up.weight"],
  }
  assert LinearRoleShape("attn_kv", "prefill", "Q4_K_M", 512, 1024, 5120).to_json()["N"] == 1024
