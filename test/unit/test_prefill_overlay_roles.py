import pathlib

import pytest

from tinygrad.llm.gguf import gguf_load_metadata
from tinygrad.llm.model import Transformer, derive_selected_gguf_prefill_inventory
from tinygrad.llm.model_facts import (
  PREFILL_OVERLAY_LINEAR_NAMES, PREFILL_OVERLAY_ROLES, estimate_prefill_overlay_bytes, is_prefill_overlay_role,
)


def test_overlay_role_set_is_canonical_and_covers_every_walk_name():
  assert PREFILL_OVERLAY_ROLES == frozenset({"ffn_gate_up", "ffn_down", "attn_qo", "attn_kv"})
  assert {is_prefill_overlay_role(name) for name in PREFILL_OVERLAY_LINEAR_NAMES} == {True}
  # MoE expert weights, norms, and lm_head are not overlay-covered by the exact-leaf alias table.
  assert not is_prefill_overlay_role("blk.0.ffn_gate_exps.weight")
  assert not is_prefill_overlay_role("blk.0.attn_q_norm.weight")
  assert not is_prefill_overlay_role("output.weight")
  assert is_prefill_overlay_role("blk.0.ffn_gate.weight")
  assert is_prefill_overlay_role("blk.3.ffn_gate_shexp.weight")
  assert is_prefill_overlay_role("attn_qo")


def test_state_dict_byte_helper_sums_covered_numels_only():
  pairs = (("blk.0.ffn_gate.weight", 100), ("blk.0.ffn_down.weight", 200),
           ("blk.0.attn_q.weight", 300), ("blk.0.ffn_gate_exps.weight", 999),
           ("token_embd.weight", 999), ("blk.0.attn_q_norm.weight", 999))
  assert estimate_prefill_overlay_bytes(pairs) == (100 + 200 + 300) * 2


_QWEN3_KV = {
  "general.architecture": "qwen3",
  "qwen3.embedding_length": 4096,
  "qwen3.block_count": 2,
  "qwen3.context_length": 8192,
  "qwen3.attention.head_count": 32,
  "qwen3.attention.head_count_kv": 8,
  "qwen3.attention.key_length": 128,
  "qwen3.feed_forward_length": 12288,
}


def _tensor_info(name, k, n):
  return (name, (k, n), 12, 0)  # ggml_type 12 = Q4_K; file dims are (k, n)


def _synthetic_meta():
  return {"data_start": 0, "tensor_infos": [
    _tensor_info("blk.0.attn_q.weight", 4096, 4096),
    _tensor_info("blk.0.attn_k.weight", 4096, 1024),
    _tensor_info("blk.0.attn_v.weight", 4096, 1024),
    _tensor_info("blk.0.attn_output.weight", 4096, 4096),
    _tensor_info("blk.0.ffn_gate.weight", 4096, 12288),
    _tensor_info("blk.0.ffn_up.weight", 4096, 12288),
    _tensor_info("blk.0.ffn_down.weight", 12288, 4096),
    _tensor_info("blk.0.ffn_gate_shexp.weight", 4096, 4096),
    _tensor_info("blk.0.attn_q_norm.weight", 4096, 1),
    _tensor_info("token_embd.weight", 4096, 151936),
    _tensor_info("output.weight", 4096, 151936),
  ]}


def test_inventory_overlay_bytes_derive_from_role_rows_with_lm_head_gate():
  inventory = derive_selected_gguf_prefill_inventory(_QWEN3_KV, _synthetic_meta())
  covered = {("attn_q", 4096, 4096), ("attn_k", 4096, 1024), ("attn_v", 4096, 1024),
             ("attn_output", 4096, 4096), ("ffn_gate", 4096, 12288), ("ffn_up", 4096, 12288),
             ("ffn_down", 12288, 4096), ("ffn_gate_shexp", 4096, 4096)}
  assert inventory["overlay_bytes"] == sum(k * n * 2 for _, k, n in covered)
  resident = derive_selected_gguf_prefill_inventory(_QWEN3_KV, _synthetic_meta(), lm_head_resident_fp16=True)
  assert resident["overlay_bytes"] == inventory["overlay_bytes"] + 4096 * 151936 * 2


def test_moe_shared_expert_rows_resolve_against_shared_expert_size():
  kv = {"general.architecture": "qwen35moe", "qwen35moe.embedding_length": 5120,
        "qwen35moe.feed_forward_length": 5120, "qwen35moe.expert_shared_feed_forward_length": 4096,
        "qwen35moe.block_count": 1, "qwen35moe.context_length": 8192,
        "qwen35moe.attention.head_count": 40, "qwen35moe.attention.head_count_kv": 8,
        "qwen35moe.attention.key_length": 128}
  meta = {"data_start": 0, "tensor_infos": [
    _tensor_info("blk.0.ffn_gate_shexp.weight", 5120, 4096),
    _tensor_info("blk.0.ffn_up_shexp.weight", 5120, 4096),
    _tensor_info("blk.0.ffn_down_shexp.weight", 4096, 5120),
  ]}
  inventory = derive_selected_gguf_prefill_inventory(kv, meta)
  assert inventory["overlay_bytes"] == (4096 * 5120 * 2) * 3
  assert {row["role"] for row in inventory["rows"]} == {"ffn_gate_up", "ffn_down"}


@pytest.mark.skipif(not pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf").exists(),
                    reason="no local Qwen3 8B GGUF fixture")
def test_inventory_overlay_bytes_equal_model_walk_on_real_qwen3_8b():
  """Ratchet: the inventory-derived estimate must equal the _prefill_v2_covered() walk on the real 8B fixture."""
  fixture = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  kv, meta = gguf_load_metadata(fixture)
  inventory = derive_selected_gguf_prefill_inventory(kv, meta)
  assert inventory["overlay_bytes"] > 0
  model, _ = Transformer.from_gguf(str(fixture), 2048)
  walk_bytes = sum(out_f * in_f * 2 for _, out_f, in_f in model._prefill_v2_covered())
  assert inventory["overlay_bytes"] == walk_bytes
  resident = derive_selected_gguf_prefill_inventory(kv, meta, lm_head_resident_fp16=True)
  lm_head_row = next(row for row in inventory["rows"] if row["role"] == "lm_head")
  assert resident["overlay_bytes"] == inventory["overlay_bytes"] + lm_head_row["shape"]["n"] * lm_head_row["shape"]["k"] * 2
