from tinygrad.llm.model_facts import GGML_QUANT_LABELS, model_facts_from_gguf_metadata, tensor_fact_from_gguf_row


def _qwen_kv(hidden: int, intermediate: int, heads: int, kv_heads: int = 8, head_dim: int = 128) -> dict:
  return {
    "general.architecture": "qwen3",
    "qwen3.embedding_length": hidden,
    "qwen3.feed_forward_length": intermediate,
    "qwen3.attention.head_count": heads,
    "qwen3.attention.head_count_kv": kv_heads,
    "qwen3.attention.key_length": head_dim,
  }


def _qwen_tensor_infos(hidden: int, intermediate: int, vocab: int = 151936) -> list[tuple[str, tuple[int, int], int, int]]:
  return [
    ("blk.0.ffn_gate.weight", (hidden, intermediate), 12, 0),
    ("blk.0.ffn_up.weight", (hidden, intermediate), 12, 1),
    ("blk.0.ffn_down.weight", (intermediate, hidden), 14, 2),
    ("blk.0.attn_q.weight", (hidden, hidden), 12, 3),
    ("blk.0.attn_output.weight", (hidden, hidden), 12, 4),
    ("blk.0.attn_k.weight", (hidden, 1024), 12, 5),
    ("blk.0.attn_v.weight", (hidden, 1024), 12, 6),
    ("output.weight", (hidden, vocab), 12, 7),
  ]


def test_ggml_quant_labels_cover_q4k_and_q6k():
  assert GGML_QUANT_LABELS[12] == "Q4_K"
  assert GGML_QUANT_LABELS[14] == "Q6_K"
  assert tensor_fact_from_gguf_row(("blk.0.ffn_gate.weight", (4096, 12288), 12, 0)).quant_label == "Q4_K"
  assert tensor_fact_from_gguf_row({"name": "blk.0.ffn_down.weight", "dims": (12288, 4096), "ggml_type": 14}).quant_label == "Q6_K"


def test_qwen3_8b_like_metadata_derives_dense_roles_from_names_and_shapes():
  facts = model_facts_from_gguf_metadata(_qwen_kv(4096, 12288, 32), {"tensor_infos": _qwen_tensor_infos(4096, 12288)})

  assert facts.architecture == "qwen3"
  assert facts.hidden_size == 4096
  assert [(t.name, t.rows, t.cols, t.quant_label, t.role) for t in facts.tensors] == [
    ("blk.0.ffn_gate.weight", 12288, 4096, "Q4_K", "ffn_gate_up"),
    ("blk.0.ffn_up.weight", 12288, 4096, "Q4_K", "ffn_gate_up"),
    ("blk.0.ffn_down.weight", 4096, 12288, "Q6_K", "ffn_down"),
    ("blk.0.attn_q.weight", 4096, 4096, "Q4_K", "attn_qo"),
    ("blk.0.attn_output.weight", 4096, 4096, "Q4_K", "attn_qo"),
    ("blk.0.attn_k.weight", 1024, 4096, "Q4_K", "attn_kv"),
    ("blk.0.attn_v.weight", 1024, 4096, "Q4_K", "attn_kv"),
    ("output.weight", 151936, 4096, "Q4_K", "lm_head"),
  ]
  assert [t.module_path for t in facts.tensors[:3]] == ["blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down"]


def test_qwen3_14b_like_metadata_uses_config_shape_without_model_size_label():
  facts = model_facts_from_gguf_metadata(_qwen_kv(5120, 17408, 40), {"tensor_infos": _qwen_tensor_infos(5120, 17408)})

  assert facts.n_heads == 40
  assert facts.head_dim == 128
  assert [(t.role, t.shape) for t in facts.tensors_for_role("ffn_gate_up")] == [
    ("ffn_gate_up", (17408, 5120)),
    ("ffn_gate_up", (17408, 5120)),
  ]
  assert [(t.role, t.shape) for t in facts.tensors_for_role("attn_kv")] == [
    ("attn_kv", (1024, 5120)),
    ("attn_kv", (1024, 5120)),
  ]


def test_qwen_resolver_rejects_name_matches_with_wrong_config_shape():
  facts = model_facts_from_gguf_metadata(_qwen_kv(4096, 12288, 32), {
    "tensor_infos": [("blk.0.ffn_gate.weight", (4096, 17408), 12, 0)]
  })

  assert facts.tensors[0].role is None


def test_model_facts_only_include_2d_weight_rows():
  facts = model_facts_from_gguf_metadata(_qwen_kv(4096, 12288, 32), {
    "tensor_infos": [
      ("blk.0.ffn_gate.weight", (4096, 12288), 12, 0),
      ("blk.0.ffn_gate.bias", (12288,), 0, 1),
      ("token_embd.weight", (4096,), 12, 2),
    ]
  })

  assert [t.name for t in facts.tensors] == ["blk.0.ffn_gate.weight"]
