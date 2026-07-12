from extra.qk.pure_anchor_workload_evidence import SCHEMA, build_ffn_gate_up_anchor_evidence
from tinygrad.llm.model_facts import model_facts_from_gguf_metadata


def _facts():
  kv = {
    "general.architecture": "qwen3", "qwen3.embedding_length": 4096,
    "qwen3.feed_forward_length": 12288, "qwen3.attention.head_count": 32,
    "qwen3.attention.head_count_kv": 8, "qwen3.attention.key_length": 128,
  }
  meta = {"tensor_infos": [
    ("blk.0.ffn_gate.weight", (4096, 12288), 12, 0),
    ("blk.0.ffn_up.weight", (4096, 12288), 12, 1),
  ]}
  return model_facts_from_gguf_metadata(kv, meta)


def test_anchor_evidence_joins_existing_profile_model_and_route_semantics():
  row = build_ffn_gate_up_anchor_evidence(model_facts=_facts())
  assert row["schema"] == SCHEMA
  assert row["shape"] == {"B": 1, "M": 512, "N": 12288, "K": 4096}
  assert row["linear_semantics"]["input_logical_shape"] == [1, 512, 4096]
  assert row["linear_semantics"]["weight_logical_shape"] == [12288, 4096]
  assert row["linear_semantics"]["output_logical_shape"] == [1, 512, 12288]
  assert [p["name"] for p in row["projections"]] == ["gate", "up"]
  assert all(p["bias"] is False for p in row["projections"])
  assert row["prefill_operand_contract"]["activation_dtype_at_linear"] == "float16"
  assert row["prefill_operand_contract"]["weight_dtype_at_linear"] == "float16"
  assert row["gguf_binding"]["status"] == "proven"
  assert row["gguf_binding"]["ggml_quant_labels"] == ["Q4_K"]
  assert row["status"] == "proven_with_named_unknowns"


def test_anchor_evidence_names_unknowns_instead_of_guessing_backend_semantics():
  row = build_ffn_gate_up_anchor_evidence()
  unknown = {item["field"] for item in row["unresolved"]}
  assert row["gguf_binding"]["status"] == "unresolved"
  assert row["status"] == "profile_proven_gguf_binding_unresolved"
  assert unknown == {
    "concrete_gguf_tensor_binding", "accumulator_dtype", "rounding_and_denormal_mode",
    "correctness_tolerance", "non_anchor_edge_behavior", "lane_fragment_and_memory_mapping",
  }
  assert row["scope_limits"] == {
    "fixed_shape_only": True, "no_tail_claim": True, "no_backend_or_schedule_claim": True,
    "no_accumulation_precision_claim": True, "no_numerical_tolerance_claim": True,
  }


def test_anchor_evidence_refuses_partial_concrete_gguf_binding():
  facts = _facts()
  partial = type(facts)(facts.architecture, facts.hidden_size, facts.intermediate_size, facts.n_heads,
                        facts.n_kv_heads, facts.head_dim, facts.tensors[:1])
  row = build_ffn_gate_up_anchor_evidence(model_facts=partial)
  assert row["gguf_binding"]["status"] == "unresolved"
  assert row["gguf_binding"]["matched_tensors"] == ["blk.0.ffn_gate.weight"]
