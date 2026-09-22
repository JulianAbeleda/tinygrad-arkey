import pytest

from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter, default_registry


def _program():
  return {"schema_version": "boltbeam.semantic_kernel_program.v1", "nodes": [
    {"id": "x", "op": "input", "inputs": [], "attrs": {"dtype": "fp16", "layout": "row_major"}},
    {"id": "c", "op": "constant", "inputs": [], "attrs": {"dtype": "fp32", "value": 1.0}},
    {"id": "o", "op": "output", "inputs": ["x"], "attrs": {"dtype": "fp16", "layout": "row_major"}}], "outputs": ["o"]}


def test_default_registry_lowers_transport_nodes_and_preserves_program_identity():
  result = default_registry().lower(_program())
  assert result.outputs[0]["kind"] == "input"
  assert len(result.program_hash) == 64


def test_registered_semantic_primitive_is_the_only_extension_point():
  registry = default_registry(); seen = []
  registry.register("residual_add", lambda node, args: seen.append((node.node_id, args)) or (args[0] + args[1]))
  program = _program(); program["nodes"].insert(2, {"id": "r", "op": "residual_add", "inputs": ["c", "c"], "attrs": {"dtype": "fp32"}})
  program["nodes"][3]["inputs"] = ["r"]
  result = registry.lower(program)
  assert result.outputs == (2.0,) and seen[0][0] == "r"


def test_missing_lowering_unknown_operation_and_forward_reference_fail_closed():
  with pytest.raises(ValueError, match="no tinygrad lowering"):
    default_registry().lower({"schema_version": "boltbeam.semantic_kernel_program.v1", "nodes": [
      {"id": "x", "op": "input", "inputs": [], "attrs": {}},
      {"id": "q", "op": "q8_1_quantize", "inputs": ["x"], "attrs": {}}], "outputs": ["q"]})
  with pytest.raises(ValueError, match="unknown semantic primitive"):
    default_registry().lower({"schema_version": "boltbeam.semantic_kernel_program.v1", "nodes": [
      {"id": "x", "op": "nope", "inputs": [], "attrs": {}}], "outputs": ["x"]})
  bad = _program(); bad["nodes"][2]["inputs"] = ["future"]
  with pytest.raises(ValueError, match="missing or forward"):
    default_registry().lower(bad)


def test_existing_q4_q6_q8_families_are_closed_registered_adapters():
  families = [
    ("q8_1_provider.v1", {"k": 4096, "source_dtype": "fp16"}),
    ("q4_g3_gemv.v1", {"rows": 4096, "k": 4096, "lanes": 32, "load_style": "vector", "epilogue": ""}),
    ("q4_w1w3.v1", {"rows": 12288, "k": 4096, "load_style": "vector", "store_fp16": True}),
    ("q4_gate_up.v1", {"vector_loads": True}),
    ("q4_ffn_down.v1", {"block_count": 3, "resadd": True, "load_style": "vector"}),
    ("q4_kv_pair.v1", {"rows": 1024, "k": 4096}),
    ("q6_ffn_down.v1", {"rows_per_block": 1, "packed_lanemap": True, "unroll_blocks": 4, "split_weight_stream": False}),
    ("q6_v.v1", {}),
    ("shared_q8_consumer.v1", {"rows": 4096, "block_count": 16, "direct_output": True, "residual_add": False, "quant": "q4"}),
  ]
  for family, params in families:
    assert callable(build_registered_llm_emitter(family, params)), family
  with pytest.raises(ValueError, match="no registered"):
    build_registered_llm_emitter("unknown.v1", {})
  with pytest.raises(ValueError, match="parameters are invalid"):
    build_registered_llm_emitter("q4_ffn_down.v1", {"block_count":12,"resadd":True,"load_style":"vector"})
