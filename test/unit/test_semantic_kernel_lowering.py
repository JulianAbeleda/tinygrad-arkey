import pytest

from extra.llm_research.semantic_kernel_lowering import default_registry


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
