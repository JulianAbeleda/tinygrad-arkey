"""Tinygrad-side lowering boundary for Boltbeam semantic kernel programs.

The registry is intentionally data-driven and closed: candidates name semantic
operations, never Python imports or backend source. Existing tinygrad primitive
implementations are registered by the owning family adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any, Callable

SEMANTIC_PROGRAM_SCHEMA = "boltbeam.semantic_kernel_program.v1"
_OPS = frozenset({
  "input", "constant", "global_load", "vector_load", "shared_stage", "barrier", "publish",
  "q8_1_quantize", "packed_q4_k_extract", "packed_q6_k_extract", "signed_int8_dot", "matrix_dot",
  "subgroup_reduce", "workgroup_reduce", "ordered_partial_reduce", "stream_k_partition",
  "rmsnorm", "rope", "flash_score", "online_softmax", "flash_pv", "silu_mul", "residual_add",
  "output_cast", "cache_store", "finite_argmax", "output"})


def _canonical(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


@dataclass(frozen=True)
class PrimitiveNode:
  node_id: str
  op: str
  attrs: dict[str, Any]


@dataclass(frozen=True)
class LoweredSemanticProgram:
  program_hash: str
  values: dict[str, Any]
  outputs: tuple[Any, ...]


Lowering = Callable[[PrimitiveNode, tuple[Any, ...]], Any]


class SemanticLoweringRegistry:
  def __init__(self) -> None:
    self._lowerings: dict[str, Lowering] = {}

  def register(self, op: str, lowering: Lowering) -> None:
    if op not in _OPS: raise ValueError(f"unknown semantic primitive {op!r}")
    if not callable(lowering): raise ValueError("semantic lowering must be callable")
    if op in self._lowerings: raise ValueError(f"duplicate semantic lowering {op!r}")
    self._lowerings[op] = lowering

  def lower(self, program: dict[str, Any]) -> LoweredSemanticProgram:
    normalized = json.loads(_canonical(program))
    if not isinstance(normalized, dict) or set(normalized) != {"schema_version", "nodes", "outputs"}:
      raise ValueError("semantic program fields do not match Boltbeam contract")
    if normalized["schema_version"] != SEMANTIC_PROGRAM_SCHEMA: raise ValueError("unsupported semantic program schema")
    nodes = normalized["nodes"]
    if not isinstance(nodes, list) or not nodes: raise ValueError("semantic program requires nodes")
    values: dict[str, Any] = {}
    for index, raw in enumerate(nodes):
      if not isinstance(raw, dict) or set(raw) != {"id", "op", "inputs", "attrs"}:
        raise ValueError(f"nodes[{index}] has invalid fields")
      node_id, op, refs, attrs = raw["id"], raw["op"], raw["inputs"], raw["attrs"]
      if not isinstance(node_id, str) or not node_id: raise ValueError(f"nodes[{index}] id is invalid")
      if node_id in values: raise ValueError(f"duplicate semantic node id {node_id!r}")
      if op not in _OPS: raise ValueError(f"unknown semantic primitive {op!r}")
      if not isinstance(refs, list) or any(ref not in values for ref in refs):
        raise ValueError(f"nodes[{index}] has missing or forward input reference")
      if not isinstance(attrs, dict): raise ValueError(f"nodes[{index}] attrs must be an object")
      lowering = self._lowerings.get(op)
      if lowering is None: raise ValueError(f"no tinygrad lowering registered for semantic primitive {op!r}")
      values[node_id] = lowering(PrimitiveNode(node_id, op, attrs), tuple(values[ref] for ref in refs))
    outputs = normalized["outputs"]
    if not isinstance(outputs, list) or not outputs or any(ref not in values for ref in outputs):
      raise ValueError("semantic program outputs are invalid")
    return LoweredSemanticProgram(hashlib.sha256(_canonical(normalized).encode("ascii")).hexdigest(), values,
                                  tuple(values[ref] for ref in outputs))


def default_registry() -> SemanticLoweringRegistry:
  registry = SemanticLoweringRegistry()
  registry.register("input", lambda node, args: {"kind": "input", "id": node.node_id, "attrs": node.attrs})
  registry.register("constant", lambda node, args: node.attrs["value"])
  registry.register("output", lambda node, args: args[0])
  return registry


__all__ = ["LoweredSemanticProgram", "PrimitiveNode", "SemanticLoweringRegistry", "default_registry"]
