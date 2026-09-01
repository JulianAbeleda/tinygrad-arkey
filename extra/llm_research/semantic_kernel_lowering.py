"""Tinygrad-side lowering boundary for Boltbeam semantic kernel programs.

The registry is intentionally data-driven and closed: candidates name semantic
operations, never Python imports or backend source. Existing tinygrad primitive
implementations are registered by the owning family adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any, Callable
from extra.llm_research.boltbeam_runtime_ticket import BoltbeamKernelTicket

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


def build_registered_llm_emitter(family: str, parameters: dict[str, Any], bindings: dict[str, Any] | None = None):
  """Adapt a closed Boltbeam family to an existing tinygrad UOp builder."""
  if not isinstance(family, str) or not isinstance(parameters, dict): raise ValueError("LLM emitter family and parameters are required")
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  if family == "q8_1_provider.v1":
    if set(parameters) != {"k", "source_dtype"} or parameters["source_dtype"] not in ("fp16", "fp32"):
      raise ValueError("q8_1_provider.v1 parameters are invalid")
    from tinygrad.llm.q4k_ffn_down_mmvq import emit_q8_provider
    source_dtype = dtypes.float16 if parameters["source_dtype"] == "fp16" else dtypes.float32
    return emit_q8_provider(source_dtype, k=parameters["k"])
  if family == "shared_q8_provider.v1":
    if set(parameters) != {"k", "source_dtype"} or parameters != {"k":4096, "source_dtype":"fp16"}:
      raise ValueError("shared_q8_provider.v1 parameters are invalid")
    from tinygrad.llm.shared_q8_attention import _emit_q8_provider
    return _emit_q8_provider()
  if family == "q4_g3_gemv.v1":
    required = {"rows", "k", "lanes", "load_style", "epilogue"}
    if set(parameters) != required or parameters["load_style"] not in ("scalar", "vector", "quad"):
      raise ValueError("q4_g3_gemv.v1 parameters are invalid")
    from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue, q4k_g3_lanemap_gemv_kernel
    return q4k_g3_lanemap_gemv_kernel(parameters["rows"], parameters["k"], lanes=parameters["lanes"],
      epilogue=Q4KGEMVEpilogue(parameters["epilogue"]), load_style=parameters["load_style"])
  if family == "q4_w1w3.v1":
    required = {"rows", "k", "load_style", "store_fp16"}
    if set(parameters) != required or parameters["load_style"] not in ("scalar", "vector", "quad") or not isinstance(parameters["store_fp16"], bool):
      raise ValueError("q4_w1w3.v1 parameters are invalid")
    from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
    return q4k_g3_lanemap_gemv_w1w3_kernel(parameters["rows"], parameters["k"], load_style=parameters["load_style"], store_fp16=parameters["store_fp16"])
  if family == "q4_gate_up.v1":
    if set(parameters) != {"vector_loads"} or not isinstance(parameters["vector_loads"], bool): raise ValueError("q4_gate_up.v1 parameters are invalid")
    from tinygrad.llm.q4k_gate_up_four_warp_mmvq import emit_q4k_gate_up_four_warp_fp16
    return emit_q4k_gate_up_four_warp_fp16(parameters["vector_loads"])
  if family == "q4_ffn_down.v1":
    if set(parameters) != {"block_count", "resadd", "load_style"} or parameters["load_style"] not in ("scalar", "vector"):
      raise ValueError("q4_ffn_down.v1 parameters are invalid")
    from tinygrad.llm.q4k_ffn_down_mmvq import emit_four_warp_fp16_direct
    return emit_four_warp_fp16_direct(UOp.const(dtypes.weakint, parameters["block_count"]), resadd=parameters["resadd"], load_style=parameters["load_style"])
  if family == "q4_kv_pair.v1":
    if set(parameters) != {"rows", "k"}: raise ValueError("q4_kv_pair.v1 parameters are invalid")
    from tinygrad.llm.q4k_kv_pair import emit_q4k_kv_pair_vector
    return emit_q4k_kv_pair_vector(parameters["rows"], parameters["k"])
  if family == "q6_ffn_down.v1":
    required = {"rows_per_block", "packed_lanemap", "unroll_blocks", "split_weight_stream"}
    if set(parameters) != required or not isinstance(parameters["rows_per_block"], int) or not all(
        isinstance(parameters[x], bool) for x in ("packed_lanemap", "split_weight_stream")) or (
        parameters["unroll_blocks"] is not None and not isinstance(parameters["unroll_blocks"], int)):
      raise ValueError("q6_ffn_down.v1 parameters are invalid")
    from tinygrad.llm.q6k_ffn_down_mmvq import emit_q6k_four_warp_fp16_direct
    return emit_q6k_four_warp_fp16_direct(rows_per_block=parameters["rows_per_block"], packed_lanemap=parameters["packed_lanemap"],
      unroll_blocks=parameters["unroll_blocks"], split_weight_stream=parameters["split_weight_stream"])
  if family == "q6_v.v1":
    if parameters: raise ValueError("q6_v.v1 takes no parameters")
    from tinygrad.llm.q6k_v_mmvq import emit_q6k_v_four_warp_fp16_direct
    return emit_q6k_v_four_warp_fp16_direct()
  if family == "shared_q8_consumer.v1":
    required = {"rows", "block_count", "direct_output", "residual_add", "quant"}
    if set(parameters) != required or parameters["quant"] not in ("q4", "q6"):
      raise ValueError("shared_q8_consumer.v1 parameters are invalid")
    from tinygrad.llm.shared_q8_attention import _emit_q4_cooperative, _emit_q6
    if parameters["quant"] == "q4":
      return _emit_q4_cooperative(parameters["rows"], UOp.const(dtypes.weakint, parameters["block_count"]),
        direct_output=parameters["direct_output"], residual_add=parameters["residual_add"])
    if parameters["direct_output"] or parameters["residual_add"]: raise ValueError("shared Q6 consumer has no Q4 epilogue flags")
    return _emit_q6(parameters["rows"])
  if family == "shared_q8_attention_consumer.v1":
    required = {"rows", "variant", "direct_output", "block_count_binding"}
    if set(parameters) != required or parameters["variant"] not in ("q4_scalar", "q4_cooperative", "q6", "q6_warp_direct"):
      raise ValueError("shared_q8_attention_consumer.v1 parameters are invalid")
    from tinygrad.llm.shared_q8_attention import _emit_q4, _emit_q4_cooperative, _emit_q6, _emit_q6_warp_direct
    variant = parameters["variant"]
    if variant == "q4_scalar": return _emit_q4(parameters["rows"])
    if variant == "q6": return _emit_q6(parameters["rows"])
    if variant == "q6_warp_direct": return _emit_q6_warp_direct(parameters["rows"])
    name = parameters["block_count_binding"]
    if not isinstance(name, str) or not bindings or name not in bindings: raise ValueError("cooperative block-count binding is missing")
    return _emit_q4_cooperative(parameters["rows"], bindings[name], direct_output=parameters["direct_output"])
  if family == "shared_q8_multi_output.v1":
    if set(parameters) != {"variant", "rows", "block_count_binding"} or parameters["variant"] not in (
        "q4q6_qkv", "q4q4_qkv", "q4kv_pair", "q4q6_pair"):
      raise ValueError("shared_q8_multi_output.v1 parameters are invalid")
    name=parameters["block_count_binding"]
    if not isinstance(name,str) or not bindings or name not in bindings: raise ValueError("multi-output block-count binding is missing")
    from tinygrad.llm.shared_q8_attention import (_emit_q4_q6_cooperative_qkv_full, _emit_q4_cooperative_qkv_full,
      _emit_q4_cooperative_pair, _emit_q4_q6_cooperative_pair)
    if parameters["variant"] == "q4q6_qkv": return _emit_q4_q6_cooperative_qkv_full(bindings[name])
    if parameters["variant"] == "q4q4_qkv": return _emit_q4_cooperative_qkv_full(bindings[name])
    emitter=_emit_q4_cooperative_pair if parameters["variant"] == "q4kv_pair" else _emit_q4_q6_cooperative_pair
    return emitter(parameters["rows"],bindings[name])
  if family == "q4q4_qkv_full.v1":
    if set(parameters) != {"mixed_q6_v"} or not isinstance(parameters["mixed_q6_v"],bool): raise ValueError("q4q4_qkv_full.v1 parameters are invalid")
    from tinygrad.llm.q4k_kv_pair import emit_q4k_q4k_q6_qkv_full, emit_q4k_qkv_full
    return emit_q4k_q4k_q6_qkv_full() if parameters["mixed_q6_v"] else emit_q4k_qkv_full()
  if family == "finite_argmax.v1":
    if set(parameters) != {"n", "threads", "host_mirror"} or not isinstance(parameters["host_mirror"], bool):
      raise ValueError("finite_argmax.v1 parameters are invalid")
    from tinygrad.llm.packed_argmax import emit_native_finite_fp32_argmax
    return emit_native_finite_fp32_argmax(parameters["n"], parameters["threads"], host_mirror=parameters["host_mirror"])
  if family == "kv_rope_store.v1":
    if set(parameters) != {"Hkv", "Hd", "max_context", "vparts"}: raise ValueError("kv_rope_store.v1 parameters are invalid")
    from tinygrad.llm.decode_routes import decode_kv_rope_store_kernel
    return decode_kv_rope_store_kernel(parameters["Hkv"], parameters["Hd"], parameters["max_context"], VPART=parameters["vparts"])
  if family == "q4_k_four_warp.v1":
    if set(parameters) != {"rows", "k"}: raise ValueError("q4_k_four_warp.v1 parameters are invalid")
    from extra.llm_research.decode.q4k_exact_group_factorized import emit_q4k_exact_four_warp
    return emit_q4k_exact_four_warp(parameters["rows"], parameters["k"])
  if family == "decode_rmsnorm.v1":
    required={"rows","dim","eps","warps_per_row","x_dtype","weight_dtype","out_dtype","x_rank"}
    if set(parameters) != required: raise ValueError("decode_rmsnorm.v1 parameters are invalid")
    dtype_map={"dtypes.half":dtypes.float16,"dtypes.float":dtypes.float32}
    try: x_dtype,weight_dtype,out_dtype=(dtype_map[parameters[name]] for name in ("x_dtype","weight_dtype","out_dtype"))
    except KeyError as exc: raise ValueError("decode_rmsnorm.v1 dtype is invalid") from exc
    from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
    spec=DecodeRMSNormSpec(rows=parameters["rows"],dim=parameters["dim"],eps=parameters["eps"],
      warps_per_row=parameters["warps_per_row"],x_dtype=x_dtype,weight_dtype=weight_dtype,
      out_dtype=out_dtype,x_rank=parameters["x_rank"])
    return emit_decode_rmsnorm_kernel(spec)
  if family == "rmsnorm_q8_provider.v1":
    required={"rows","dim","eps","recipe","warps","x_dtype","weight_dtype","spec_binding"}
    if set(parameters) != required: raise ValueError("rmsnorm_q8_provider.v1 parameters are invalid")
    name=parameters["spec_binding"]
    if not isinstance(name,str) or not bindings or name not in bindings: raise ValueError("RMSNorm Q8 spec binding is missing")
    spec=bindings[name]
    for field in ("rows","dim","eps","recipe","warps"):
      if getattr(spec,field) != parameters[field]: raise ValueError(f"RMSNorm Q8 spec {field} drift")
    dtype_map={"dtypes.half":dtypes.float16,"dtypes.float":dtypes.float32}
    try: x_dtype,weight_dtype=dtype_map[parameters["x_dtype"]],dtype_map[parameters["weight_dtype"]]
    except KeyError as exc: raise ValueError("rmsnorm_q8_provider.v1 dtype is invalid") from exc
    from tinygrad.llm.shared_q8_attention import _emit_rmsnorm_q8_provider
    return _emit_rmsnorm_q8_provider(spec,x_dtype,weight_dtype)
  if family == "q4_g3_route.v1":
    required={"rows","k","load_style","epilogue_kind","epilogue_binding"}
    if set(parameters) != required: raise ValueError("q4_g3_route.v1 parameters are invalid")
    epilogue=None
    if parameters["epilogue_binding"] is not None:
      name=parameters["epilogue_binding"]
      if not bindings or name not in bindings: raise ValueError("Q4 epilogue binding is missing")
      epilogue=bindings[name]
      if getattr(epilogue,"kind",None) != parameters["epilogue_kind"]: raise ValueError("Q4 epilogue kind drift")
    elif parameters["epilogue_kind"]: raise ValueError("Q4 epilogue binding is required")
    from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
    return q4k_g3_lanemap_gemv_kernel(parameters["rows"],parameters["k"],epilogue=epilogue,load_style=parameters["load_style"])
  if family in ("q6_gemv_route.v1","q6_vocab_reduce_route.v1"):
    required={"rows","k","row_tile","reduction","epilogue","spec_binding"}
    if set(parameters) != required: raise ValueError("Q6 route parameters are invalid")
    name=parameters["spec_binding"]
    if not bindings or name not in bindings: raise ValueError("Q6 spec binding is missing")
    spec=bindings[name]
    for field,value in (("rows",parameters["rows"]),("k",parameters["k"]),("row_tile",parameters["row_tile"]),
                        ("reduction",parameters["reduction"]),("epilogue",parameters["epilogue"])):
      if getattr(spec,field) != value: raise ValueError(f"Q6 spec {field} drift")
    from tinygrad.llm.decode_kernels import emit_q6k_gemv_kernel, emit_q6k_vocab_scalar_reduce_kernel
    return emit_q6k_gemv_kernel(spec) if family == "q6_gemv_route.v1" else emit_q6k_vocab_scalar_reduce_kernel(spec)
  if family == "qk_norm_rope_cache_sink.v1":
    required={"spec_repr","producer_dtype","weight_dtype","cache_dtype","max_context","spec_binding"}
    if set(parameters) != required: raise ValueError("qk_norm_rope_cache_sink.v1 parameters are invalid")
    name=parameters["spec_binding"]
    if not bindings or name not in bindings: raise ValueError("cache-sink spec binding is missing")
    spec=bindings[name]
    if repr(spec) != parameters["spec_repr"]: raise ValueError("cache-sink spec drift")
    dtype_map={"dtypes.half":dtypes.float16,"dtypes.float":dtypes.float32}
    try: producer_dtype,weight_dtype,cache_dtype=(dtype_map[parameters[x]] for x in ("producer_dtype","weight_dtype","cache_dtype"))
    except KeyError as exc: raise ValueError("cache-sink dtype is invalid") from exc
    from tinygrad.llm.producer_kv_cache_sink import emit_reduce_output_rope_kv_cache
    return emit_reduce_output_rope_kv_cache(spec,producer_dtype,weight_dtype,cache_dtype,parameters["max_context"])
  raise ValueError(f"no registered tinygrad LLM emitter for family {family!r}")


def make_boltbeam_ticket(program_hash: str, route_hash: str, family: str, target_identity: str,
                         provider_revision: str = "semantic-lowering-v1") -> BoltbeamKernelTicket:
  """Create the runtime provenance token after BoltBeam has authorized a candidate."""
  return BoltbeamKernelTicket(program_hash, route_hash, family, target_identity, provider_revision)


__all__ = ["LoweredSemanticProgram", "PrimitiveNode", "SemanticLoweringRegistry", "build_registered_llm_emitter", "default_registry", "make_boltbeam_ticket"]
