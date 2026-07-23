from __future__ import annotations

import importlib
from functools import cache


@cache
def _attr(module:str, name:str):
  return getattr(importlib.import_module(module), name)

def automatic_promoted_prefill_graph_policy(*args, **kwargs):
  return _attr("extra.qk.route_manifest", "automatic_promoted_prefill_graph_policy")(*args, **kwargs)
def install_memory_adaptive_model_adapters():
  return _attr("extra.qk.memory_adaptive_runtime_collector", "install_model_adapters")()


def route_pf16_graph_gemm(*args, **kwargs): return _attr("extra.qk.prefill_graph_gemm_route", "route_pf16_graph_gemm")(*args, **kwargs)

def q4k_parse_opt(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "parse_opt")(*args, **kwargs)
def q4k_gemm_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_kernel")(*args, **kwargs)
def q4k_gemm_packed_load_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_packed_load_kernel")(*args, **kwargs)
def q4k_gemm_packed_load_direct_out_kernel(*args, **kwargs):
  return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_packed_load_direct_out_kernel")(*args, **kwargs)
def q4k_gemm_packed_load_reduce_out_kernel(*args, **kwargs):
  return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_packed_load_reduce_out_kernel")(*args, **kwargs)
def describe_q4k_packed_prefill_generated(*args, **kwargs): return _attr("extra.qk.q4k_prefill_route_spec", "describe_q4k_packed_prefill")(*args, **kwargs)
def emit_q4k_packed_prefill_kernel(*args, **kwargs):
  return _attr("extra.qk.q4k_prefill_route_spec", "emit_q4k_packed_prefill_kernel")(*args, **kwargs)
def describe_q4k_int8_wmma_prefill(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "describe_q4k_int8_wmma_prefill")(*args, **kwargs)
def describe_q4k_int8_wmma_tiled_prefill(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "describe_q4k_int8_wmma_tiled_prefill")(*args, **kwargs)
def emit_q4k_int8_wmma_prefill_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_prefill_tensor")(*args, **kwargs)
def emit_q4k_int8_wmma_tiled_prefill_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_tiled_prefill_tensor")(*args, **kwargs)
def emit_q4k_int8_wmma_tiled_exec_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_tiled_exec_tensor")(*args, **kwargs)
def emit_q4k_int8_wmma_tiled_lifecycle_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_tiled_lifecycle_tensor")(*args, **kwargs)
def emit_q4k_int8_wmma_tiled_scheduler_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_tiled_scheduler_tensor")(*args, **kwargs)
def q8_1_quantize(*args, **kwargs): return _attr("extra.qk.layout", "q8_1_quantize")(*args, **kwargs)
def packed_ds4_candidate(*args, **kwargs): return _attr("extra.qk.mmq_ds4_logical_emitter", "packed_ds4_candidate")(*args, **kwargs)
def packed_row_major_candidate(*args, **kwargs): return _attr("extra.qk.mmq_ds4_logical_emitter", "packed_row_major_candidate")(*args, **kwargs)
def packed_fused_candidate(*args, **kwargs): return _attr("extra.qk.mmq_ds4_logical_emitter", "packed_fused_candidate")(*args, **kwargs)
def pack_q8_1_mmq_ds4(*args, **kwargs): return _attr("extra.qk.mmq_ds4_logical_emitter", "pack_q8_1_mmq_ds4")(*args, **kwargs)
def pack_q8_1_mmq_fused(*args, **kwargs): return _attr("extra.qk.mmq_ds4_logical_emitter", "pack_q8_1_mmq_fused")(*args, **kwargs)
def emit_q4k_q8_mmq_ds4(*args, **kwargs): return _attr("extra.qk.mmq_ds4_logical_emitter", "emit_q4k_q8_mmq_ds4")(*args, **kwargs)
def quantize_q4_k(*args, **kwargs): return _attr("extra.qk.quantize", "quantize_q4_k")(*args, **kwargs)

def should_route_q4k_lane_partition(*args, **kwargs): return _attr("extra.qk.bubblebeam_futuresight", "should_route_q4k_lane_partition")(*args, **kwargs)
def q4k_g3_manifest_shape(*args, **kwargs): return _attr("extra.qk.bubblebeam_futuresight", "q4k_g3_manifest_shape")(*args, **kwargs)
def q4k_g3_lanemap_gemv_kernel(*args, **kwargs): return _attr("extra.qk.gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_kernel")(*args, **kwargs)
def q4k_g3_lanemap_gemv_splitk_kernel(*args, **kwargs):
  return _attr("extra.qk.gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_splitk_kernel")(*args, **kwargs)
def q4k_g3_lanemap_gemv_inkernel_combine_kernel(*args, **kwargs):
  return _attr("extra.qk.gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_inkernel_combine_kernel")(*args, **kwargs)

def q6k_parse_opt(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "parse_opt")(*args, **kwargs)
def q6k_gemm_kernel(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemm_kernel")(*args, **kwargs)
def q6k_gemm_packed_load_kernel(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemm_packed_load_kernel")(*args, **kwargs)
def q6k_gemm_packed_load_direct_out_kernel(*args, **kwargs):
  return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemm_packed_load_direct_out_kernel")(*args, **kwargs)
def q6k_spec_for_role(*args, **kwargs): return _attr("extra.qk.q6k_route_spec", "spec_for_role")(*args, **kwargs)
def emit_q6k_gemv_kernel(*args, **kwargs): return _attr("extra.qk.q6k_route_spec", "emit_q6k_gemv_kernel")(*args, **kwargs)
def q6k_vocab_scalar_reduce_eligible(*args, **kwargs):
  return _attr("extra.qk.q6k_route_spec", "q6k_vocab_scalar_reduce_eligible")(*args, **kwargs)
def emit_q6k_vocab_scalar_reduce_kernel(*args, **kwargs):
  return _attr("extra.qk.q6k_route_spec", "emit_q6k_vocab_scalar_reduce_kernel")(*args, **kwargs)
def describe_q6k_packed_prefill(*args, **kwargs): return _attr("extra.qk.q6k_prefill_route_spec", "describe_q6k_packed_prefill")(*args, **kwargs)
def emit_q6k_packed_prefill_kernel(*args, **kwargs):
  return _attr("extra.qk.q6k_prefill_route_spec", "emit_q6k_packed_prefill_kernel")(*args, **kwargs)

def flash_decode_live_split_block_tile(*args, **kwargs):
  return _attr("extra.qk.flash_decode_attention_executor", "flash_decode_live_split_block_tile")(*args, **kwargs)

def assert_pure_machine_search(*args, **kwargs): return _attr("extra.qk.pure_search_guard", "assert_pure_machine_search")(*args, **kwargs)
