from __future__ import annotations

import importlib
from functools import cache


@cache
def _attr(module:str, name:str):
  return getattr(importlib.import_module(module), name)


def route_pf16_graph_gemm(*args, **kwargs): return _attr("extra.qk.prefill_graph_gemm_route", "route_pf16_graph_gemm")(*args, **kwargs)
def route_q4k_graph_gemm(*args, **kwargs): return _attr("extra.qk.prefill_graph_gemm_route", "route_q4k_graph_gemm")(*args, **kwargs)

def q4k_parse_opt(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "parse_opt")(*args, **kwargs)
def q4k_gemm_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_kernel")(*args, **kwargs)
def q4k_gemm_packed_load_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_packed_load_kernel")(*args, **kwargs)
def q4k_gemm_packed_load_direct_out_kernel(*args, **kwargs):
  return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_packed_load_direct_out_kernel")(*args, **kwargs)
def q4k_gemm_packed_load_reduce_out_kernel(*args, **kwargs):
  return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemm_packed_load_reduce_out_kernel")(*args, **kwargs)
def q4k_q8_1_gemm_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_q8_1_gemm_kernel")(*args, **kwargs)
def q4k_q8_1_sdot4_gemm_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_q8_1_sdot4_gemm_kernel")(*args, **kwargs)
def q4k_q8_1_sdot4_coop_gemm_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_q8_1_sdot4_coop_gemm_kernel")(*args, **kwargs)
def q4k_q8_1_sdot4_coop_direct_out_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_q8_1_sdot4_coop_direct_out_kernel")(*args, **kwargs)
def describe_q4k_int8_wmma_prefill(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "describe_q4k_int8_wmma_prefill")(*args, **kwargs)
def describe_q4k_int8_wmma_tiled_prefill(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "describe_q4k_int8_wmma_tiled_prefill")(*args, **kwargs)
def emit_q4k_int8_wmma_prefill_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_prefill_tensor")(*args, **kwargs)
def emit_q4k_int8_wmma_tiled_prefill_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_tiled_prefill_tensor")(*args, **kwargs)
def emit_q4k_int8_wmma_tiled_lifecycle_tensor(*args, **kwargs): return _attr("extra.qk.prefill_int8_wmma_spec", "emit_q4k_int8_wmma_tiled_lifecycle_tensor")(*args, **kwargs)
def describe_q4k_packed_prefill_tile(*args, **kwargs): return _attr("extra.qk.prefill_packed_tile_spec", "describe_q4k_packed_prefill_tile")(*args, **kwargs)
def emit_q4k_packed_prefill_tile(*args, **kwargs): return _attr("extra.qk.prefill_packed_tile_spec", "emit_q4k_packed_prefill_tile")(*args, **kwargs)
def q4k_gemv_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemv_kernel")(*args, **kwargs)
def q4k_gemv_partial_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemv_partial_kernel")(*args, **kwargs)
def q4k_gemv_warp_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_gemv_warp_kernel")(*args, **kwargs)
def q4k_coop_partial_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_coop_partial_kernel")(*args, **kwargs)
def q4k_q8_1_vdot_builtin_partial_kernel(*args, **kwargs):
  return _attr("extra.qk.quant.q4_k_gemv_primitive", "q4k_q8_1_vdot_builtin_partial_kernel")(*args, **kwargs)
def q8_signed_pack_u32_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q8_signed_pack_u32_kernel")(*args, **kwargs)
def q8_1_bias_pack_u32_kernel(*args, **kwargs): return _attr("extra.qk.quant.q4_k_gemv_primitive", "q8_1_bias_pack_u32_kernel")(*args, **kwargs)
def q8_1_quantize(*args, **kwargs): return _attr("extra.qk.layout", "q8_1_quantize")(*args, **kwargs)
def quantize_q4_k(*args, **kwargs): return _attr("extra.qk.quantize", "quantize_q4_k")(*args, **kwargs)

def should_route_q4k_lane_partition(*args, **kwargs): return _attr("extra.qk.bubblebeam_futuresight", "should_route_q4k_lane_partition")(*args, **kwargs)
def q4k_g3_lanemap_gemv_kernel(*args, **kwargs): return _attr("extra.qk.gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_kernel")(*args, **kwargs)
def q4k_g3_lanemap_gemv_splitk_kernel(*args, **kwargs):
  return _attr("extra.qk.gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_splitk_kernel")(*args, **kwargs)
def q4k_g3_lanemap_gemv_inkernel_combine_kernel(*args, **kwargs):
  return _attr("extra.qk.gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_inkernel_combine_kernel")(*args, **kwargs)
def q4k_lane_partition_gemv_kernel(*args, **kwargs): return _attr("extra.qk.q4k_lane_partition_gemv", "q4k_lane_partition_gemv_kernel")(*args, **kwargs)
def q4k_scheduler_matvec(*args, **kwargs): return _attr("extra.qk.q4k_scheduler_gemv", "q4k_scheduler_matvec")(*args, **kwargs)
def q4k_scheduler_matvec_wordlane(*args, **kwargs): return _attr("extra.qk.q4k_scheduler_gemv", "q4k_scheduler_matvec_wordlane")(*args, **kwargs)
def q4k_scheduler_matvec_lanemap(*args, **kwargs): return _attr("extra.qk.q4k_scheduler_gemv", "q4k_scheduler_matvec_lanemap")(*args, **kwargs)

def q6k_parse_opt(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "parse_opt")(*args, **kwargs)
def q6k_gemm_kernel(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemm_kernel")(*args, **kwargs)
def q6k_gemm_packed_load_kernel(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemm_packed_load_kernel")(*args, **kwargs)
def q6k_gemm_packed_load_direct_out_kernel(*args, **kwargs):
  return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemm_packed_load_direct_out_kernel")(*args, **kwargs)
def q6k_gemv_warp_kernel(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemv_warp_kernel")(*args, **kwargs)
def q6k_coop_partial_kernel(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_coop_partial_kernel")(*args, **kwargs)
def q6k_gemv_partial_kernel(*args, **kwargs): return _attr("extra.qk.quant.q6_k_gemv_primitive", "q6k_gemv_partial_kernel")(*args, **kwargs)
def q6k_spec_for_role(*args, **kwargs): return _attr("extra.qk.q6k_route_spec", "spec_for_role")(*args, **kwargs)
def emit_q6k_gemv_kernel(*args, **kwargs): return _attr("extra.qk.q6k_route_spec", "emit_q6k_gemv_kernel")(*args, **kwargs)

def flash_decode_attention(*args, **kwargs): return _attr("extra.qk.flash_decode", "flash_decode_attention")(*args, **kwargs)
def flash_decode_attention_whole_cache(*args, **kwargs): return _attr("extra.qk.flash_decode", "flash_decode_attention_whole_cache")(*args, **kwargs)
def flash_decode_g5_block_tile(*args, **kwargs): return _attr("extra.qk.flash_decode", "flash_decode_g5_block_tile")(*args, **kwargs)
def flash_decode_attention_kv_flat(*args, **kwargs): return _attr("extra.qk.flash_decode", "flash_decode_attention_kv_flat")(*args, **kwargs)
def flash_decode_live_split_block_tile(*args, **kwargs): return _attr("extra.qk.live_split_geometry", "flash_decode_live_split_block_tile")(*args, **kwargs)
def flash_decode_fused_combine(*args, **kwargs): return _attr("extra.qk.flash_decode_fused_combine", "flash_decode_fused_combine")(*args, **kwargs)

def assert_pure_machine_search(*args, **kwargs): return _attr("extra.qk.pure_search_guard", "assert_pure_machine_search")(*args, **kwargs)
