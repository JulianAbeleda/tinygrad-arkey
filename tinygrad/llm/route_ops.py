from __future__ import annotations

import importlib
from functools import cache


@cache
def _attr(module:str, name:str):
  return getattr(importlib.import_module(module), name)


def route_pf16_graph_gemm(*args, **kwargs): return _attr("extra.qk_prefill_graph_gemm_route", "route_pf16_graph_gemm")(*args, **kwargs)
def route_pf16(*args, **kwargs): return _attr("extra.qk_tensile_inmodel", "route_pf16")(*args, **kwargs)
def route_pf16_col(*args, **kwargs): return _attr("extra.qk_tensile_inmodel", "route_pf16_col")(*args, **kwargs)
def install_tensile(*args, **kwargs): return _attr("extra.qk_tensile_inmodel", "install")(*args, **kwargs)

def q4k_parse_opt(*args, **kwargs): return _attr("extra.q4_k_gemv_primitive", "parse_opt")(*args, **kwargs)
def q4k_gemm_kernel(*args, **kwargs): return _attr("extra.q4_k_gemv_primitive", "q4k_gemm_kernel")(*args, **kwargs)
def q4k_gemv_kernel(*args, **kwargs): return _attr("extra.q4_k_gemv_primitive", "q4k_gemv_kernel")(*args, **kwargs)
def q4k_gemv_partial_kernel(*args, **kwargs): return _attr("extra.q4_k_gemv_primitive", "q4k_gemv_partial_kernel")(*args, **kwargs)
def q4k_gemv_warp_kernel(*args, **kwargs): return _attr("extra.q4_k_gemv_primitive", "q4k_gemv_warp_kernel")(*args, **kwargs)
def q4k_coop_partial_kernel(*args, **kwargs): return _attr("extra.q4_k_gemv_primitive", "q4k_coop_partial_kernel")(*args, **kwargs)
def q4k_q8_1_vdot_builtin_partial_kernel(*args, **kwargs):
  return _attr("extra.q4_k_gemv_primitive", "q4k_q8_1_vdot_builtin_partial_kernel")(*args, **kwargs)
def q8_1_bias_pack_u32_kernel(*args, **kwargs): return _attr("extra.q4_k_gemv_primitive", "q8_1_bias_pack_u32_kernel")(*args, **kwargs)
def q8_1_quantize(*args, **kwargs): return _attr("extra.qk_layout", "q8_1_quantize")(*args, **kwargs)
def quantize_q4_k(*args, **kwargs): return _attr("extra.qk_quantize", "quantize_q4_k")(*args, **kwargs)

def should_route_q4k_lane_partition(*args, **kwargs): return _attr("extra.qk_bubblebeam_futuresight", "should_route_q4k_lane_partition")(*args, **kwargs)
def q4k_g3_lanemap_gemv_kernel(*args, **kwargs): return _attr("extra.qk_gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_kernel")(*args, **kwargs)
def q4k_g3_lanemap_gemv_splitk_kernel(*args, **kwargs):
  return _attr("extra.qk_gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_splitk_kernel")(*args, **kwargs)
def q4k_g3_lanemap_gemv_inkernel_combine_kernel(*args, **kwargs):
  return _attr("extra.qk_gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_inkernel_combine_kernel")(*args, **kwargs)
def q4k_lane_partition_gemv_kernel(*args, **kwargs): return _attr("extra.qk_q4k_lane_partition_gemv", "q4k_lane_partition_gemv_kernel")(*args, **kwargs)
def q4k_scheduler_matvec(*args, **kwargs): return _attr("extra.qk_q4k_scheduler_gemv", "q4k_scheduler_matvec")(*args, **kwargs)
def q4k_scheduler_matvec_wordlane(*args, **kwargs): return _attr("extra.qk_q4k_scheduler_gemv", "q4k_scheduler_matvec_wordlane")(*args, **kwargs)
def q4k_scheduler_matvec_lanemap(*args, **kwargs): return _attr("extra.qk_q4k_scheduler_gemv", "q4k_scheduler_matvec_lanemap")(*args, **kwargs)

def q6k_parse_opt(*args, **kwargs): return _attr("extra.q6_k_gemv_primitive", "parse_opt")(*args, **kwargs)
def q6k_gemm_kernel(*args, **kwargs): return _attr("extra.q6_k_gemv_primitive", "q6k_gemm_kernel")(*args, **kwargs)
def q6k_gemv_warp_kernel(*args, **kwargs): return _attr("extra.q6_k_gemv_primitive", "q6k_gemv_warp_kernel")(*args, **kwargs)
def q6k_halfwarp_partition_kernel(*args, **kwargs): return _attr("extra.q6_k_gemv_primitive", "q6k_halfwarp_partition_kernel")(*args, **kwargs)
def q6k_coop_partial_kernel(*args, **kwargs): return _attr("extra.q6_k_gemv_primitive", "q6k_coop_partial_kernel")(*args, **kwargs)
def q6k_gemv_partial_kernel(*args, **kwargs): return _attr("extra.q6_k_gemv_primitive", "q6k_gemv_partial_kernel")(*args, **kwargs)
def q6k_spec_for_role(*args, **kwargs): return _attr("extra.qk_q6k_route_spec", "spec_for_role")(*args, **kwargs)
def emit_q6k_gemv_kernel(*args, **kwargs): return _attr("extra.qk_q6k_route_spec", "emit_q6k_gemv_kernel")(*args, **kwargs)

def route_q8_ffn(*args, **kwargs): return _attr("extra.q8_ffn_graph_route", "route_q8_ffn")(*args, **kwargs)
def install_q8_ffn_artifacts(*args, **kwargs): return _attr("extra.q8_ffn_graph_route", "install_q8_ffn_artifacts")(*args, **kwargs)
def imported_q4_mmvq_q8_bytes() -> int: return _attr("extra.qk_decode_mmvq_graph_route", "Q8_BYTES")
def route_imported_q4_mmvq(*args, **kwargs): return _attr("extra.qk_decode_mmvq_graph_route", "route_imported_q4_mmvq")(*args, **kwargs)

def flash_decode_attention(*args, **kwargs): return _attr("extra.qk_flash_decode", "flash_decode_attention")(*args, **kwargs)
def flash_decode_attention_whole_cache(*args, **kwargs): return _attr("extra.qk_flash_decode", "flash_decode_attention_whole_cache")(*args, **kwargs)
def flash_decode_g5_block_tile(*args, **kwargs): return _attr("extra.qk_flash_decode", "flash_decode_g5_block_tile")(*args, **kwargs)
def flash_decode_attention_kv_flat(*args, **kwargs): return _attr("extra.qk_flash_decode", "flash_decode_attention_kv_flat")(*args, **kwargs)
def flash_decode_live_split_block_tile(*args, **kwargs): return _attr("extra.qk_live_split_geometry", "flash_decode_live_split_block_tile")(*args, **kwargs)
def flash_decode_fused_combine(*args, **kwargs): return _attr("extra.qk_flash_decode_fused_combine", "flash_decode_fused_combine")(*args, **kwargs)
def amdgcn_flash_decode(*args, **kwargs): return _attr("extra.qk_owned_flash_decode_graph_node", "amdgcn_flash_decode")(*args, **kwargs)

def assert_pure_machine_search(*args, **kwargs): return _attr("extra.qk_pure_search_guard", "assert_pure_machine_search")(*args, **kwargs)
