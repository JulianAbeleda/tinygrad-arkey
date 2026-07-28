from __future__ import annotations

import importlib
from functools import cache


@cache
def _attr(module:str, name:str):
  return getattr(importlib.import_module(module), name)

# Keep this boundary limited to route entrypoints with current production callers.
__all__ = [
  "automatic_promoted_prefill_graph_policy", "install_memory_adaptive_model_adapters", "route_pf16_graph_gemm",
  "describe_q4k_packed_prefill_generated", "emit_q4k_packed_prefill_kernel", "q4k_g3_lanemap_gemv_kernel",
  "q6k_spec_for_role", "emit_q6k_gemv_kernel", "q6k_vocab_scalar_reduce_eligible",
  "emit_q6k_vocab_scalar_reduce_kernel", "describe_q6k_packed_prefill", "emit_q6k_packed_prefill_kernel",
  "flash_decode_live_split_block_tile", "select_packed_wmma_prefill_candidate", "build_packed_wmma_warmstart_tables",
]

def automatic_promoted_prefill_graph_policy(*args, **kwargs):
  return _attr("extra.llm_research.route_manifest", "automatic_promoted_prefill_graph_policy")(*args, **kwargs)
def install_memory_adaptive_model_adapters():
  return _attr("extra.llm_research.memory_adaptive_runtime_collector", "install_model_adapters")()


def route_pf16_graph_gemm(*args, **kwargs): return _attr("extra.llm_research.prefill.prefill_graph_gemm_route", "route_pf16_graph_gemm")(*args, **kwargs)

def describe_q4k_packed_prefill_generated(*args, **kwargs): return _attr("extra.llm_research.prefill.q4k_prefill_route_spec", "describe_q4k_packed_prefill")(*args, **kwargs)
def emit_q4k_packed_prefill_kernel(*args, **kwargs):
  return _attr("extra.llm_research.prefill.q4k_prefill_route_spec", "emit_q4k_packed_prefill_kernel")(*args, **kwargs)
def q4k_g3_lanemap_gemv_kernel(*args, **kwargs): return _attr("extra.llm_research.gemv_g3_codegen_lowering", "q4k_g3_lanemap_gemv_kernel")(*args, **kwargs)

def q6k_spec_for_role(*args, **kwargs): return _attr("extra.llm_research.q6k_route_spec", "spec_for_role")(*args, **kwargs)
def emit_q6k_gemv_kernel(*args, **kwargs): return _attr("extra.llm_research.q6k_route_spec", "emit_q6k_gemv_kernel")(*args, **kwargs)
def q6k_vocab_scalar_reduce_eligible(*args, **kwargs):
  return _attr("extra.llm_research.q6k_route_spec", "q6k_vocab_scalar_reduce_eligible")(*args, **kwargs)
def emit_q6k_vocab_scalar_reduce_kernel(*args, **kwargs):
  return _attr("extra.llm_research.q6k_route_spec", "emit_q6k_vocab_scalar_reduce_kernel")(*args, **kwargs)
def describe_q6k_packed_prefill(*args, **kwargs): return _attr("extra.llm_research.prefill.q6k_prefill_route_spec", "describe_q6k_packed_prefill")(*args, **kwargs)
def emit_q6k_packed_prefill_kernel(*args, **kwargs):
  return _attr("extra.llm_research.prefill.q6k_prefill_route_spec", "emit_q6k_packed_prefill_kernel")(*args, **kwargs)

def flash_decode_live_split_block_tile(*args, **kwargs):
  return _attr("extra.llm_research.decode.flash_decode_attention_executor", "flash_decode_live_split_block_tile")(*args, **kwargs)

# Boundary adapters (test/unit/test_tinygrad_boundary.py): tinygrad/ must not import extra.llm_research directly.
# The flash-prefill descriptor was promoted to tinygrad.schedule.wmma.flash_prefill; the remaining
# adapters are tracked in the production reorganization packet and will move in bounded slices.
def select_packed_wmma_prefill_candidate(*args, **kwargs):
  return _attr("extra.llm_research.prefill.packed_wmma_prefill_candidates", "select_packed_wmma_prefill_candidate")(*args, **kwargs)
def build_packed_wmma_warmstart_tables(*args, **kwargs):
  return _attr("extra.llm_research.prefill.packed_wmma_prefill_candidates", "build_packed_wmma_warmstart_tables")(*args, **kwargs)
