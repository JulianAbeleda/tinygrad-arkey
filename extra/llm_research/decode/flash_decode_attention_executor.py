#!/usr/bin/env python3
"""Tensor-level execution for descriptor-backed live-split flash decode."""
from __future__ import annotations

from tinygrad import Tensor, dtypes
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program

from extra.llm_research.decode.flash_decode_attention_spec import describe_flash_decode_attention


def flash_decode_live_split_block_tile(q, cache_kv, Tc_u, Hd: int, Hq: int, Hkv: int, MAXC: int, S: int,
                                       staging: str = "K_ONLY", fused_combine: bool = True, kv_scale=None, freqs=None,
                                       query_group_size: int | None = None, stage_width: int = 1):
  """Execute generated block-tile flash decode with live-context split geometry and return ``[Hq, Hd]``."""
  W2 = Hd + 2
  q_f = q.reshape(Hq * Hd)
  # KV-quant long-context tier dequantizes in-register; rope-at-read rotates un-roped K from freqs in-register.
  quant, rope = kv_scale is not None, freqs is not None
  inputs = (q_f, cache_kv) + ((kv_scale,) if quant else ()) + ((freqs,) if rope else ())
  spec = describe_flash_decode_attention(Hq=Hq, Hd=Hd, Hkv=Hkv, MAXC=MAXC, S=S, staging=staging,
                                         quant=quant, rope=rope, query_group_size=query_group_size, stage_width=stage_width)
  po = execute_research_program(Tensor.empty(Hq * S * W2, dtype=dtypes.float32), *inputs, program=KernelProgram(
    "dev.flash_decode_legacy", f"tile.hq{Hq}.hkv{Hkv}.hd{Hd}.s{S}", KernelProgramProvenance.RESEARCH_ONLY,
    spec.emit_tile(Tc_u)))
  # The old two-kernel combine was removed 2026-07-06; preserve the fail-loud contract for stale callers.
  if not fused_combine:
    raise ValueError("fused_combine=False is no longer supported for decode live-split routes")
  out = execute_research_program(Tensor.empty(Hq * Hd, dtype=dtypes.float32), po, program=KernelProgram(
    "dev.flash_decode_legacy", f"combine.hq{Hq}.hkv{Hkv}.hd{Hd}.s{S}", KernelProgramProvenance.RESEARCH_ONLY,
    spec.emit_combine()))
  return out.reshape(Hq, Hd)
