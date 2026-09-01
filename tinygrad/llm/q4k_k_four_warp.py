"""Closed-default native Q4_K attention-K four-warp exact decode candidate.

The installed Q4 attention-K route emits one
``q4k_g3_lanemap_gemv_1024_4096`` kernel per block with a single warp per
1024-row output.  This candidate swaps it 1:1 for the already-measured exact
group-factorized four-warp consumer: 128 threads per output row, no Q8
provider node, and one contiguous fp32[1024] output.  It is deliberately
unreachable unless a research harness attaches an explicit
``Q4KKFourWarpAdmission`` to one concrete Q4_K attention-K linear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.llm.kernel_program import (
  KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program)
from extra.llm_research.boltbeam_authority import tickets_for_candidate

ROWS, K = 1024, 4096


@dataclass(frozen=True)
class Q4KKFourWarpAdmission:
  block_index: int
  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q4_K attention-K four-warp block index must be a non-negative integer")


def q4k_k_four_warp_call(admission: object, linear: Any, x: Tensor, binding: Any) -> Tensor | None:
  """Return the leased candidate, or None without changing the installed path."""
  if not isinstance(admission, Q4KKFourWarpAdmission): return None
  capability = getattr(getattr(linear, "route_admission", None), "capability", None)
  if (getattr(capability, "backend", None), getattr(capability, "architecture", None)) != ("NV", "sm_120"): return None
  if (getattr(linear, "route_role", None), getattr(binding, "N", None), getattr(binding, "K", None)) != ("attn_kv", ROWS, K): return None
  if getattr(linear, "bias", None) is not None or not str(x.device).startswith("NV"): return None
  from extra.llm_research.decode.q4k_exact_group_factorized import emit_q4k_exact_four_warp
  words = linear.q4k_storage.words.to(x.device)
  xv = x[:, 0, :].reshape(K).cast(dtypes.float16).contiguous()
  consumer = KernelProgram("decode_q4k_k_four_warp", f"blk{admission.block_index}.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q4k_exact_four_warp(ROWS, K),
    output_spec=OutputSpec((ROWS,), dtypes.float32),
    boltbeam_ticket=tickets_for_candidate({"family":"q4_k_four_warp.v1","rows":ROWS,"k":K},
      (("decode_q4k_k_four_warp","q4_k_four_warp"),)))
  out = execute_promoted_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=x.device),
    words, xv, program=consumer)
  return out.reshape(1, 1, ROWS)


__all__ = ["Q4KKFourWarpAdmission", "ROWS", "K", "q4k_k_four_warp_call"]
