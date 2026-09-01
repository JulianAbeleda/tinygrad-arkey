"""Closed-default native Q6_K attention-V four-warp fp16 MMVQ admission candidate.

The installed Q6 attention-V route emits one ``q6k_gen_partial_1024_4096_4``
parts kernel per block and folds the parts reduce into the decode kv-store.
This candidate swaps it 1:1 for a 128-thread four-warp fp16 direct consumer
(the same geometry that landed on Q4/Q6 FFN-down): one pass to a contiguous
fp32[1024] output, no parts buffer, no provider node.  It is deliberately
unreachable unless a research harness attaches an explicit
``Q6KVFourWarpAdmission`` to one concrete Q6_K attention-V linear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (
  Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, Q6K_POS_EXTENT, _q6k_block_dot)
from tinygrad.llm.kernel_program import (
  KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program)
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.boltbeam_authority import tickets_for_candidate

ROWS, K = 1024, 4096
WARP, WARPS_PER_ROW, POS = 32, 4, Q6K_POS_EXTENT
K_BLOCKS = K // Q6_K_BLOCK_ELEMS              # 16
BLOCKS_PER_WARP = K_BLOCKS // WARPS_PER_ROW    # 4
BLOCKS_PER_SUB = BLOCKS_PER_WARP // 2         # 2


@dataclass(frozen=True)
class Q6KVFourWarpAdmission:
  block_index: int
  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q6_K attention-V four-warp block index must be a non-negative integer")


def emit_q6k_v_four_warp_fp16_direct() -> callable:
  """Q6_K attention-V four-warp fp16-direct consumer (128 threads/row, no Q8)."""
  def kernel(out:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.special(ROWS, "gidx0")
    lid = UOp.special(WARP * WARPS_PER_ROW, "lidx0")
    warp, lane = lid // WARP, lid % WARP
    sub, pos = lane // POS, lane % POS

    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    blk = UOp.range(BLOCKS_PER_SUB, 0, axis_type=AxisType.REDUCE)
    block = warp * BLOCKS_PER_WARP + sub * BLOCKS_PER_SUB + blk
    base = (row * K_BLOCKS + block) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, block, pos)
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = acc[0]

    for slot, offset in enumerate((16, 8, 4, 2, 1), 90):
      total = total + _staged_shfl(total, offset, lane, slot)
    smem = UOp.placeholder((WARPS_PER_ROW,), dtypes.float32, 40, addrspace=AddrSpace.LOCAL)
    published = smem[warp].store(total, lane.eq(0))
    ready = UOp.barrier(UOp.group(published))
    merged = UOp.const(dtypes.float32, 0.0)
    for wi in range(WARPS_PER_ROW):
      merged = merged + smem.after(ready)[wi]
    return out[row].store(merged, lid.eq(0)).sink(
      arg=KernelInfo(name="q6k_v_four_warp_fp16_direct_1024_4096", opts_to_apply=()))
  return kernel


def q6k_v_four_warp_call(admission:object, linear:Any, x:Tensor, binding:Any) -> Tensor|None:
  """Return the leased candidate, or None without changing the installed path."""
  if not isinstance(admission, Q6KVFourWarpAdmission): return None
  capability = getattr(getattr(linear, "route_admission", None), "capability", None)
  if (getattr(capability, "backend", None), getattr(capability, "architecture", None)) != ("NV", "sm_120"): return None
  if (getattr(linear, "route_role", None), getattr(binding, "N", None), getattr(binding, "K", None)) != ("attn_kv", ROWS, K): return None
  if getattr(linear, "bias", None) is not None or not str(x.device).startswith("NV"): return None
  halfs = linear.q6k_storage.halfs.to(x.device)
  xv = x[:, 0, :].reshape(K).cast(dtypes.float16).contiguous()
  consumer = KernelProgram("decode_q6k_v_four_warp", f"blk{admission.block_index}.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q6k_v_four_warp_fp16_direct(),
    output_spec=OutputSpec((ROWS,), dtypes.float32),
    boltbeam_ticket=tickets_for_candidate({"family":"q6_v.v1"},
      (("decode_q6k_v_four_warp_fp16_geometry","q6_v_four_warp_fp16"),)))
  out = execute_promoted_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=x.device),
    halfs, xv, program=consumer)
  return out.reshape(1, 1, ROWS)


__all__ = ["Q6KVFourWarpAdmission", "ROWS", "K", "emit_q6k_v_four_warp_fp16_direct",
           "q6k_v_four_warp_call"]
