"""Closed-default native Q6_K FFN-down four-warp fp16 MMVQ route.

This is the Q6 analog of the landed Q4 four-warp fp16 geometry promotion
(``q4k_ffn_down_mmvq.py`` / commit ``765f03f30``).  It swaps the installed
row_tile-2 coop consumer (16 threads/row) for a 128-thread, four-warp fp16
direct consumer with the same Q6 dequant arithmetic, no Q8 provider node, and
the M2b residual add absorbed in-kernel.  The mechanism is the occupancy/DRAM
geometry fix measured in
``docs/task_workflow/input/nv-q6-ffn-down-four-warp-fp16-microgate-20260815.md``
(device time 25.7 us vs 31.0 us control, below llama's 28.75 us on this node).

The route is deliberately unreachable unless a promoted policy attaches an
explicit ``Q6KFFNDownMMVQAdmission`` to a concrete Q6_K FFN-down linear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, _q6k_block_dot
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance,
  OutputSpec, ResidualViewRequest, TypedLayout, TypedViewRequest, execute_promoted_program)
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

ROWS, K = 4096, 12288
WARP, WARPS_PER_ROW, POS = 32, 4, 16
K_BLOCKS = K // Q6_K_BLOCK_ELEMS            # 48
BLOCKS_PER_WARP = K_BLOCKS // WARPS_PER_ROW # 12
BLOCKS_PER_SUB = BLOCKS_PER_WARP // 2      # 6


@dataclass(frozen=True)
class Q6KFFNDownMMVQAdmission:
  block_index: int
  fp16_fma: bool = True
  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q6_K FFN-down MMVQ block index must be a non-negative integer")
    if not isinstance(self.fp16_fma, bool):
      raise ValueError("fp16_fma must be bool")


def emit_q6k_four_warp_fp16_direct() -> callable:
  """Four-warp fp16-direct Q6_K FFN-down consumer (128 threads/row, no Q8 provider)."""
  def kernel(out:UOp, halfs:UOp, x:UOp, h:UOp) -> UOp:
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
    result = merged + h[row].cast(dtypes.float32)
    return out[row].store(result, lid.eq(0)).sink(
      arg=KernelInfo(name="q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd", opts_to_apply=()))
  return kernel


def q6k_ffn_down_mmvq_call(admission:object, linear:Any, x:Tensor, binding:Any,
                            epilogue_inputs:dict[str,Tensor]) -> Tensor|None:
  """Return the promoted four-warp Q6 consumer, or None without changing the installed path."""
  if not isinstance(admission, Q6KFFNDownMMVQAdmission): return None
  capability = getattr(getattr(linear, "route_admission", None), "capability", None)
  if (getattr(capability, "backend", None), getattr(capability, "architecture", None)) != ("NV", "sm_120"):
    return None
  if (getattr(linear, "route_role", None), binding.N, binding.K) != ("ffn_down", ROWS, K): return None
  if getattr(linear, "bias", None) is not None or not str(x.device).startswith("NV"): return None
  if any(key != "normed_h" for key in epilogue_inputs): return None

  xv = x[:, 0, :].reshape(K).cast(dtypes.float16).contiguous()
  residual = epilogue_inputs["normed_h"][:, 0, :].reshape(ROWS).cast(dtypes.float32)
  consumer = KernelProgram("decode_q6k_ffn_down_mmvq", f"blk{admission.block_index}.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q6k_four_warp_fp16_direct(),
    output_spec=OutputSpec((ROWS,), dtypes.float32,
      typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32, (ROWS,), (1, 1, ROWS)),
        combine_fusion_admitted=False, epilogue_absorption_admitted=True)),
    typed_input_views=(TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(K,), route_role="ffn_down",
      requires_combine_fusion=False, requires_epilogue_absorption=True),),
    residual_input_views=(ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(ROWS,),
      route_role="ffn_down", kind="residual_add"),))
  out = execute_promoted_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=x.device),
    linear.q6k_storage.halfs.to(x.device), xv, residual, program=consumer)
  return out.reshape(1, 1, ROWS)


__all__ = ["Q6KFFNDownMMVQAdmission", "ROWS", "K", "emit_q6k_four_warp_fp16_direct",
           "q6k_ffn_down_mmvq_call"]
