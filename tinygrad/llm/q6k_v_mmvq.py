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
from tinygrad.llm.boltbeam_authority import lower_authorized_candidate

ROWS, K = 1024, 4096
WARP, WARPS_PER_ROW, POS = 32, 4, Q6K_POS_EXTENT

SUBS = WARP // POS  # each warp holds WARP // POS sub-groups of POS lanes, one block row-slice each


def validate_four_warp(rows:int, k:int, warps_per_row:int=WARPS_PER_ROW) -> None:
  """The real limits of the four-warp Q6_K emitter: whole 256-element blocks, split evenly over the row's warps and
  each warp's sub-groups, and a CTA of warps_per_row warps."""
  if not isinstance(rows, int) or rows < 1: raise ValueError(f"Q6_K four-warp GEMV requires rows >= 1, got {rows!r}")
  if not isinstance(warps_per_row, int) or warps_per_row < 1: raise ValueError(f"warps_per_row must be >= 1, got {warps_per_row!r}")
  if not isinstance(k, int) or k < Q6_K_BLOCK_ELEMS or k % Q6_K_BLOCK_ELEMS:
    raise ValueError(f"Q6_K four-warp GEMV requires k a positive multiple of {Q6_K_BLOCK_ELEMS}, got {k!r}")
  if (k // Q6_K_BLOCK_ELEMS) % (warps_per_row * SUBS):
    raise ValueError(f"Q6_K four-warp GEMV splits k's {k // Q6_K_BLOCK_ELEMS} blocks over {warps_per_row} warps x {SUBS} "
                     f"sub-groups: they must divide evenly")


def validate_shape(rows:int, k:int) -> None:
  """The shape authority the loader's route gate asks: the emitter's real limits. The shapes the route was measured
  at live in its record (decode-q6k-v-four-warp-fp16-geometry-route-policy.json, measured_models)."""
  validate_four_warp(rows, k)


@dataclass(frozen=True)
class Q6KVFourWarpAdmission:
  block_index: int
  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q6_K attention-V four-warp block index must be a non-negative integer")


def emit_q6k_v_four_warp_fp16_direct(*, rows:int=ROWS, k:int=K, warps_per_row:int=WARPS_PER_ROW,
                                     kernel_name:str|None=None) -> callable:
  """Q6_K multi-warp fp16-direct GEMV (warps_per_row x 32 threads a row, no Q8). The default is the attention-V
  geometry it was measured at; any (rows, k, warps_per_row) that validate_four_warp accepts lowers."""
  validate_four_warp(rows, k, warps_per_row)
  WARPS = warps_per_row
  K_BLOCKS = k // Q6_K_BLOCK_ELEMS
  BLOCKS_PER_WARP = K_BLOCKS // WARPS
  BLOCKS_PER_SUB = BLOCKS_PER_WARP // SUBS
  wsuffix = "" if WARPS == WARPS_PER_ROW else f"_w{WARPS}"
  name = kernel_name or f"q6k_v_four_warp_fp16_direct_{rows}_{k}{wsuffix}"
  def kernel(out:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0")
    lid = UOp.special(WARP * WARPS, "lidx0")
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
    smem = UOp.placeholder((WARPS,), dtypes.float32, 40, addrspace=AddrSpace.LOCAL)
    published = smem[warp].store(total, lane.eq(0))
    ready = UOp.barrier(UOp.group(published))
    merged = UOp.const(dtypes.float32, 0.0)
    for wi in range(WARPS):
      merged = merged + smem.after(ready)[wi]
    return out[row].store(merged, lid.eq(0)).sink(
      arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def q6k_v_four_warp_call(admission:object, linear:Any, x:Tensor, binding:Any) -> Tensor|None:
  """Return the leased candidate, or None without changing the installed path."""
  if not isinstance(admission, Q6KVFourWarpAdmission): return None
  capability = getattr(getattr(linear, "route_admission", None), "capability", None)
  if (getattr(capability, "backend", None), getattr(capability, "architecture", None)) != ("NV", "sm_120"): return None
  rows, k = getattr(binding, "N", None), getattr(binding, "K", None)
  if getattr(linear, "route_role", None) != "attn_kv": return None
  try: validate_four_warp(rows, k)
  except ValueError: return None
  if getattr(linear, "bias", None) is not None or not str(x.device).startswith("NV"): return None
  halfs = linear.q6k_storage.halfs.to(x.device)
  xv = x[:, 0, :].reshape(k).cast(dtypes.float16).contiguous()
  emitter,ticket=lower_authorized_candidate({"family":"q6_v.v1","rows":rows,"k":k},
    (("decode_q6k_v_four_warp_fp16_geometry","q6_v_four_warp_fp16"),))
  consumer = KernelProgram("decode_q6k_v_four_warp", f"blk{admission.block_index}.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter,
    output_spec=OutputSpec((rows,), dtypes.float32),
    boltbeam_ticket=ticket)
  out = execute_promoted_program(Tensor.empty((rows,), dtype=dtypes.float32, device=x.device),
    halfs, xv, program=consumer)
  return out.reshape(1, 1, rows)


__all__ = ["Q6KVFourWarpAdmission", "ROWS", "K", "emit_q6k_v_four_warp_fp16_direct",
           "q6k_v_four_warp_call"]
