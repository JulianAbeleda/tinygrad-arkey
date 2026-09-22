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
import os
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (
  Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, _f16_half, _half4_lane, _q6k_block_dot, _q6k_byte)
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance,
  OutputSpec, ResidualViewRequest, TypedLayout, TypedViewRequest, execute_promoted_program)
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tinygrad.llm.boltbeam_authority import lower_authorized_candidate

ROWS, K = 4096, 12288
WARP, WARPS_PER_ROW, POS = 32, 4, 16
K_BLOCKS = K // Q6_K_BLOCK_ELEMS            # 48
BLOCKS_PER_WARP = K_BLOCKS // WARPS_PER_ROW # 12
BLOCKS_PER_SUB = BLOCKS_PER_WARP // 2      # 6


@dataclass(frozen=True)
class Q6KFFNDownMMVQAdmission:
  block_index: int
  fp16_fma: bool = True
  rows_per_block: int = 1
  packed_lanemap: bool = False
  unroll_blocks: int|None = None
  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q6_K FFN-down MMVQ block index must be a non-negative integer")
    if not isinstance(self.fp16_fma, bool):
      raise ValueError("fp16_fma must be bool")
    if not isinstance(self.rows_per_block, int) or isinstance(self.rows_per_block, bool) or self.rows_per_block not in (1, 2, 4, 8):
      raise ValueError("rows_per_block must be one of 1, 2, 4, or 8")
    if not isinstance(self.packed_lanemap, bool):
      raise ValueError("packed_lanemap must be bool")
    if self.packed_lanemap and self.rows_per_block != 1:
      raise ValueError("packed_lanemap is admitted only for rows_per_block=1")
    if self.unroll_blocks not in (None, 2, 3, 4, 6, 12):
      raise ValueError("unroll_blocks must divide the 12-block packed-lane loop")
    if self.unroll_blocks is not None and not self.packed_lanemap:
      raise ValueError("unroll_blocks requires packed_lanemap=True")


def _i8f(v:UOp) -> UOp: return v.cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float32)


def _q6k_block_dot_packed_lanemap(halfs:UOp, x:UOp, base:UOp, x_block:UOp, lane:UOp,
                                  data_halfs:UOp|None=None) -> UOp:
  """Q6_K block dot with llama's packed lane ownership and an fp16 activation.

  Each lane decodes two packed int8x4-shaped qwords and reads the matching two
  half4 activation spans. The Q6 block remains scalar-halfword addressed
  because its 210-byte stride misaligns every other block; the same-byte win
  comes from packed ownership and load deduplication, not an unsafe wide cast.
  """
  # A research split may address the one-touch ql/qh payload through a second alias while keeping
  # reused scale/d metadata on the ordinary-caching pointer. Both aliases name the same immutable
  # allocation; the split exists solely so the renderer can apply a load policy by reuse class.
  qhalfs = halfs if data_halfs is None else data_halfs
  vl = qhalfs[base + lane * 2].cast(dtypes.uint32).bitwise_or(
    qhalfs[base + lane * 2 + 1].cast(dtypes.uint32).lshift(16))
  qh_half = 16 * (lane // 16) + 2 * (lane % 8)
  vh = qhalfs[base + 64 + qh_half].cast(dtypes.uint32).bitwise_or(
    qhalfs[base + 65 + qh_half].cast(dtypes.uint32).lshift(16))
  vh = vh.rshift(2 * ((lane % 16) // 8))
  scale_idx = 8 * (lane // 16) + (lane % 16) // 4
  x_group0 = 4 * (lane // 16) + (lane % 16) // 8
  d = _f16_half(halfs[base + 104])
  contribution = UOp.const(dtypes.float32, 0.0)
  for term in range(2):
    low = vl.rshift(4 * term).bitwise_and(0x0F0F0F0F)
    high = vh.rshift(4 * term).lshift(4).bitwise_and(0x30303030)
    qword = low.bitwise_or(high)
    scale = _i8f(_q6k_byte(halfs, base, 192 + scale_idx + 4 * term))
    x_group = x_group0 + 2 * term
    xv = x.index(x_block * Q6_K_BLOCK_ELEMS + x_group * 32 + (lane % 8) * 4).load(dtype=dtypes.float16.vec(4))
    for nib in range(4):
      q = qword.rshift(nib * 8).bitwise_and(0xff).cast(dtypes.float32) - 32.0
      contribution = contribution + (d * q * scale) * _half4_lane(xv, nib)
  return contribution


def emit_q6k_four_warp_fp16_direct(*, rows_per_block:int=1, packed_lanemap:bool=False,
                                   unroll_blocks:int|None=None, split_weight_stream:bool=False,
                                   research_name_suffix:str="") -> callable:
  """Four-warp fp16-direct Q6_K FFN-down consumer (128 threads/row, no Q8 provider)."""
  if rows_per_block not in (1, 2, 4, 8): raise ValueError(f"unsupported Q6 FFN-down rows_per_block {rows_per_block}")
  if not isinstance(packed_lanemap, bool): raise ValueError("packed_lanemap must be bool")
  if packed_lanemap and rows_per_block != 1: raise ValueError("packed_lanemap requires rows_per_block=1")
  if unroll_blocks not in (None, 2, 3, 4, 6, 12): raise ValueError("unroll_blocks must divide the 12-block packed-lane loop")
  if unroll_blocks is not None and not packed_lanemap: raise ValueError("unroll_blocks requires packed_lanemap=True")
  if split_weight_stream and not packed_lanemap: raise ValueError("split_weight_stream requires packed_lanemap=True")
  def build(out:UOp, halfs:UOp, x:UOp, h:UOp, data_halfs:UOp|None=None) -> UOp:
    row_group = UOp.special(ROWS // rows_per_block, "gidx0")
    lid = UOp.special(WARP * WARPS_PER_ROW * rows_per_block, "lidx0")
    row_in_block, row_lid = lid // (WARP * WARPS_PER_ROW), lid % (WARP * WARPS_PER_ROW)
    row = row_group * rows_per_block + row_in_block
    warp, lane = row_lid // WARP, row_lid % WARP

    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    outer = UOp.range((BLOCKS_PER_WARP // unroll_blocks) if unroll_blocks is not None else
                      (BLOCKS_PER_WARP if packed_lanemap else BLOCKS_PER_SUB), 0, axis_type=AxisType.REDUCE)
    blocks = [outer * unroll_blocks + j for j in range(unroll_blocks)] if unroll_blocks is not None else [outer]
    contribs = []
    for blk in blocks:
      block = (warp * BLOCKS_PER_WARP + blk if packed_lanemap else
               warp * BLOCKS_PER_WARP + (lane // POS) * BLOCKS_PER_SUB + blk)
      base = (row * K_BLOCKS + block) * Q6K_HALFWORDS_PER_BLOCK
      contribs.append(_q6k_block_dot_packed_lanemap(halfs, x, base, block, lane, data_halfs) if packed_lanemap else
                      _q6k_block_dot(halfs, x, base, block, lane % POS))
    next_acc = acc.after(outer)[0]
    for contrib in contribs: next_acc = next_acc + contrib
    acc = acc.after(acc[0].store(next_acc).end(outer))
    total = acc[0]

    for slot, offset in enumerate((16, 8, 4, 2, 1), 90):
      total = total + _staged_shfl(total, offset, lane, slot)
    smem = UOp.placeholder((WARPS_PER_ROW * rows_per_block,), dtypes.float32, 40, addrspace=AddrSpace.LOCAL)
    published = smem[row_in_block * WARPS_PER_ROW + warp].store(total, lane.eq(0))
    ready = UOp.barrier(UOp.group(published))
    merged = UOp.const(dtypes.float32, 0.0)
    for wi in range(WARPS_PER_ROW):
      merged = merged + smem.after(ready)[row_in_block * WARPS_PER_ROW + wi]
    result = merged + h[row].cast(dtypes.float32)
    name = (f"q6k_fp16_packed_lanemap_u{unroll_blocks}_4096_12288_epi_ffnresadd" if unroll_blocks is not None else
            "q6k_fp16_packed_lanemap_4096_12288_epi_ffnresadd" if packed_lanemap else
            "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd" if rows_per_block == 1 else
            f"q6k_fp16_mmvq_direct_rpb{rows_per_block}_4096_12288_epi_ffnresadd")
    if split_weight_stream: name += "_splitstream"
    name += research_name_suffix
    return out[row].store(result, row_lid.eq(0)).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  if split_weight_stream:
    def kernel_split(out:UOp, halfs:UOp, data_halfs:UOp, x:UOp, h:UOp) -> UOp:
      return build(out, halfs, x, h, data_halfs)
    return kernel_split
  def kernel(out:UOp, halfs:UOp, x:UOp, h:UOp) -> UOp: return build(out, halfs, x, h)
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
  split_weight_stream = os.environ.get("NV_Q6_FFN_DOWN_SPLIT_WEIGHT_STREAM", "0") == "1"
  authorities = (("decode_q6k_ffn_down_fp16_geometry", "q6_ffn_down_fp16"),
                 ("decode_ffn_down_resadd", "q6_ffn_down_resadd"))
  if admission.packed_lanemap: authorities += (("decode_q6k_ffn_down_packed_lanemap", "q6_ffn_down_packed_lanemap"),)
  if admission.unroll_blocks is not None: authorities += (("decode_q6k_ffn_down_unroll", "q6_ffn_down_packed_unroll"),)
  emitter,ticket=lower_authorized_candidate({"family":"q6_ffn_down.v1","rows_per_block":admission.rows_per_block,
    "packed_lanemap":admission.packed_lanemap,"unroll_blocks":admission.unroll_blocks,
    "split_weight_stream":split_weight_stream}, authorities)
  consumer = KernelProgram("decode_q6k_ffn_down_mmvq", f"blk{admission.block_index}.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    emitter,
    output_spec=OutputSpec((ROWS,), dtypes.float32,
      typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32, (ROWS,), (1, 1, ROWS)),
        combine_fusion_admitted=False, epilogue_absorption_admitted=True)),
    typed_input_views=(TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(K,), route_role="ffn_down",
      requires_combine_fusion=False, requires_epilogue_absorption=True),),
    residual_input_views=(ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(ROWS,),
      route_role="ffn_down", kind="residual_add"),),
    boltbeam_ticket=ticket)
  weights = linear.q6k_storage.halfs.to(x.device)
  inputs = (weights, weights, xv, residual) if split_weight_stream else (weights, xv, residual)
  out = execute_promoted_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=x.device), *inputs, program=consumer)
  return out.reshape(1, 1, ROWS)


__all__ = ["Q6KFFNDownMMVQAdmission", "ROWS", "K", "emit_q6k_four_warp_fp16_direct",
           "q6k_ffn_down_mmvq_call"]
