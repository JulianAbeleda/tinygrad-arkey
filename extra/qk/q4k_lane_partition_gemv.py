#!/usr/bin/env python3
"""Q4_K decode GEMV using the first-class LanePartitionReduce primitive.

This is the narrow P2.1 retry path: it keeps the owned q4k thread map default-off/research-only, but expresses the
lane split and wave combine through `LanePartition` instead of open-coded `lane//8`, `lane%8`, and `warp_reduce_sum`.
It does not change generic add_gpudims REDUCE handling.
"""
from __future__ import annotations

from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.dtype import AddrSpace, dtypes
from extra.qk.amd_warp_reduce import WARP
from extra.qk.lane_partition_reduce import LanePartition, lane_partition_reduce_sum
from extra.qk.quant.q4_k_gemv_primitive import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, _q4k_block_dot_packed_load


def q4k_lane_partition_gemv_kernel(rows:int, k:int, lanes:int=WARP):
  """One-wave-per-row Q4_K GEMV kernel using LanePartitionReduce.

  Shape constraints mirror the owned warp kernel: wave32, 4 block-groups, k_blocks divisible by 4.
  """
  if lanes != WARP: raise ValueError(f"q4k lane partition GEMV supports wave{WARP} only")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  if k % Q4_K_BLOCK_ELEMS != 0: raise ValueError(f"k={k} must be divisible by {Q4_K_BLOCK_ELEMS}")
  if k_blocks % 4 != 0: raise ValueError(f"k_blocks={k_blocks} must be divisible by 4")
  bpb = k_blocks // 4

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0")
    lane = UOp.special(WARP, "lidx0")
    part = LanePartition(lane)
    lblk = UOp.range(bpb, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * bpb + lblk
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = lane_partition_reduce_sum(acc[0], part)
    return out[row].store(total).sink(arg=KernelInfo(name=f"q4k_lane_partition_gemv_{rows}_{k}", opts_to_apply=()))

  return kernel
