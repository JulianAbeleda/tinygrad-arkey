#!/usr/bin/env python3
"""G3.1 generated Q4_K GEMV lowering probe.

This is the first executable lowering hook after G2: build a named wave32 gate/up
program from the bridge-independent Q4KGateUpLaneMap. It intentionally does not
import or call qk_q4k_lane_partition_gemv.py.
"""
from __future__ import annotations

from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.dtype import AddrSpace, dtypes
from extra.amd_warp_reduce import WARP
from extra.q4_k_gemv_primitive import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, _q4k_block_dot_packed_load
from extra.qk_gemv_g2_lanemap import Q4KGateUpLaneMap
from extra.qk_lane_partition_reduce import LanePartition, lane_partition_reduce_sum


def q4k_g3_lanemap_gemv_kernel(rows:int, k:int, lanes:int=WARP):
  """Named wave32 UOp kernel generated from Q4KGateUpLaneMap.

  PROMOTED: this is now the default Q4_K decode GEMV route (BUBBLEBEAM_FUTURESIGHT
  defaults on), speed-equivalent to the owned warp kernel and token-identical.
  Rollback to owned warp: BUBBLEBEAM_FUTURESIGHT=0. The path is bound to the G2
  LaneMap representation and has its own program name, so the capture/gates can
  distinguish it from qk_q4k_lane_partition_gemv_*.
  """
  lm = Q4KGateUpLaneMap(k=k, n=rows, lane_extent=lanes)
  lm.validate()

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0")
    lane = UOp.special(lanes, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = lane_partition_reduce_sum(acc[0], part)
    return out[row].store(total).sink(arg=KernelInfo(name=f"q4k_g3_lanemap_gemv_{rows}_{k}", opts_to_apply=()))

  return kernel
