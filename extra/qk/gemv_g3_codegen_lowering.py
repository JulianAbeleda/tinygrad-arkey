#!/usr/bin/env python3
"""G3.1 generated Q4_K GEMV lowering probe.

This is the first executable lowering hook after G2: build a named wave32 gate/up
program from the bridge-independent Q4KGateUpLaneMap. It intentionally does not
import or call qk_q4k_lane_partition_gemv.py.
"""
from __future__ import annotations

from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.dtype import AddrSpace, dtypes
from extra.qk.amd_warp_reduce import WARP
from extra.qk.quant.q4_k_gemv_primitive import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, _q4k_block_dot_packed_load
from extra.qk.gemv_g2_lanemap import Q4KGateUpLaneMap
from extra.qk.lane_partition_reduce import LanePartition, lane_partition_reduce_sum


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


def q4k_g3_lanemap_gemv_splitk_kernel(rows:int, k:int, parts:int, lanes:int=WARP):
  """Split-K variant of the generated G3 Q4_K GEMV (L2 parity lever).

  Generic, not model/shape-specific: adds a SECOND global axis (gidx1) over `parts` K-slices so each output row
  is computed by `parts` workgroups instead of one. This raises occupancy for row-starved GEMVs (e.g. the KV
  projections 5120->1024, which otherwise launch only `rows` workgroups). The per-lane block reduce
  (blocks_per_group) is partitioned across the K-parts; each (row, kpart) writes a partial to out[row*parts+kpart]
  and the caller finalizes with a sum over parts. `parts` must divide blocks_per_group (checked here).
  """
  lm = Q4KGateUpLaneMap(k=k, n=rows, lane_extent=lanes)
  lm.validate()
  bpg = lm.blocks_per_group
  if parts < 1 or bpg % parts != 0:
    raise ValueError(f"parts={parts} must be a positive divisor of blocks_per_group={bpg} (k={k}, rows={rows})")
  sub = bpg // parts

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0")
    kpart = UOp.special(parts, "gidx1")
    lane = UOp.special(lanes, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(sub, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * bpg + kpart * sub + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = lane_partition_reduce_sum(acc[0], part)
    return out[row * parts + kpart].store(total).sink(
      arg=KernelInfo(name=f"q4k_g3_lanemap_gemv_splitk_{rows}_{k}_{parts}", opts_to_apply=()))

  return kernel


def q4k_g3_lanemap_gemv_inkernel_combine_kernel(rows:int, k:int, parts:int, lanes:int=WARP):
  """In-kernel-combine variant of the generated G3 Q4_K GEMV (the unifying parity capability).

  Unlike the split-K variant (which adds a second GLOBAL axis + an EXTERNAL .sum over parts — occupancy gain
  offset by the added combine reduce), this raises per-row parallelism with a WIDER workgroup: `parts` waves
  (parts*lanes threads) per output row. Each wave reduces its K-slice via the existing single-wave
  lane_partition_reduce_sum, writes its partial to LDS, and after a workgroup barrier the `parts` partials are
  combined IN-KERNEL to write out[row] directly — ONE store, NO external partials buffer, NO external reduce.

  Generic (any rows/k/parts with parts | blocks_per_group); native codegen only (LDS DEFINE_LOCAL + s_barrier +
  reduce), no handwritten kernel. `parts` must divide blocks_per_group.
  """
  lm = Q4KGateUpLaneMap(k=k, n=rows, lane_extent=lanes)
  lm.validate()
  bpg = lm.blocks_per_group
  if parts < 1 or bpg % parts != 0:
    raise ValueError(f"parts={parts} must be a positive divisor of blocks_per_group={bpg} (k={k}, rows={rows})")
  sub = bpg // parts

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0")
    tid = UOp.special(parts * lanes, "lidx0")      # parts waves x `lanes` lanes = one wide workgroup per row
    wave = tid // lanes                            # 0..parts-1 (physical wavefront index)
    lane = tid % lanes                             # 0..lanes-1 within the wave
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(sub, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * bpg + wave * sub + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    wave_partial = lane_partition_reduce_sum(acc[0], part)     # per-wave partial; every lane in the wave holds it
    # in-kernel LDS combine of the `parts` per-wave partials -> out[row] (no external reduce). `parts` is small
    # (5 for KV) so the combine is UNROLLED (a fixed-size ALU sum of LDS slots after one barrier), not a
    # REDUCE range over an LDS load (which the custom-kernel linearizer does not lower).
    # LDS MUST use an INT slot (placeholder), not a raw DEFINE_LOCAL with a string name: the linearizer's
    # priority tiebreaker compares the slot against the REG accumulators' int slots, and int-vs-str is unsortable.
    lds = UOp.placeholder((parts,), dtypes.float32, 205, addrspace=AddrSpace.LOCAL)
    st = lds[wave].store(wave_partial, gate=(lane < 1))       # lane 0 of each wave writes its partial
    bar = st.barrier()
    total = lds.after(bar)[0].load()
    for j in range(1, parts):
      total = total + lds.after(bar)[j].load()
    return out[row].store(total, gate=(tid < 1)).sink(       # only tid 0 writes the combined result
      arg=KernelInfo(name=f"q4k_g3_lanemap_gemv_inkcomb_{rows}_{k}_{parts}", opts_to_apply=()))

  return kernel
