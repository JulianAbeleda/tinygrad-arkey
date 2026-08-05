"""Research-only 128-thread Q4_K decode ownership witness.

The installed Q4_K G3 kernel assigns one 32-lane warp to one output row.  Its
lane owns one packed word in each of all eight 32-value groups, across four
Q4_K blocks: 128 fp16 products/lane.  Current llama.cpp MMVQ on generic CUDA
(including the Blackwell fallback table) assigns four warps to one row.  Each
lane owns two four-value words in each of four blocks: 32 products/lane.

This file deliberately does *not* wire a route or run hardware.  It is a
static construction: one flat 128-thread LOCAL axis and warp-local reductions
to the established four-partial consumer ABI.  It exists to make the different ownership
claim renderable and testable before a numerics and GPU gate is authorized.
"""
from __future__ import annotations

from tinygrad import dtypes
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import Q4K_WORDS_PER_BLOCK, _q4k_group_dot_packed_load
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

ROWS, K, WARP, WARPS_PER_ROW, BLOCKS_PER_WARP = 4096, 4096, 32, 4, 4

def installed_ownership_coordinates(k:int=K):
  """(warp, lane, q4-block, group, word-in-group), one row's installed work."""
  if k != 4096: raise ValueError("v1 witness is fixed to K=4096")
  # installed: lane//8 chooses a four-block K partition; lane%8 chooses word.
  return [(0, lane, (lane//8)*4 + block_rel, group, lane%8)
          for lane in range(32) for block_rel in range(4) for group in range(8)]

def cooperative_ownership_coordinates(k:int=K):
  """(warp, lane, q4-block, group, word-in-group), one row's proposed work."""
  if k != 4096: raise ValueError("v1 witness is fixed to K=4096")
  # Four warps own disjoint four-block K stripes.  A lane's two words are in
  # adjacent groups selected by lane//8, and lane%8 is the packed-word column.
  return [(warp, lane, warp*4 + block_rel, 2*(lane//8) + pair, lane%8)
          for warp in range(4) for lane in range(32) for block_rel in range(4) for pair in range(2)]

def flat_cooperative_ownership_coordinates(k:int=K):
  if k != 4096: raise ValueError("v1 witness is fixed to K=4096")
  return [(lid//32, lid%32, (lid//32)*4 + block_rel, 2*((lid%32)//8) + pair, lid%8)
          for lid in range(128) for block_rel in range(4) for pair in range(2)]

def emit_q4k_warp_cooperative_partial(rows:int=ROWS, k:int=K):
  """Return a direct fp16-input research emitter, never a production route.

  The output is four fp32 partials per row.  The fp16 input intentionally preserves
  installed Q4_K arithmetic/ABI; Q8 activation representation is a separate
  closed shared-Q8 question.  Thus this isolates lane/warp ownership only.
  """
  if k != 4096 or rows <= 0: raise ValueError("v1 requires positive rows and K=4096")
  k_blocks = k//256
  if k_blocks != WARPS_PER_ROW*BLOCKS_PER_WARP: raise ValueError("unexpected K-block partition")
  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    # SPECIAL pins the intended flat CUDA launch geometry.  A RANGE LOCAL can
    # be factorized by generic GPU-dimension lowering before this research
    # body is inspected, which would defeat the ownership witness.
    row = UOp.special(rows, "gidx0")
    lid = UOp.special(128, "lidx0")
    warp, lane = lid//32, lid%32
    lane_group, word = lane//8, lane%8
    block_rel = UOp.range(BLOCKS_PER_WARP, 2, axis_type=AxisType.REDUCE)
    block = warp*BLOCKS_PER_WARP + block_rel
    base = (row*k_blocks + block)*Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    # Each lane owns exactly two packed words (=8 values) per block.  The
    # four static arms are an authoring-time stand-in for the dynamic subgroup
    # selection used by MMVQ; their rendered predication is an explicit audit
    # target, not a claim that this is already llama-equivalent instruction code.
    for group_pair in range(4):
      active = lane_group.eq(group_pair)
      for pair in range(2):
        group = 2*group_pair + pair
        contrib = contrib + active.where(_q4k_group_dot_packed_load(words, x, base, block, group, word),
                                          UOp.const(dtypes.float32, 0.0))
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(block_rel)[0] + contrib).end(block_rel))
    total = acc[0]
    for slot, off in enumerate((16, 8, 4, 2, 1), 90): total = total + _staged_shfl(total, off, lane, slot)
    return out[row, warp].store(total, lane.eq(0)).sink(
      arg=KernelInfo(name=f"q4k_warp_coop_partial_{rows}_{k}", opts_to_apply=()))
  return kernel
