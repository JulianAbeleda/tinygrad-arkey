"""Research-only Q4_K/Q8_1 four-warp-per-row MMVQ-shaped lowering.

This is deliberately outside every runtime route.  Unlike the first ownership
witness, the two logical Q4 words owned by a lane are addressed from its
runtime lane coordinates: the body does not expand eight possible Q4 groups
and mask seven of them.  It is a static/PTX gate candidate, not a promotion.
"""
from __future__ import annotations

from tinygrad import dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import Q4K_WORDS_PER_BLOCK, _f16_word
from tinygrad.llm.shared_q8_attention import _i8lane, _q8_d
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

ROWS, K, WARP, WARPS_PER_ROW, BLOCKS_PER_WARP = 4096, 4096, 32, 4, 4


def dynamic_ownership_coordinates(k:int=K):
  """(warp, lane, q4-block, group, logical-word), exactly two words/lane/block.

  A group has eight four-value logical words.  ``lane//4`` selects its group
  and ``(lane%4)*2 + word_slot`` selects two contiguous words.  Thus all
  8*8 logical words of every Q4_K block occur once per warp, while a physical
  Q4_K u32 can be shared by the adjacent low/high-nibble groups as specified
  by the format.
  """
  if k != K: raise ValueError("v2 is fixed to K=4096")
  return [(warp, lane, warp*BLOCKS_PER_WARP+block_rel, lane//4, (lane%4)*2+word_slot)
          for warp in range(WARPS_PER_ROW) for lane in range(WARP)
          for block_rel in range(BLOCKS_PER_WARP) for word_slot in range(2)]


def emit_q4k_warp_cooperative_q8_partial(rows:int=ROWS, k:int=K, block_count:UOp|None=None):
  """Emit four Q8_1/DP4A partials per output row with one flat LOCAL=128 axis.

  ``xp`` is the existing llama-Q8 provider ABI: 1024 int8x4 packets followed
  by 128 d|s metadata words.  Q4's minimum correction intentionally uses d
  and the int8 sum, matching the existing Q4 Q8 consumer.
  """
  if k != K or rows <= 0: raise ValueError("v2 requires positive rows and K=4096")
  block_count = BLOCKS_PER_WARP if block_count is None else block_count
  def kernel(out:UOp, words:UOp, xp:UOp) -> UOp:
    row, lid = UOp.special(rows, "gidx0"), UOp.special(128, "lidx0")
    warp, lane = lid//WARP, lid%WARP
    # Dynamic lane ownership: one of 8 groups and two contiguous logical words.
    group, word_base = lane//4, (lane%4)*2
    # Keep the four owned Q4_K blocks as a real loop.  A REDUCE axis would
    # author four copies of the dynamic fragment, defeating the body-wide
    # instruction-footprint gate before hardware ever sees it.
    block_rel = UOp.range(block_count, 2, axis_type=AxisType.LOOP)
    block = warp*BLOCKS_PER_WARP + block_rel
    base = (row*(k//256)+block)*Q4K_WORDS_PER_BLOCK

    # Only four fixed header words are loaded.  The group-dependent scale
    # extraction is arithmetic on those registers, not an eight-arm load tree.
    w0, w1, w2, w3 = words[base], words[base+1], words[base+2], words[base+3]
    d, dmin = _f16_word(w0, False), _f16_word(w0, True)
    g4 = group%4
    b1 = w1.rshift(g4*8).bitwise_and(0xff)
    b2 = w2.rshift(g4*8).bitwise_and(0xff)
    # For groups 4..7, w3 owns one full byte/group: low nibble is scale,
    # high nibble is min. The failed first live gate used a four-bit stride
    # here and therefore made the high-group min nibble zero.
    hb = w3.rshift(g4*8).bitwise_and(0xff)
    sc = (group < 4).where(b1.bitwise_and(63), hb.bitwise_and(0xf).bitwise_or(b1.rshift(6).lshift(4)))
    mn = (group < 4).where(b2.bitwise_and(63), hb.rshift(4).bitwise_or(b2.rshift(6).lshift(4)))

    contrib = UOp.const(dtypes.float32, 0.0)
    for word_slot in range(2):
      word = word_base + word_slot
      # Q4_K stores the low/high group nibbles in the same u32; group selects
      # its nibble dynamically.  Address ownership is contiguous within each
      # lane and no inactive group is ever dereferenced.
      qw = words[base + 4 + (group//2)*8 + word].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
      xv = xp[block*64 + group*8 + word]
      dot = int8x4_dot(UOp.const(dtypes.int32, 0), qw, xv).cast(dtypes.float32)
      xsum = _i8lane(xv, 0)+_i8lane(xv, 1)+_i8lane(xv, 2)+_i8lane(xv, 3)
      contrib = contrib + _q8_d(xp, block*8+group)*(d*sc.cast(dtypes.float32)*dot - dmin*mn.cast(dtypes.float32)*xsum.cast(dtypes.float32))
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(block_rel)[0] + contrib).end(block_rel))
    total = acc[0]
    for slot, off in enumerate((16, 8, 4, 2, 1), 90): total = total + _staged_shfl(total, off, lane, slot)
    return out[row, warp].store(total, lane.eq(0)).sink(
      arg=KernelInfo(name=f"q4k_warp_coop_q8_dp4a_partial_{rows}_{k}", opts_to_apply=()))
  return kernel


def emit_q4k_warp_cooperative_q8_partial_runtime_blocks(rows:int=ROWS, k:int=K):
  """Same body with an unbound scalar extent, solely to probe no-unroll IR."""
  return emit_q4k_warp_cooperative_q8_partial(rows, k, UOp.variable("q4k_coop_blocks", 1, BLOCKS_PER_WARP))
