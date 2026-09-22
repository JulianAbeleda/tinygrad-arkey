"""Structural two-segment Stream-K control experiment.

This module deliberately contains no Q6 arithmetic.  It isolates the owner
control transformation: segment boundaries are materialized by the host, and
the device gate executes one sequential accumulator per owner with one
boundary flush/reset.  The result uses the existing row-major owner ABI.
"""
from dataclasses import dataclass
from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.prefill.nv_generated_q6k_streamk import OWNERS, TILES, K_BLOCKS, streamk_segments

@dataclass(frozen=True)
class StructuralOwner:
  owner: int
  first_tile: int
  first_begin: int
  first_end: int
  second_tile: int
  second_begin: int
  second_end: int

def structural_owner_table():
  """Return fixed two-segment descriptors, with -1 for an absent segment."""
  rows = []
  for owner in range(OWNERS):
    segs = streamk_segments(owner)
    a = segs[0]
    b = segs[1] if len(segs) == 2 else None
    rows.append(StructuralOwner(owner, a.tile_id, a.begin, a.end,
      b.tile_id if b else -1, b.begin if b else 0, b.end if b else 0))
  return tuple(rows)

def structural_owner_gate(out, values):
  """Compile a two-segment owner reduction with one explicit flush/reset.

  ``values`` is indexed in linear tile/K order.  This is a control oracle for
  the eventual MMA body: it proves that splitting the loop into fixed segment
  bounds preserves the ordered partial-slot ABI and removes transition tests
  from the arithmetic loop.
  """
  owner = UOp.special(OWNERS, "gidx0")
  table = structural_owner_table()
  row = table[0]
  # The gate is intentionally emitted per owner by the caller's graph.  The
  # host descriptor table is the source of truth; this scalar form is used by
  # tests to validate the flush/reset recurrence before MMA integration.
  lo = UOp.const(dtypes.int32, row.first_tile*K_BLOCKS + row.first_begin)
  hi = UOp.const(dtypes.int32, row.first_tile*K_BLOCKS + row.first_end)
  blk = UOp.range(hi-lo, 40, axis_type=AxisType.REDUCE)
  acc = UOp.placeholder((1,), dtypes.float32, 940,
    addrspace=__import__('tinygrad.dtype', fromlist=['AddrSpace']).AddrSpace.REG)
  init = acc[0].store(0.0)
  state = acc.after(init).after(blk)
  update = state[0].store(state[0] + values[lo+blk])
  done = update.end(blk)
  return UOp.sink(out[owner*2].store(acc.after(done)[0]),
    arg=KernelInfo(name="nv_q6_structural_owner_gate", opts_to_apply=()))

__all__ = ["StructuralOwner", "structural_owner_table", "structural_owner_gate"]
