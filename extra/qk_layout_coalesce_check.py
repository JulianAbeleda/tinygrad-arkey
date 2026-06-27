#!/usr/bin/env python3
"""Static coalescing predicate -- the SEED of the layout/mapping IR (docs/layout-mapping-ir-design-20260625.md).

Makes "is this buffer access coalesced for a given thread axis" a STATIC query over the existing RANGE/INDEX
address algebra, instead of an emergent property you can only benchmark. Definition (CuTe-style): the stride of a
RANGE in an index expression = address(rng+1) - address(rng) = idx[rng=1] - idx[rng=0]; the access is COALESCED for
a thread/lane RANGE iff that stride is 1 (consecutive lanes -> consecutive addresses). Vector/float4 width = the
longest unit-stride run. This is the first-class predicate `OptOps.COALESCE`/the layout IR needs (M0).
"""
from __future__ import annotations
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic

def axis_stride(idx:UOp, rng:UOp) -> int | None:
  """Stride of `rng` in the symbolic index expr `idx` = idx[rng=1] - idx[rng=0]. None if data-dependent."""
  d = graph_rewrite(idx.substitute({rng: rng.const_like(1)}) - idx.substitute({rng: rng.const_like(0)}), symbolic)
  # data-dependent / masked (PAD->Invalid) indices are not a constant stride -> decline (don't crash on int(Invalid))
  return int(d.arg) if d.op is Ops.CONST and isinstance(d.arg, int) else None

def is_coalesced(idx:UOp, thread_rng:UOp) -> bool:
  """Coalesced for `thread_rng` iff consecutive lanes hit consecutive addresses (unit stride)."""
  return axis_stride(idx, thread_rng) == 1

def vector_width(idx:UOp, thread_rng:UOp, maxw:int = 4) -> int:
  """float4-style coalesced load width for `thread_rng`: largest pow2 <= maxw consistent with unit stride."""
  return maxw if axis_stride(idx, thread_rng) == 1 else 1
