"""LR-040: the realization map -- which nodes must become real buffers.

This is the first of Phase 4's ownership splits, and it is deliberately the smallest one: a pure move of the
realization-map calculation out of `indexing.py` into the owner the target architecture names for it
("schedule/realize.py -- materialization and producer/consumer partition"). The graph rewrite that consumes this map
stays exactly where it was, per LR-040: *preserve the existing graph rewrite implementation until parity is
established*.

Nothing here decides *how* something is stored, when its lifetime ends, or which ranges it owns -- those are LR-041's
and LR-042's boundaries. This module answers one question: which nodes are forced to be real?

Not to be confused with `tinygrad/engine/realize.py`, which executes a schedule. This module is a scheduling
decision; that one is the runtime.
"""
from __future__ import annotations
from tinygrad.dtype import AddrSpace  # noqa: F401  (re-exported for callers that took it from here historically)
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp

# Nodes that are already real, or are pure views onto something real. A source whose base is one of these never needs
# forcing -- it is either a buffer, a device/const leaf, or an op that owns its own output.
ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.AFTER, Ops.COPY, Ops.BUFFER, Ops.SLICE,
                     Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.MSELECT, Ops.MSTACK, Ops.PARAM,
                     Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.LOAD, Ops.CALL, Ops.FUNCTION, Ops.MEMORY_SEMANTIC}

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_srcs(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_store_after_src(ctx:dict[UOp, None], dest:UOp, src:UOp):
  # don't realize COPY/SLICE when they are the direct source of STORE+AFTER — the target buffer is the output
  if src.op in {Ops.COPY, Ops.SLICE} and src in ctx \
     and not dest.op_in_backward_slice_with_self(Ops.SHRINK, Ops.PERMUTE, Ops.FLIP, Ops.PAD):
    del ctx[src]
  # you don't usually have to do this for assign unless there's a WAR hazard like TestAssign.test_assign_double_diamond_reduce
  if dest.base in src.backward_slice_with_self: ctx[src] = None

pm_generate_realize_map = PatternMatcher([
  # always realize
  (UPat({Ops.COPY, Ops.CONTIGUOUS, Ops.STORE}, name="tr"), realize),
  # realize srcs of these
  (UPat((Ops.COPY, Ops.MSELECT, Ops.MSTACK), name="rb"), realize_srcs),
  # sometimes we need to realize the src of STORE if there's a self-access
  (UPat(Ops.STORE, src=(UPat.var("dest"), UPat.var("src"))), realize_store_after_src),
])
