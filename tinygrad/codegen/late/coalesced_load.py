#!/usr/bin/env python3
"""Coalesced vector-load lowering -- the bandwidth PRIMITIVE for generated kernels.

Promoted to core codegen under LR-050 (docs/task_workflow/output/lowering-architecture-refactor-scope-20260726.md
Phase 5): this pass and the static coalescing predicate it steers on are pure UOp/AxisType transforms with no
backend-specific assumptions (no wave width, no ISA intrinsic, no device string baked into the algorithm itself).
Originally authored as `extra/qk/coalesced_load_lowering.py` + `extra/qk/layout_coalesce_check.py`; those modules
now re-export this implementation rather than forking it.

The codegen realization of the layout IR's `OptOps.COALESCE` (docs/layout-mapping-ir-design-20260625.md,
docs/decode-coalesced-load-primitive-scope-20260626.md). Custom/generated kernels author with
`opts_to_apply=()`, so the heuristic never tags any axis `UPCAST` and every load stays scalar
(`global_load_d16=0`). This pass makes "which contiguous load axis to widen" a first-class, PREDICATE-DRIVEN
decision: it finds a small loop/reduce RANGE that is **unit-stride in a GLOBAL/LOCAL load index** (via the
static coalescing predicate `axis_stride`) and promotes its `AxisType` to `UPCAST`, so the EXISTING expander
+ devectorizer fold the load into a vector load (`*((float4*)..)` -> `global_load_dwordx4`/`d16`). It pairs
with `REG_STORE_DEVEC` (which keeps the REG accumulator stores scalar -- avoiding the
`make_float4(...)=make_float4(...)` invalid-C); the codegen hook fires reg-store-devec whenever this is on.

Call-site gating (still in `tinygrad/codegen/__init__.py`, unchanged by this promotion): default-off behind
`COALESCED_LOAD_LOWERING`, and further restricted to `ren.target.device == "AMD"` -- that device restriction is a
validation-scope gate (only AMD's codegen path has been proven end-to-end with this pass), not a property of the
algorithm, which is why it stays at the call site rather than inside this module. Pure AST transform; only
changes the AxisType of axes proven unit-stride in a load. Nothing about correctness changes -- an UPCAST axis is
fully unrolled, same values, just vectorized loads + (with reg-store-devec) scalar accumulates.
"""
from __future__ import annotations
from tinygrad.dtype import PtrDType, AddrSpace
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops, UOp, AxisType, graph_rewrite
from tinygrad.uop.symbolic import symbolic


def axis_stride(idx:UOp, rng:UOp) -> int | None:
  """Stride of `rng` in the symbolic index expr `idx` = idx[rng=1] - idx[rng=0]. None if data-dependent.

  Static coalescing predicate (CuTe-style): the stride of a RANGE in an index expression =
  address(rng+1) - address(rng) = idx[rng=1] - idx[rng=0]; the access is COALESCED for a thread/lane RANGE
  iff that stride is 1 (consecutive lanes -> consecutive addresses). Makes "is this buffer access coalesced
  for a given thread axis" a static query over the existing RANGE/INDEX address algebra, instead of an
  emergent property you can only benchmark.
  """
  d = graph_rewrite(idx.substitute({rng: rng.const_like(1)}) - idx.substitute({rng: rng.const_like(0)}), symbolic)
  # data-dependent / masked (PAD->Invalid) indices are not a constant stride -> decline (don't crash on int(Invalid))
  return int(d.arg) if d.op is Ops.CONST and isinstance(d.arg, int) else None

def is_coalesced(idx:UOp, thread_rng:UOp) -> bool:
  """Coalesced for `thread_rng` iff consecutive lanes hit consecutive addresses (unit stride)."""
  return axis_stride(idx, thread_rng) == 1

def vector_width(idx:UOp, thread_rng:UOp, maxw:int = 4) -> int:
  """float4-style coalesced load width for `thread_rng`: largest pow2 <= maxw consistent with unit stride."""
  return maxw if axis_stride(idx, thread_rng) == 1 else 1


def _buf_of(index: UOp) -> UOp:
  b = index.src[0]
  while b.op in (Ops.INDEX, Ops.AFTER, Ops.RESHAPE): b = b.src[0]
  return b


def coalesce_loads(sink: UOp, max_width: int = 4) -> UOp:
  """Promote unit-stride small load axes to UPCAST so the existing coalescer vectorizes them."""
  tl = list(sink.toposort())
  # At this (pre-expander) stage buffer reads are bare Ops.INDEX; Ops.LOAD is added later in devectorize.
  # A load-INDEX is any INDEX into a non-REG ptr buffer that is NOT a STORE target.
  store_targets: set[UOp] = set()
  acc_axes: set[UOp] = set()   # ranges that index a REG store -> accumulator/carry axes; never coalesce
  for u in tl:
    if u.op is Ops.STORE and len(u.src) >= 1:
      store_targets.add(u.src[0])
      tgt = u.src[0]
      if tgt.op is Ops.INDEX and len(tgt.src) >= 2 and isinstance((tb := _buf_of(tgt)).dtype, PtrDType) \
         and tb.dtype.addrspace == AddrSpace.REG:
        acc_axes.update(r for r in tgt.src[1].ranges if r.op is Ops.RANGE)

  promote: dict[UOp, UOp] = {}
  for idxn in tl:
    if idxn.op is not Ops.INDEX or idxn in store_targets: continue
    if not isinstance(idxn.src[0].dtype, PtrDType) or len(idxn.src) < 2: continue
    buf = _buf_of(idxn)
    if not isinstance(buf.dtype, PtrDType) or buf.dtype.addrspace == AddrSpace.REG: continue
    idx = idxn.src[1]
    for r in idx.ranges:
      if r.op is not Ops.RANGE or r in promote or r in acc_axes: continue   # skip accumulator axes
      if r.arg[-1] not in (AxisType.LOOP, AxisType.REDUCE): continue
      if r.src[0].op is not Ops.CONST or not isinstance(r.src[0].arg, int): continue  # constant bound only (no symbolic)
      n = int(r.vmax) + 1
      if n <= 1 or n > max_width or max_width % n != 0: continue
      if axis_stride(idx, r) != 1: continue            # the coalescing predicate -- steers the decision
      # fresh UPCAST range, same id/size/dtype; substitute swaps it everywhere incl. the END that closes it
      promote[r] = UOp(Ops.RANGE, r.dtype, r.src, (r.arg[0], AxisType.UPCAST))

  if not promote: return sink
  if getenv("COALESCED_LOAD_DEBUG"):
    for r, nr in promote.items():
      print(f"[COALESCE] promote axis={r.arg[0]} {r.arg[-1].name} size={int(r.vmax)+1} -> UPCAST")
  return sink.substitute(promote)
