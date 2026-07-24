"""Regression test for the --logits-only LM-head unassignable-store codegen bug.

The --logits-only prefill path (extra/qk/prefill_whole_synced.py) compiles the full
512x151936 vocab-projection reduce kernel r_16_2374_..._<hash>.  Its final output
projection store lowers, at the reg-store-devectorize stage, to a STORE whose *target*
is an Ops.STACK of output-logit LANE READS -- `GEP(LOAD(INDEX(data0_GLOBAL, addr)), lane)`
-- with contiguous DUPLICATE groups: the size-2 inner reduce axis was UPCAST'd, so the
32 value lanes map onto only 16 distinct output addresses, each appearing twice
(lane0 is lane1).  The cstyle renderer renders such a STACK store target as a
make_floatN(...) constructor call -- an rvalue -- so gfx1100 rejects it:

    make_float32(val43.x,val43.x,val43.y,val43.y, ... ) = make_float32( ... );
    error: expression is not assignable

Neither pm_reduce_acc_upcast_fix (this is NOT a manual accumulator: the target is a
non-uniform grouped STACK of GLOBAL-load lane reads, so _manual_acc_store returns None
at its _reg_index check, and there is no reduce range -- the buf0 accumulator itself is
already correctly lowered to scalar per-lane stores) nor the existing
pm_distinct_reg_store_devec rules (the lanes are GEPs, not distinct REG INDEXes, so
_distinct_reg_store_indexes returns None) lower this store, so the unassignable
make_floatN target survives to the renderer.

The correct lowering restores the addressable GLOBAL store: for each distinct output
address it emits `data0[addr+lane] = sum(partial_lanes)`, horizontally reducing (ADD)
the duplicated partial-reduction lanes -- exactly the sum-reduce the UPCAST'd inner axis
represents.

This test rebuilds that minimal doubled-lane output-projection store, runs it through
pm_reduce_acc_upcast_fix then pm_distinct_reg_store_devec, then renders via the HIP
cstyle renderer, and asserts the rendered source contains no make_floatN(...) on the
LEFT of an '=' .  It FAILS on today's code and PASSES once the reg-store devectorizer
lowers the grouped output store to addressable per-address global stores.

NOTE: the manual-accumulator widen + REG-store devectorization passes live in
tinygrad/codegen/late/reg_store.py (moved there from devectorizer.py in the reg_store
refactor); the fix adds a rule to pm_distinct_reg_store_devec there.
"""
import re
from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UOp, graph_rewrite, pm_lower_index_dtype
from tinygrad.codegen.late.reg_store import pm_reduce_acc_upcast_fix, pm_distinct_reg_store_devec
from tinygrad.codegen.late.linearizer import linearize, pm_add_control_flow, CFGContext
from tinygrad.codegen import line_rewrite, pm_linearize_cleanups, pm_index_is_shrink, pm_remove_vec_dtypes
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer import Target

_LHS_MAKE_FLOAT = re.compile(r"make_float\d*\([^=]*\)\s*=")


def _build_output_projection_store(nload=2, width=4, dup=2, nelem=256):
  # STORE(STACK(GEP(LOAD(INDEX(data0_GLOBAL, addr)), lane) ...), STACK(distinct scalars))
  # models the LM-head output projection: nload*width distinct output slots, each fed by
  # `dup` distinct partial-reduction lanes that must be summed into the output address.
  data0 = UOp.placeholder((nelem,), dtypes.float, 0, addrspace=AddrSpace.GLOBAL)
  tgt_lanes, val_lanes, c = [], [], 0.0
  for r in range(nload):
    idx = data0.index(UOp.const(dtypes.weakint, r * width))
    load = idx.cast(dtypes.float.vec(width).ptr(nelem, addrspace=AddrSpace.GLOBAL)).load(dtype=dtypes.float.vec(width))
    for lane in range(width):
      gep = load.gep((lane,))
      for _ in range(dup):
        tgt_lanes.append(gep)
        val_lanes.append(UOp.const(dtypes.float, c)); c += 1.0
  n = len(tgt_lanes)
  tgt = UOp(Ops.STACK, dtypes.float.vec(n), tuple(tgt_lanes))
  val = UOp(Ops.STACK, dtypes.float.vec(n), tuple(val_lanes))
  return UOp.sink(tgt.store(val))


def _finalize_and_render(sink):
  ren = HIPRenderer(Target("AMD", arch="gfx1100"))
  sink = graph_rewrite(sink, pm_lower_index_dtype, name="lower all index dtypes")
  sink = graph_rewrite(sink, pm_index_is_shrink, name="index is shrink")
  sink = graph_rewrite(sink, pm_remove_vec_dtypes, name="transform to new style")
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)
  sink = graph_rewrite(sink, pm_distinct_reg_store_devec, name="post control-flow stack store devec")
  return ren.render(line_rewrite(linearize(sink), pm_linearize_cleanups))


def test_logits_only_output_projection_store_is_assignable():
  sink = _build_output_projection_store()
  # the two passes that are supposed to lower a vector REG/output store target
  sink = graph_rewrite(sink, pm_reduce_acc_upcast_fix, name="reduce acc upcast fix")
  sink = graph_rewrite(sink, pm_distinct_reg_store_devec, name="distinct reg store devec")

  # no STORE may keep a multi-lane STACK/VCAT target (that renders as an unassignable
  # make_floatN(...) constructor call).
  stack_target_stores = [u for u in sink.backward_slice
                         if u.op is Ops.STORE and u.src[0].op in (Ops.STACK, Ops.VCAT) and len(u.src[0].src) > 1]
  assert not stack_target_stores, \
    f"doubled-lane output-projection store target was not lowered; {len(stack_target_stores)} STACK-target " \
    f"store(s) survive (would render as make_floatN(...) = ..., 'expression is not assignable')"

  src = _finalize_and_render(sink)
  bad = [ln for ln in src.splitlines() if _LHS_MAKE_FLOAT.search(ln)]
  assert not bad, "rendered source has an unassignable make_floatN(...) store LHS:\n" + "\n".join(bad)


def test_manual_accumulator_widener_does_not_claim_the_output_projection():
  # Guard the attribution: the failing store is NOT a manual accumulator, so
  # pm_reduce_acc_upcast_fix must leave it untouched (it has no reduce range and the
  # value does not reference a REG accumulator).  If a future change made the widener
  # start claiming this shape that would be the wrong owner.
  sink = _build_output_projection_store()
  out = graph_rewrite(sink, pm_reduce_acc_upcast_fix, name="reduce acc upcast fix")
  assert out is sink, "pm_reduce_acc_upcast_fix must not rewrite the output-projection store"
