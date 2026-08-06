"""D1 substrate lock (m4-resadd-rangeify-substrate-scope-20260806.md section 4, S2):

``remove_movement_op_after_rangeify`` must drop a movement op whose source is a rangeified
REDUCE scalar just like it drops one whose source is an INDEX scalar.  Before the D1 arm the
``RESHAPE(REDUCE(INDEX(STAGE(...), CONST 0)), (1,1,4096))`` nodes from the inlined flash
tile/combine chains survived into the debuf pass and crashed with ``bad reshape:
() -> (1, 1, 4096)``.  The rules here are the exact shape of that chain and its neighbors.
"""
from tinygrad import dtypes
from tinygrad.schedule.indexing import IndexingContext, pm_apply_rangeify
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.uop.ops import Ops, UOp


def _stage_scalar_reduce_chain() -> tuple[UOp, UOp, UOp, UOp]:
  """The exact crash shape: STAGE(1,1,1,4096,128) indexed by CONST 0, reduced to a scalar,
  then RESHAPEd to (1,1,4096)."""
  param = UOp.param(0, dtypes.float.ptr(4096 * 128), (1, 1, 1, 4096, 128), "CPU")
  rng = UOp.range(128, 0)
  stage = UOp(Ops.STAGE, dtypes.float, (param, rng),
              arg=BufferizeOpts(device="CPU", removable=True))
  idx = stage.index(UOp.const(dtypes.weakint, 0), ptr=False)
  red = idx.reduce(Ops.ADD)
  reshaped = red._mop(Ops.RESHAPE, (1, 1, 4096))
  return stage, idx, red, reshaped


def _index_scalar_chain() -> tuple[UOp, UOp]:
  """The pre-existing INDEX arm: movement over a rangeified INDEX scalar is a pure view."""
  param = UOp.param(0, dtypes.float.ptr(4096), (4096,), "CPU")
  idx = param.index(UOp.const(dtypes.weakint, 0), ptr=False)
  return idx, idx._mop(Ops.RESHAPE, (1, 1, 4096))


def test_movement_over_rangeified_reduce_scalar_drops():
  _, _, red, reshaped = _stage_scalar_reduce_chain()
  assert reshaped.src[0] is red
  ret = pm_apply_rangeify.rewrite(reshaped, IndexingContext())
  assert ret is red, "RESHAPE(REDUCE(...)) must collapse to the REDUCE after rangeify"


def test_movement_over_rangeified_index_scalar_drops():
  idx, reshaped = _index_scalar_chain()
  ret = pm_apply_rangeify.rewrite(reshaped, IndexingContext())
  assert ret is idx, "RESHAPE(INDEX(...)) must collapse to the INDEX after rangeify"


def test_movement_over_rangeified_reduce_drops_via_range_map():
  """The original arm: a movement op whose own uop is in range_map collapses even when the
  source is not an INDEX/REDUCE scalar."""
  _, _, red, reshaped = _stage_scalar_reduce_chain()
  ctx = IndexingContext()
  ctx.range_map[reshaped] = ((), ())
  ret = pm_apply_rangeify.rewrite(reshaped, ctx)
  assert ret is red


def test_movement_over_non_scalar_src_stays():
  """Negative case: a RESHAPE over an ordinary CAST is not a scalar view and must survive."""
  param = UOp.param(0, dtypes.float, (4096,), "CPU")
  cast = param.cast(dtypes.float16)
  reshaped = cast.reshape(1, 1, 4096)
  ret = pm_apply_rangeify.rewrite(reshaped, IndexingContext())
  assert ret is None, "RESHAPE(CAST(...)) must stay after rangeify"
