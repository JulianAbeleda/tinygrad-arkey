from tinygrad.dtype import dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite

from tinygrad.codegen.late.warp_reduce import WARP, pm_warp_reduce


def _lowered(lane_width: int, axis: AxisType, alu=Ops.ADD):
  lane = UOp.range(lane_width, 0, axis)
  return graph_rewrite(lane.cast(dtypes.float32).reduce(lane, arg=alu), pm_warp_reduce)


def _bpermutes(u: UOp) -> list[UOp]:
  return [x for x in u.toposort() if x.op is Ops.CUSTOMI and "ds_bpermute" in str(x.arg)]


def test_warp_reduce_lowers_sum_and_max_ladders():
  for alu in (Ops.ADD, Ops.MAX):
    out = _lowered(WARP, AxisType.WARP, alu)
    assert not any(x.op is Ops.REDUCE for x in out.toposort())
    assert len(_bpermutes(out)) == 5


def test_warp_reduce_lowers_group_reduce_width_16():
  assert len(_bpermutes(_lowered(16, AxisType.GROUP_REDUCE))) == 4


def test_warp_reduce_fails_closed_for_non_lane_axes_and_non_power_of_two_widths():
  for width, axis in ((WARP, AxisType.REDUCE), (24, AxisType.WARP)):
    out = _lowered(width, axis)
    assert any(x.op is Ops.REDUCE for x in out.toposort())
    assert not _bpermutes(out)


def test_warp_reduce_core_owns_old_extra_boundary():
  import importlib.util
  assert importlib.util.find_spec("tinygrad.codegen.experimental") is None
  assert importlib.util.find_spec("extra.llm_research.amd_warp_reduce") is None
  assert importlib.util.find_spec("extra.llm_research.warp_reduce_lowering") is None
