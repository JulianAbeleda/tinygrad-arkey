from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite

from tinygrad.codegen.late.warp_reduce import WARP, WARP_SHFL_XOR_TAG, pm_warp_reduce, pm_lower_warp_shfl_xor
from tinygrad.codegen.late.warp_reduce import warp_reduce_sum_across_groups, warp_reduce_max_across_groups

_amd = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))


def _lowered(lane_width: int, axis: AxisType, alu=Ops.ADD):
  lane = UOp.range(lane_width, 0, axis)
  out = graph_rewrite(lane.cast(dtypes.float32).reduce(lane, arg=alu), pm_warp_reduce)
  # pm_warp_reduce alone only produces the renderer-agnostic warp_shfl_xor tag (TG1); resolve it against a
  # concrete AMD renderer here, the same way codegen/__init__.py does, to get back the ds_bpermute text.
  return graph_rewrite(out, pm_lower_warp_shfl_xor, ctx=_amd)


def _bpermutes(u: UOp) -> list[UOp]:
  return [x for x in u.toposort() if x.op is Ops.CUSTOMI and "ds_bpermute" in str(x.arg)]


def test_warp_shfl_xor_is_renderer_agnostic_before_lowering():
  lane = UOp.range(WARP, 0, AxisType.WARP)
  out = graph_rewrite(lane.cast(dtypes.float32).reduce(lane, arg=Ops.ADD), pm_warp_reduce)
  tagged = [x for x in out.toposort() if x.op is Ops.CUSTOMI and isinstance(x.arg, tuple) and x.arg[:1] == (WARP_SHFL_XOR_TAG,)]
  assert tagged, "pm_warp_reduce should emit the renderer-agnostic tag, not a concrete target string"
  assert not any(x.op is Ops.CUSTOMI and isinstance(x.arg, str) and "amdgcn" in x.arg for x in out.toposort())


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


def test_across_group_reduces_emit_expected_xor_stages():
  # 8-lane groups over a 32-lane warp need exactly two cross-group stages (offsets 8, then 16).
  lane = UOp.range(WARP, 0, AxisType.WARP)
  val = lane.cast(dtypes.float32)
  for fxn in (warp_reduce_sum_across_groups, warp_reduce_max_across_groups):
    out = fxn(val, lane, 8)
    lowered = graph_rewrite(out, pm_lower_warp_shfl_xor, ctx=_amd)
    bpermutes = [x for x in lowered.toposort() if x.op is Ops.CUSTOMI and "ds_bpermute" in str(x.arg)]
    assert len(bpermutes) == 2
    assert "__shfl_xor_sync" not in str(out)


def test_across_group_reduce_is_group_aligned_and_renderer_agnostic():
  lane = UOp.range(WARP, 0, AxisType.WARP)
  val = lane.cast(dtypes.float32)
  out = warp_reduce_max_across_groups(val, lane, 16)
  # group_size=16 leaves only one cross-group stage before the width limit (offset 16).
  tagged = [x for x in out.toposort() if x.op is Ops.CUSTOMI and isinstance(x.arg, tuple) and x.arg[:1] == (WARP_SHFL_XOR_TAG,)]
  assert len(tagged) == 1
  assert tagged[0].arg[1] == 16
