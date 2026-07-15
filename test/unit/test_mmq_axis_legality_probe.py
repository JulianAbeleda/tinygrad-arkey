from extra.qk.mmq_axis_legality_probe import build_mmq_axis_probe, compile_mmq_axis_probe
from tinygrad.uop.ops import AxisType, Ops


def test_generated_mmq_global_local_axes_compile_without_range_leakage():
  info = compile_mmq_axis_probe()
  assert info.global_size == (2, 1, 1)
  assert info.local_size == (4, 1, 1)
  ranges = [u for u in build_mmq_axis_probe().toposort() if u.op is Ops.RANGE]
  assert {(r.arg[0], r.arg[1]) for r in ranges} == {
    (0, AxisType.GLOBAL), (1, AxisType.LOCAL), (2, AxisType.REDUCE)}
  # REDUCE remains semantic work, never a launch dimension.
  assert all(r.arg[1] is not AxisType.REDUCE for r in ranges if r.arg[0] in (0, 1))
