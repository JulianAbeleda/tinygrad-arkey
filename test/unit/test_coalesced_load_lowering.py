"""LR-050: coalesced-load promotion tests.

`extra/qk/coalesced_load_lowering.py` and `extra/qk/layout_coalesce_check.py` used to own this pass; it is now
core codegen (`tinygrad/codegen/late/coalesced_load.py`), with the extra/qk modules re-exporting rather than
forking. These tests are CPU/compile-only (no GPU): the "AMD" renderer used below is the ISA renderer's
compile-only oracle (`Target.parse("AMD:ISA:...")`), which never touches a device.

Positive/negative control: `coalesce_loads` must promote a unit-stride LOOP/REDUCE axis feeding a load to
AxisType.UPCAST, and must NOT touch a non-unit-stride axis or a REG accumulator axis. A second, pipeline-level
control enables the `COALESCED_LOAD_LOWERING` env gate (default off) end to end through
`tinygrad.codegen.full_rewrite_to_sink` with an AMD ISA renderer and shows the emitted AST actually changes
(a vectorized LOAD appears) only when the gate is on -- proving the promoted pass does something, not just that
it is importable.
"""
import os
import unittest

from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.coalesced_load import coalesce_loads, axis_stride
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


def _set_gate(value: str | None) -> None:
  # tinygrad's getenv() is @functools.cache'd (tinygrad/helpers.py), so a bare os.environ write is invisible
  # to code that already called getenv() for this key in this process -- must clear the cache too.
  if value is None: os.environ.pop("COALESCED_LOAD_LOWERING", None)
  else: os.environ["COALESCED_LOAD_LOWERING"] = value
  getenv.cache_clear()


def _sum4_ast() -> UOp:
  """out[0] = sum(src[0:4]) as a hand-built AST with opts_to_apply=() (no heuristic optimization), so any
  vectorization present after full_rewrite_to_sink must come from coalesce_loads, not the normal optimizer."""
  src = UOp.param(1, dtypes.float.ptr(4))
  out = UOp.param(0, dtypes.float.ptr(1))
  i = UOp.range(4, 0, AxisType.REDUCE)
  loaded = src.index(i).load()
  red = loaded.reduce(i, arg=Ops.ADD)
  return out.index(UOp.const(dtypes.int, 0), ptr=True).store(red).sink(arg=KernelInfo(opts_to_apply=()))


class TestCoalesceLoadsUnit(unittest.TestCase):
  """Direct positive/negative control on the promoted function itself."""

  def test_unit_stride_load_axis_is_promoted_to_upcast(self):
    src = UOp.param(1, dtypes.float.ptr(4))
    out = UOp.param(0, dtypes.float.ptr(1))
    i = UOp.range(4, 0, AxisType.REDUCE)
    loaded = src.index(i).load()
    red = loaded.reduce(i, arg=Ops.ADD)
    sink = out.index(UOp.const(dtypes.int, 0), ptr=True).store(red).sink()

    before_ranges = {u.arg[0]: u.arg[-1] for u in sink.toposort() if u.op is Ops.RANGE}
    assert before_ranges == {0: AxisType.REDUCE}

    after = coalesce_loads(sink)
    after_ranges = {u.arg[0]: u.arg[-1] for u in after.toposort() if u.op is Ops.RANGE}
    # positive control: the AST actually changed -- the axis is now UPCAST, not REDUCE.
    assert after_ranges == {0: AxisType.UPCAST}
    assert after is not sink

  def test_non_unit_stride_axis_is_left_untouched(self):
    # index by 2*i: stride 2, not coalesced -> must not be promoted.
    src = UOp.param(1, dtypes.float.ptr(8))
    out = UOp.param(0, dtypes.float.ptr(1))
    i = UOp.range(4, 0, AxisType.REDUCE)
    loaded = src.index(i * 2).load()
    red = loaded.reduce(i, arg=Ops.ADD)
    sink = out.index(UOp.const(dtypes.int, 0), ptr=True).store(red).sink()

    assert axis_stride(loaded.src[0].src[1], i) == 2
    after = coalesce_loads(sink)
    # negative control: no promotable axis found -> function is a no-op (same object back).
    assert after is sink
    ranges = {u.arg[0]: u.arg[-1] for u in after.toposort() if u.op is Ops.RANGE}
    assert ranges == {0: AxisType.REDUCE}

  def test_reg_accumulator_axis_is_never_promoted(self):
    # a REG-store target's own index range must be treated as an accumulator/carry axis, never coalesced,
    # even if it happens to be unit-stride.
    reg = UOp.placeholder((4,), dtypes.float, 10, addrspace=AddrSpace.REG)
    src = UOp.param(1, dtypes.float.ptr(4))
    i = UOp.range(4, 1, AxisType.LOOP)
    store = reg.index(i).store(src.index(i).load())
    sink = store.end(i).sink()
    after = coalesce_loads(sink)
    ranges = {u.arg[0]: u.arg[-1] for u in after.toposort() if u.op is Ops.RANGE}
    assert ranges == {1: AxisType.LOOP}

class TestCoalesceLoadsPipelineGate(unittest.TestCase):
  """Pipeline-level positive control: enable COALESCED_LOAD_LOWERING (default off) and prove the generated
  program differs. Uses the compile-only AMD:ISA oracle renderer -- no GPU involved."""

  def setUp(self):
    self._saved = os.environ.get("COALESCED_LOAD_LOWERING")

  def tearDown(self):
    _set_gate(self._saved)

  def _vectorized_load_count(self, sink: UOp) -> int:
    return sum(1 for u in sink.toposort() if u.op is Ops.LOAD and u.dtype.count > 1)

  def test_gate_off_by_default_leaves_load_scalar(self):
    _set_gate(None)
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    fs = full_rewrite_to_sink(_sum4_ast(), ren, optimize=True)
    assert self._vectorized_load_count(fs) == 0

  def test_gate_on_vectorizes_the_load_on_amd(self):
    _set_gate("1")
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    fs = full_rewrite_to_sink(_sum4_ast(), ren, optimize=True)
    # positive control: enabling the (default-off) gate on the AMD path actually changes the emitted AST --
    # the unit-stride load axis folds into a wide (vector) load.
    assert self._vectorized_load_count(fs) > 0


if __name__ == "__main__":
  unittest.main()
