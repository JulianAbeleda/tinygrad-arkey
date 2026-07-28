from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

from tinygrad.codegen.late.list_scheduler import _STRUCTURAL, list_schedule


def _block_ops():
  c = UOp.const(dtypes.float, 1.0)
  cross_lane = UOp(Ops.CUSTOMI, dtypes.float, (c,), arg="bpermute")
  dependent = cross_lane + c
  independent = c + c
  return cross_lane, dependent, independent


def test_list_scheduler_fills_latency_shadow_without_changing_dependencies():
  cross_lane, dependent, independent = _block_ops()
  original = [cross_lane, dependent, independent]
  scheduled = list_schedule(original)
  assert set(scheduled) == set(original)
  assert scheduled.index(cross_lane) < scheduled.index(dependent)
  assert scheduled.index(independent) < scheduled.index(dependent)


def test_list_scheduler_keeps_structural_boundaries_fixed():
  cross_lane, dependent, independent = _block_ops()
  barrier = UOp(Ops.BARRIER, dtypes.void)
  original = [cross_lane, dependent, barrier, independent]
  scheduled = list_schedule(original)
  assert set(scheduled) == set(original)
  assert scheduled.index(barrier) == original.index(barrier)
  assert scheduled.index(dependent) < scheduled.index(barrier)
  assert scheduled.index(independent) > scheduled.index(barrier)


def test_structural_inventory_contains_control_and_definition_ops():
  assert Ops.RANGE in _STRUCTURAL
  assert Ops.END in _STRUCTURAL
  assert Ops.BARRIER in _STRUCTURAL
  assert Ops.DEFINE_REG in _STRUCTURAL


def test_list_scheduler_core_owns_boundary():
  import importlib.util
  assert importlib.util.find_spec("tinygrad.codegen.experimental") is None
  assert importlib.util.find_spec("extra.llm_research.codegen_list_scheduler") is None
