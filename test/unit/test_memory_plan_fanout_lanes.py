from types import SimpleNamespace

from tinygrad.dtype import dtypes
from tinygrad.schedule.memory import _fanout_lanes, _independent_reuse_lanes
from tinygrad.uop.ops import UOp


def _buf(size=4096):
  return UOp.new_buffer("CPU", size, dtypes.float)


def _call(*bufs):
  return SimpleNamespace(src=(None,) + bufs)


def test_sibling_fanout_outputs_get_distinct_lanes():
  common = _buf()
  q, k, v = _buf(), _buf(), _buf()
  linear = SimpleNamespace(src=(
    _call(common),
    _call(common, q),
    _call(common, k),
    _call(common, v),
  ))
  first = {common: 0, q: 1, k: 2, v: 3}
  last = {common: 0, q: 5, k: 6, v: 7}

  assert _fanout_lanes(linear, first, last) == {q: 1, k: 2, v: 3}


def test_empty_input_group_and_single_producer_are_not_split():
  a, b = _buf(), _buf()
  linear = SimpleNamespace(src=(_call(a), _call(b)))
  first = {a: 0, b: 1}
  last = {a: 0, b: 1}

  assert _fanout_lanes(linear, first, last) == {}


def test_non_overlapping_lifetimes_are_not_split():
  common = _buf()
  a, b = _buf(), _buf()
  linear = SimpleNamespace(src=(
    _call(common),
    _call(common, a),
    _call(common, b),
  ))
  first = {common: 0, a: 1, b: 2}
  last = {common: 0, a: 1, b: 2}

  assert _fanout_lanes(linear, first, last) == {}


def test_independent_reuse_of_dead_branch_is_moved_to_its_own_lane():
  produced = _buf()
  consumed = _buf()
  independent = _buf()
  linear = SimpleNamespace(src=(
    _call(produced),
    _call(produced, consumed),
    _call(independent),
  ))
  first = {produced: 0, consumed: 1, independent: 2}
  last = {produced: 1, consumed: 1, independent: 2}

  lanes = _independent_reuse_lanes(linear, first, last)
  assert lanes[independent] == 1
  assert lanes[produced] == 0 and lanes[consumed] == 0


def test_dependent_reuse_is_not_moved_to_its_own_lane():
  produced = _buf()
  dependent = _buf()
  linear = SimpleNamespace(src=(
    _call(produced),
    _call(produced, dependent),
  ))
  first = {produced: 0, dependent: 1}
  last = {produced: 1, dependent: 1}

  # dependent reads produced, so it is not independent and can reuse the lane.
  lanes = _independent_reuse_lanes(linear, first, last)
  assert lanes[dependent] == 0
