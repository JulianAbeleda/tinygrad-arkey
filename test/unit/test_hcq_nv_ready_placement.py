"""Hermetic CPU tests for the generic NV readiness-based multi-queue placement.
The primitive replaces the name-pinned admission with a dependency-readiness decision:
a node that directly depends on the primary queue's current tail stays on the primary;
every other node goes to the least-loaded GPFIFO (ties seed the primary so the whole DAG
cannot collapse onto one aux queue). The readiness signal comes from a read-only dep probe
so placement never mutates the recorded dependency maps."""
import collections
from types import SimpleNamespace

from tinygrad.engine.jit import DepsTracker
from tinygrad.runtime.graph import hcq


class FakeBuf:
  def __init__(self, base, offset=0, nbytes=64):
    self.base, self.offset, self.nbytes = base, offset, nbytes


def _runner(primary_load=0, secondary_load=0, tail=None):
  primary, secondary = object(), object()
  dev = type("FakeDev", (), {"device": "NV"})()
  return SimpleNamespace(
    compute_queues={dev: [primary, secondary]},
    compute_queue_load=collections.defaultdict(int, {primary: primary_load, secondary: secondary_load}),
    last_j=collections.defaultdict(lambda: None, {primary: tail} if tail is not None else {}),
    nv_multi_queue_indices=frozenset(),
  ), primary, secondary


def _pick(runner, rdeps, monkey):
  monkey.setattr(hcq, "HCQ_NV_READY_PLACEMENT", 1)
  monkey.setattr(hcq, "NV_MULTI_QUEUE_PROGRAMS", frozenset())
  monkey.setattr(hcq, "NV_MULTI_QUEUE_INDICES", frozenset())
  dev = next(iter(runner.compute_queues))
  return hcq.HCQGraph._pick_compute_queue(runner, dev, SimpleNamespace(name="x"), -1, rdeps)


def test_peek_is_read_only():
  dt = DepsTracker()
  a, b = FakeBuf(object()), FakeBuf(object())
  dt.access_resources([a], [0], ("primary", 1))
  before = (dict(dt.w_dependency_map), dict(dt.r_dependency_map))
  assert dt.peek_access_resources([a], [0]) == [("primary", 1)]
  assert dict(dt.w_dependency_map) == before[0] and dict(dt.r_dependency_map) == before[1]
  assert dt.access_resources([b], [0], ("primary", 2)) == []
  assert dt.peek_access_resources([a], [0]) == [("primary", 1)]


def test_peek_sees_write_after_read():
  dt = DepsTracker()
  a = FakeBuf(object())
  dt.access_resources([a], [0], ("primary", 1))
  dt.access_resources([a], [], ("primary", 2))
  assert sorted(map(str, dt.peek_access_resources([a], [0]))) == sorted(map(str, [("primary", 1), ("primary", 2)]))


def test_tail_dependent_stays_primary(monkeypatch):
  runner, primary, _ = _runner(primary_load=0, secondary_load=0, tail=4)
  assert _pick(runner, [(primary, 5)], monkeypatch) is primary
  assert _pick(runner, [(primary, 5), ("secondary", 9)], monkeypatch) is primary


def test_non_tail_dependent_goes_to_least_loaded(monkeypatch):
  runner, primary, secondary = _runner(primary_load=5, secondary_load=0, tail=4)
  assert _pick(runner, [(primary, 3)], monkeypatch) is secondary
  assert _pick(runner, [], monkeypatch) is secondary
  runner, primary, secondary = _runner(primary_load=0, secondary_load=5, tail=4)
  assert _pick(runner, [("secondary", 7)], monkeypatch) is primary


def test_tie_seeds_primary(monkeypatch):
  runner, primary, secondary = _runner(primary_load=0, secondary_load=0, tail=4)
  assert _pick(runner, [], monkeypatch) is primary


def test_gate_off_keeps_name_pinned_path(monkeypatch):
  runner, primary, secondary = _runner(primary_load=5, secondary_load=0)
  monkeypatch.setattr(hcq, "HCQ_NV_READY_PLACEMENT", 0)
  monkeypatch.setattr(hcq, "NV_MULTI_QUEUE_PROGRAMS",
    frozenset({"support_quant_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}))
  monkeypatch.setattr(hcq, "NV_MULTI_QUEUE_INDICES", frozenset())
  dev = next(iter(runner.compute_queues))
  assert hcq.HCQGraph._pick_compute_queue(runner, dev, SimpleNamespace(name="other"), -1, None) is primary
  admitted = SimpleNamespace(name="support_quant_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
  assert hcq.HCQGraph._pick_compute_queue(runner, dev, admitted, -1, None) is secondary
