import pytest
from types import SimpleNamespace

from extra.llm_research.decode.nv_dependency_closed_cut import schedule
from tinygrad.runtime.graph import hcq
from tinygrad.runtime.graph.hcq import _match_nv_multi_queue_cut_policy, _parse_nv_multi_queue_indices


def test_index_selector_ranges():
  assert _parse_nv_multi_queue_indices("") == frozenset()
  assert _parse_nv_multi_queue_indices("1,3-5,4") == frozenset((1, 3, 4, 5))
  with pytest.raises(ValueError): _parse_nv_multi_queue_indices("5-3")
  with pytest.raises(ValueError): _parse_nv_multi_queue_indices("nope")


def test_index_selector_pins_auxiliary_queue(monkeypatch):
  primary, secondary = object(), object()
  class FakeDev: device = "NV"
  dev = FakeDev()
  runner = SimpleNamespace(compute_queues={dev:[primary, secondary]}, compute_queue_load={primary:0, secondary:999},
                           nv_multi_queue_indices=frozenset((7,)))
  monkeypatch.setattr(hcq, "NV_MULTI_QUEUE_INDICES", frozenset((7,)))
  monkeypatch.setattr(hcq, "NV_MULTI_QUEUE_PROGRAMS", frozenset())
  assert hcq.HCQGraph._pick_compute_queue(runner, dev, SimpleNamespace(name="x"), 7) is secondary
  assert hcq.HCQGraph._pick_compute_queue(runner, dev, SimpleNamespace(name="x"), 8) is primary


def test_cut_policy_matches_exact_prefix_and_names():
  names = ["a", "b", "c", "diagnostic-extra"]
  import hashlib
  row = {"prefix_count":3, "prefix_name_digest":hashlib.sha256("a\nb\nc".encode()).hexdigest(),
         "selected":[{"index":1, "identity":"b"}]}
  assert _match_nv_multi_queue_cut_policy(names, [row]) == frozenset((1,))
  assert _match_nv_multi_queue_cut_policy(["a", "changed", "c"], [row]) == frozenset()


def test_signal_cache_charges_one_entrance_and_exit():
  # 0 forks to a three-node auxiliary chain and independent primary work; 5 joins.
  dag = {"nodes":[{"id":i, "name":str(i), "duration_us":d} for i,d in enumerate((1, 2, 3, 4, 8, 1))],
         "edges":[{"from":a, "to":b} for a,b in ((0,1),(1,2),(2,3),(0,4),(3,5),(4,5))]}
  row = schedule(dag, {1,2,3}, wait_cost_us=.5)
  assert row["wait_count"] == 2
  assert [(x["node"], x["waits_for"]) for x in row["wait_events"]] == [(1, 0), (5, 3)]


def test_signal_cache_does_not_charge_redundant_raw_edges():
  dag = {"nodes":[{"id":i, "name":str(i), "duration_us":1} for i in range(5)],
         "edges":[{"from":a, "to":b} for a,b in ((0,1),(0,2),(1,3),(2,3),(1,4),(3,4))]}
  row = schedule(dag, {1}, wait_cost_us=0)
  # Node 3 waits for aux signal 2. Node 4's edge from 1 is already covered.
  assert row["wait_count"] == 2
