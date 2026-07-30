import json, types
from unittest.mock import patch
import pytest

from tinygrad.engine import jit
from tinygrad.engine.jit import CapturedJit, GraphAdmission, GraphAdmissionCensus, GraphAdmissionDecision, GraphAdmissionReason, GraphException, \
  GraphRunner, MultiGraphRunner, graph_split_rewrite, observe_graph_admissions
from tinygrad.helpers import Context, Metadata
from tinygrad.uop.ops import Ops


class _Linear:
  def __init__(self, src): self.src = tuple(src)
  def replace(self, *, src): return _Linear(src)
  def substitute(self, *_args, **_kwargs): return self
  def toposort(self): return []


class _DeviceMap:
  class _Dev:
    def __init__(self, graph): self.graph = graph
  def __init__(self, graph): self.dev = self._Dev(graph)
  def __getitem__(self, _name): return self.dev


class _SyntheticGraph(GraphRunner):
  @staticmethod
  def admission(_batch_devs, _new_call): return GraphAdmission(True, GraphAdmissionReason.ADMITTED)


def _call(name="kernel", *, op=Ops.PROGRAM, device="SYNTHETIC", metadata=()):
  program = types.SimpleNamespace(op=op, arg=types.SimpleNamespace(name=name), key=bytes.fromhex("12" * 32))
  buf = types.SimpleNamespace(op=Ops.BUFFER, device=device)
  return types.SimpleNamespace(src=(program, buf), arg=types.SimpleNamespace(metadata=metadata), name=name)


def _split(linear, *, observer=None, max_batch_size=0, graph=_SyntheticGraph):
  with patch.object(jit, "Device", _DeviceMap(graph)), patch.object(jit, "create_graph_call", lambda batch: ("graph", tuple(batch))):
    return graph_split_rewrite(linear, max_batch_size=max_batch_size, observer=observer)


def _lower(linear, *, graph=_SyntheticGraph):
  with patch.object(jit, "Device", _DeviceMap(graph)), patch.object(jit, "create_graph_call", lambda batch: ("graph", tuple(batch))), \
       patch.object(jit, "memory_plan_rewrite", lambda value, _held: value), patch.object(jit, "compile_linear", lambda value: value):
    return jit.jit_lower(linear, set(), [])


def test_typed_generic_admission_is_canonical_for_boolean_compatibility():
  program, copy = _call(), _call(op=Ops.COPY)
  one, other = object(), object()
  with patch.object(GraphRunner, "_all_devs", side_effect=([one], [one], [one, other])):
    admitted = GraphRunner.admission([one], program)
    assert admitted == GraphAdmission(True, GraphAdmissionReason.ADMITTED)
    assert GraphRunner.supports_uop([one], program) is True
    mixed = GraphRunner.admission([one], program)
    assert mixed.reason is GraphAdmissionReason.MIXED_DEVICE and bool(mixed) is False
  with patch.object(GraphRunner, "_all_devs", return_value=[one]):
    unsupported = GraphRunner.admission([one], copy)
    assert unsupported.reason is GraphAdmissionReason.UNSUPPORTED_CALL_OP
    assert GraphRunner.supports_uop([one], copy) is False


def test_multigraph_boolean_delegates_to_typed_admission():
  call, dev = _call(op=Ops.COPY), object()
  with patch.object(GraphRunner, "_all_devs", return_value=[dev]):
    assert MultiGraphRunner.admission([dev], call).reason is GraphAdmissionReason.ADMITTED
    assert MultiGraphRunner.supports_uop([dev], call) is True


def test_observation_disabled_and_enabled_produce_identical_graph_output_and_record_batch_limit():
  linear = _Linear([_call(str(index)) for index in range(4)])
  control = _split(linear, max_batch_size=2)
  observed = []
  candidate = _split(linear, max_batch_size=2, observer=observed.append)
  assert control.src == candidate.src
  assert len(observed) == len(linear.src)
  assert all(row.decision is GraphAdmissionDecision.ADMITTED and row.admission.reason is GraphAdmissionReason.ADMITTED for row in observed)
  assert observed[2].batch_boundary_reason is GraphAdmissionReason.BATCH_SIZE_LIMIT
  assert sum(row.batch_boundary_reason is GraphAdmissionReason.BATCH_SIZE_LIMIT for row in observed) == 1
  assert [(row.assignment, row.batch_index, row.batch_member_index, row.batch_size) for row in observed] == [
    ("graph", 0, 0, 2), ("graph", 0, 1, 2), ("graph", 1, 0, 2), ("graph", 1, 1, 2)]


def test_prefix_barrier_is_accounted_as_boundary_not_backend_rejection(monkeypatch):
  monkeypatch.setenv("JIT_NO_GRAPH_KERNEL_PREFIXES", "barrier_")
  linear = _Linear([_call("first"), _call("barrier_exact"), _call("last")])
  observed = []
  control = _split(linear)
  candidate = _split(linear, observer=observed.append)
  assert control.src == candidate.src
  assert [row.decision for row in observed] == [GraphAdmissionDecision.ADMITTED, GraphAdmissionDecision.BATCH_BOUNDARY,
                                                GraphAdmissionDecision.ADMITTED]
  assert observed[1].admission.reason is GraphAdmissionReason.EXPLICIT_GRAPH_BARRIER
  assert observed[1].batch_boundary_reason is None
  assert observed[1].assignment == "direct" and observed[1].direct_call_index == 1


def test_no_graph_and_ignored_slice_are_explicit_and_reconciled():
  ignored = types.SimpleNamespace(src=(types.SimpleNamespace(op=Ops.SLICE),), name="ignored")
  linear = _Linear([ignored, _call("direct")])
  observed = []
  _split(linear, observer=observed.append, graph=None)
  assert len(observed) == len(linear.src)
  assert observed[0].decision is GraphAdmissionDecision.IGNORED and observed[0].admission.reason is GraphAdmissionReason.IGNORED_SLICE_NODE
  assert observed[1].decision is GraphAdmissionDecision.REJECTED and observed[1].admission.reason is GraphAdmissionReason.NO_GRAPH_BACKEND
  assert observed[0].assignment == "ignored"
  assert observed[1].assignment == "direct" and observed[1].direct_call_index == 0


def test_scoped_context_is_automatic_nested_and_restored():
  linear = _Linear([_call("outer")])
  with observe_graph_admissions() as outer:
    _lower(linear)
    with observe_graph_admissions() as inner: _lower(linear)
    assert len(outer.records) == 1 and len(inner.records) == 1
  _lower(linear)
  assert len(outer.records) == 1


def test_census_serialization_is_deterministic_reconciled_and_carries_available_identity():
  metadata = (Metadata("decode_projection", "fixture"),)
  linear = _Linear([_call(str(index), metadata=metadata if index == 0 else ()) for index in range(4)])
  with Context(JIT_BATCH_SIZE=2), observe_graph_admissions() as census:
    observed_linear = _lower(linear)
  control_linear = _split(linear, max_batch_size=2)
  assert observed_linear.src == control_linear.src
  payload = census.to_dict()
  assert json.loads(census.deterministic_json()) == payload
  assert census.deterministic_json() == census.deterministic_json()
  assert payload["counts"] == {"logical_calls":4, "graph_members":4, "direct_calls":0, "ignored_slice_nodes":0,
                               "graph_batches":2, "constructor_failures":0}
  assert payload["batches"] == [{"batch_index":0, "size":2}, {"batch_index":1, "size":2}]
  assert payload["batch_boundary_histogram"] == {"batch_size_limit":1}
  assert payload["records"][0]["program_hash"] == "12" * 32
  assert payload["records"][0]["metadata"] == [{"name":"decode_projection", "caller":"fixture", "backward":False}]
  assert payload["records"][2]["batch_boundary_reason"] == "batch_size_limit"
  assert [record["batch_member_index"] for record in payload["records"]] == [0, 1, 0, 1]


def test_constructor_failure_is_captured_separately_and_original_error_survives():
  captured = CapturedJit("ret", _Linear(()), [], [])
  with observe_graph_admissions(GraphAdmissionCensus()) as census, patch.object(jit, "run_linear", side_effect=GraphException("fixture failure")):
    with pytest.raises(GraphException, match="fixture failure"): captured([], {})
  payload = census.to_dict()
  assert payload["counts"]["logical_calls"] == 0 and payload["counts"]["constructor_failures"] == 1
  assert payload["constructor_failures"] == [{"error_type":"GraphException", "message":"fixture failure"}]
