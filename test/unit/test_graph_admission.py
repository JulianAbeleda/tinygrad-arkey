import hashlib, json, types
from unittest.mock import patch
import pytest

from tinygrad.engine import jit
from tinygrad.engine.jit import CapturedJit, GraphAdmission, GraphAdmissionCensus, GraphAdmissionDecision, GraphAdmissionReason, GraphException, \
  GraphAdmissionResource, GraphRunner, MultiGraphRunner, graph_split_rewrite, observe_graph_admissions
from tinygrad.helpers import Context, Metadata
from tinygrad.llm.model_facts import ProgramIdentityMetadata
from tinygrad.tensor import role_metadata
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, all_metadata


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


class _HybridSyntheticGraph(GraphRunner):
  @staticmethod
  def admission(_batch_devs, _new_call):
    resource = GraphAdmissionResource(2, 7, 0x100000000, 64)
    return GraphAdmission(True, GraphAdmissionReason.BACKEND_BUFFER_OFFSET_WIDTH, "icb_buffer_offset_bits", 0xFFFFFFFF,
                          0x100000000, (resource,))


class _HybridCensusScaleGraph(GraphRunner):
  @staticmethod
  def admission(_batch_devs, new_call):
    if int(new_call.name) < 736: return GraphAdmission(True, GraphAdmissionReason.ADMITTED)
    resource = GraphAdmissionResource(2, 7, 0x100000000, 64)
    return GraphAdmission(True, GraphAdmissionReason.BACKEND_BUFFER_OFFSET_WIDTH, "icb_buffer_offset_bits", 0xFFFFFFFF,
                          0x100000000, (resource,))


def _call(name="kernel", *, op=Ops.PROGRAM, device="SYNTHETIC", metadata=(), compiled=False):
  program = types.SimpleNamespace(op=op, arg=types.SimpleNamespace(name=name), key=bytes.fromhex("12" * 32))
  if compiled: program.src = (types.SimpleNamespace(op=Ops.SOURCE, arg="kernel source"),
                              types.SimpleNamespace(op=Ops.BINARY, arg=b"compiled binary"))
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


def test_observation_records_compiled_source_and_binary_content_identities():
  observed = []
  _split(_Linear([_call("a", compiled=True), _call("b", compiled=True)]), observer=observed.append)
  assert all(row.source_sha256 == hashlib.sha256(b"kernel source").hexdigest() for row in observed)
  assert all(row.binary_sha256 == hashlib.sha256(b"compiled binary").hexdigest() for row in observed)


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
                               "graph_batches":2, "constructor_failures":0, "semantic_calls":0, "generic_calls":4}
  assert payload["workload_role_histogram"] == {"generic":4}
  assert payload["batches"] == [{"batch_index":0, "size":2}, {"batch_index":1, "size":2}]
  assert payload["batch_boundary_histogram"] == {"batch_size_limit":1}
  assert payload["records"][0]["program_hash"] == "12" * 32
  assert payload["records"][0]["metadata"] == [{"name":"decode_projection", "caller":"fixture", "backward":False}]
  assert payload["records"][2]["batch_boundary_reason"] == "batch_size_limit"
  assert [record["batch_member_index"] for record in payload["records"]] == [0, 1, 0, 1]


def test_census_transports_rich_metadata_without_parsing_program_names():
  identity = ProgramIdentityMetadata(name="ffn_down", caller="", phase="decode", tensor_name="blk.0.ffn_down.weight",
    module_path="blk.0.ffn_down", role="ffn_down", logical_m=1, logical_n=4, logical_k=8,
    source_quant_storage="Q6_K", source_layout="gguf_packed_row_major", module_representation="nn_linear",
    input_dtype="float16", output_dtype="float16")
  linear = _Linear([_call("opaque_backend_name", metadata=(Metadata("generic", "fixture"), identity))])
  with observe_graph_admissions() as census: _lower(linear)
  record = census.to_dict()["records"][0]
  assert record["program_name"] == "opaque_backend_name"
  assert record["metadata_status"] == "semantic" and record["metadata_unavailable"] is False
  assert record["workload_roles"] == ["ffn_down"]
  assert record["semantic_identities"] == [{"phase":"decode", "tensor_name":"blk.0.ffn_down.weight", "module_path":"blk.0.ffn_down",
    "role":"ffn_down", "logical_m":1, "logical_n":4, "logical_k":8, "source_quant_storage":"Q6_K",
    "source_layout":"gguf_packed_row_major", "module_representation":"nn_linear", "input_dtype":"float16",
    "output_dtype":"float16", "accumulator_dtype":None}]


def test_semantic_accounting_is_stable_across_control_hybrid_and_repeated_traces():
  identity = ProgramIdentityMetadata(name="ffn_down", caller="", phase="decode", tensor_name="blk.0.ffn_down.weight",
    module_path="blk.0.ffn_down", role="ffn_down", logical_m=1, logical_n=4, logical_k=8,
    source_quant_storage="Q6_K", source_layout="gguf_packed_row_major", module_representation="nn_linear",
    input_dtype="float16", output_dtype="float16", accumulator_dtype="float32")
  linear = _Linear([_call("opaque_a", metadata=(identity,), compiled=True), _call("opaque_b", compiled=True)])
  def capture(graph):
    census = GraphAdmissionCensus(); _split(linear, observer=census, graph=graph); return census
  control, repeated, hybrid = capture(_SyntheticGraph), capture(_SyntheticGraph), capture(_HybridSyntheticGraph)
  assert control.deterministic_json() == repeated.deterministic_json()
  control_payload, hybrid_payload = control.to_dict(), hybrid.to_dict()
  assert control_payload["counts"]["semantic_calls"] == hybrid_payload["counts"]["semantic_calls"] == 1
  assert control_payload["counts"]["generic_calls"] == hybrid_payload["counts"]["generic_calls"] == 1
  assert control_payload["workload_role_histogram"] == hybrid_payload["workload_role_histogram"] == {"ffn_down":1, "generic":1}
  stable_fields = ("program_hash", "program_name", "source_sha256", "binary_sha256", "metadata_status", "workload_roles", "semantic_identities")
  assert [[record[field] for field in stable_fields] for record in control_payload["records"]] == \
         [[record[field] for field in stable_fields] for record in hybrid_payload["records"]]


def test_runtime_tracemeta_context_controls_explicit_metadata_without_import_time_mode():
  tagged = Metadata("runtime_context", "fixture")
  with Context(TRACEMETA=0), role_metadata(tagged): disabled = Tensor.empty(1) + 1
  with Context(TRACEMETA=1), role_metadata(tagged): enabled = Tensor.empty(1) + 1
  assert disabled.uop not in all_metadata
  assert all_metadata[enabled.uop] == (tagged,)


def test_supported_backend_limit_is_graphed_and_remains_visible_in_census():
  with observe_graph_admissions() as census: _split(_Linear([_call("a"), _call("b")]), observer=census, graph=_HybridSyntheticGraph)
  payload = census.to_dict()
  assert payload["counts"]["graph_members"] == 2 and payload["counts"]["direct_calls"] == 0
  assert payload["reason_histogram"] == {"admitted":2}
  assert payload["admission_reason_histogram"] == {"backend_buffer_offset_width":2}
  assert all(row["supported"] is True and row["decision"] == "admitted" and row["assignment"] == "graph" for row in payload["records"])
  assert all(row["admission_reason"] == "backend_buffer_offset_width" and row["capability"] == "icb_buffer_offset_bits"
             for row in payload["records"])


def test_hybrid_census_scale_projects_five_batches_without_stranded_singletons():
  census = GraphAdmissionCensus()
  _split(_Linear([_call(str(i)) for i in range(803)]), observer=census, max_batch_size=32, graph=_HybridCensusScaleGraph)
  payload = census.to_dict()
  assert payload["counts"] == {"logical_calls":803, "graph_members":803, "direct_calls":0, "ignored_slice_nodes":0,
                               "graph_batches":5, "constructor_failures":0, "semantic_calls":0, "generic_calls":803}
  assert [batch["size"] for batch in payload["batches"]] == [32, 64, 128, 256, 323]
  assert payload["admission_reason_histogram"] == {"admitted":736, "backend_buffer_offset_width":67}
  assert payload["reason_histogram"] == {"admitted":803}


def test_constructor_failure_is_captured_separately_and_original_error_survives():
  captured = CapturedJit("ret", _Linear(()), [], [])
  with observe_graph_admissions(GraphAdmissionCensus()) as census, patch.object(jit, "run_linear", side_effect=GraphException("fixture failure")):
    with pytest.raises(GraphException, match="fixture failure"): captured([], {})
  payload = census.to_dict()
  assert payload["counts"]["logical_calls"] == 0 and payload["counts"]["constructor_failures"] == 1
  assert payload["constructor_failures"] == [{"error_type":"GraphException", "message":"fixture failure"}]
