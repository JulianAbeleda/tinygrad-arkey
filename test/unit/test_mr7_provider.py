import io
import json

import pytest

from extra.llm_research.finalized_census import validate_finalized_census
from extra.llm_research.mr7_provider import PROTOCOL, IDENTITY_FIELDS_SHA256, serve, validate_request
from tinygrad.engine.metadata import PROGRAM_IDENTITY_FIELDS
from tinygrad.engine.jit import GraphAdmissionCensus
from tinygrad.uop.ops import Ops, UOp


def _identity(tensor, role):
  return {"phase":"decode", "tensor_name":tensor, "module_path":tensor.removesuffix(".weight"), "role":role,
    "logical_m":1, "logical_n":256, "logical_k":256, "source_quant_storage":"Q4_K",
    "source_layout":"gguf_packed_row_major", "module_representation":"nn_linear",
    "input_dtype":"float16", "output_dtype":"float16", "accumulator_dtype":"float32"}


def _row(index, identities, *, program="a", batch=0):
  return {"call_index":index, "program_hash":program*64, "source_sha256":chr(ord(program)+1)*64,
    "binary_sha256":chr(ord(program)+2)*64, "semantic_identities":identities, "assignment":"graph",
    "batch_index":batch, "batch_member_index":index, "batch_size":2, "direct_call_index":None,
    "metadata_status":"semantic", "workload_roles":list(dict.fromkeys(x["role"] for x in identities)),
    "decision":"admitted", "reason":"admitted", "admission_reason":"admitted", "batch_boundary_reason":None}


def _census():
  rows = [_row(0, [_identity("blk.0.ffn_down.weight", "ffn_down"),
                   _identity("blk.0.attn_output.weight", "attn_qo")]),
          _row(1, [_identity("blk.1.ffn_down.weight", "ffn_down")], program="d")]
  return {"schema":"tinygrad.graph_admission_census.v1", "records":rows,
    "counts":{"logical_calls":2,"graph_members":2,"direct_calls":0,"ignored_slice_nodes":0,"graph_batches":1,
              "constructor_failures":0,"semantic_calls":2,"generic_calls":0},
    "reason_histogram":{"admitted":2}, "admission_reason_histogram":{"admitted":2},
    "batch_boundary_histogram":{}, "workload_role_histogram":{"attn_qo":1,"ffn_down":2},
    "batches":[{"batch_index":0,"size":2}],
    "constructor_failures":[]}


def _mixed_census():
  census = _census(); generic = census["records"][1]
  generic.update({"semantic_identities":[], "metadata_status":"generic", "workload_roles":["generic"]})
  census["counts"].update({"semantic_calls":1, "generic_calls":1})
  census["workload_role_histogram"] = {"attn_qo":1,"ffn_down":1,"generic":1}
  return census


def _execution(row):
  return {key:row[key] for key in ("program_hash", "source_sha256", "binary_sha256")} | {
    "semantic_identities":sorted(row["semantic_identities"], key=lambda value:json.dumps(value, sort_keys=True, separators=(",", ":")))}


def test_request_validation_preserves_fused_identity_and_complete_outer_membership():
  census = _census(); first = census["records"][0]
  isolated = {"kind":"exact_isolated_execution", "call_indices":[0], "shape_mode":"exact_workload",
    "correctness_required":True, "execution_identity":_execution(first)}
  kind, rows = validate_request(census, isolated)
  assert kind == "exact_isolated_execution" and rows == [first] and len(rows[0]["semantic_identities"]) == 2
  outer = {"kind":"complete_outer_enclosure", "outer_id":"graph:0", "call_indices":[0, 1],
    "shape_mode":"complete_outer", "no_member_duration_splitting":True}
  assert validate_request(census, outer)[1] == census["records"]
  with pytest.raises(ValueError, match="complete census enclosure"):
    validate_request(census, dict(outer, call_indices=[0]))
  tampered = json.loads(json.dumps(isolated)); tampered["execution_identity"]["program_hash"] = "0" * 64
  with pytest.raises(ValueError, match="differs from census"): validate_request(census, tampered)


def test_persistent_jsonl_server_reuses_one_session_and_binds_response_ids():
  class FakeSession:
    def __init__(self): self.calls = []
    def execute(self, request): self.calls.append(request["request_id"]); return {"echo":request["request_id"]}
  session = FakeSession()
  rows = [{"protocol":PROTOCOL, "plan_sha256":"a"*64, "request_id":name,
           "request":{"request_id":name, "kind":"fake"}} for name in ("one", "two")]
  source, sink = io.StringIO("".join(json.dumps(row)+"\n" for row in rows)), io.StringIO()
  assert serve(source, sink, session) == 0 and session.calls == ["one", "two"]
  responses = [json.loads(line) for line in sink.getvalue().splitlines()]
  assert [row["request_id"] for row in responses] == ["one", "two"]
  assert all(row["status"] == "ok" and row["protocol"] == PROTOCOL for row in responses)


def test_server_fail_closed_response_does_not_terminate_following_request():
  class FailOnce:
    def __init__(self): self.count = 0
    def execute(self, request):
      self.count += 1
      if self.count == 1: raise ValueError("bad request")
      return {"ok":True}
  rows = [{"protocol":PROTOCOL, "plan_sha256":"a"*64, "request_id":name,
           "request":{"request_id":name}} for name in ("bad", "good")]
  sink = io.StringIO(); serve(io.StringIO("".join(json.dumps(row)+"\n" for row in rows)), sink, FailOnce())
  responses = [json.loads(line) for line in sink.getvalue().splitlines()]
  assert [row["status"] for row in responses] == ["blocked", "ok"]
  assert responses[0]["error"]["code"] == "provider_failure"


def test_graph_census_runtime_bindings_never_enter_serialized_authority():
  census = GraphAdmissionCensus(); call = UOp(Ops.NOOP); linear = UOp(Ops.LINEAR, src=(call,)); inp = UOp(Ops.NOOP)
  census.bind_call(0, call); census.bind_execution(linear, (inp,), {"start_pos":7})
  assert census.calls == {0:call} and census.execution_linear is linear and census.execution_inputs == (inp,)
  serialized = census.to_dict()
  assert all(key not in serialized for key in ("calls", "execution_linear", "execution_inputs", "execution_var_vals"))


def test_finalized_census_requires_reconciliation_known_decisions_and_material_identity():
  census = _census(); validate_finalized_census(census); validate_finalized_census(_mixed_census())
  for mutate, match in (
    (lambda value: value["counts"].__setitem__("logical_calls", 3), "reconcile"),
    (lambda value: value["records"][0].__setitem__("reason", "unknown"), "unknown"),
    (lambda value: value["records"][0].__setitem__("binary_sha256", None), "program identity"),
    (lambda value: value["records"][0]["semantic_identities"][0].pop("accumulator_dtype"), "semantic identity"),
    (lambda value: value["counts"].__setitem__("constructor_failures", 1), "constructor failures")):
    tampered = json.loads(json.dumps(census)); mutate(tampered)
    with pytest.raises(ValueError, match=match): validate_finalized_census(tampered)


@pytest.mark.parametrize("mutate,match", (
  (lambda value: value["records"][0].__setitem__("decision", "forged"), "unknown decisions"),
  (lambda value: value["records"][0].__setitem__("reason", "backend_resource_limit"), "graph assignment"),
  (lambda value: value["records"][0].update({"assignment":"ignored", "decision":"admitted"}), "ignored assignment"),
  (lambda value: value["records"][1].__setitem__("batch_index", 99), "batch coordinates"),
  (lambda value: value["records"][1].__setitem__("batch_member_index", 0), "batch coordinates"),
  (lambda value: value["records"][1].__setitem__("batch_size", 3), "batch sizes"),
  (lambda value: value["batches"][0].__setitem__("size", 99), "batches"),
  (lambda value: value["counts"].__setitem__("semantic_calls", 99), "derived counts"),
  (lambda value: value["counts"].__setitem__("graph_batches", 99), "derived counts"),
  (lambda value: value["reason_histogram"].__setitem__("admitted", 99), "reason histogram"),
  (lambda value: value["admission_reason_histogram"].__setitem__("admitted", 99), "admission histogram"),
  (lambda value: value["batch_boundary_histogram"].__setitem__("forged", 1), "boundary histogram"),
  (lambda value: value["workload_role_histogram"].__setitem__("generic", 99), "workload-role histogram"),
))
def test_finalized_census_rejects_forged_derived_authority(mutate, match):
  census = _census(); mutate(census)
  with pytest.raises(ValueError, match=match): validate_finalized_census(census)


@pytest.mark.parametrize("field,value", (
  ("phase", "training"), ("role", "forged"), ("source_quant_storage", "F32"),
  ("source_layout", "forged"), ("module_representation", "forged"),
  ("input_dtype", "int8"), ("output_dtype", "int8"), ("accumulator_dtype", "int8"),
))
def test_finalized_census_round_trips_identity_through_typed_authority(field, value):
  census = _census(); census["records"][0]["semantic_identities"][0][field] = value
  with pytest.raises(ValueError, match="semantic identity"): validate_finalized_census(census)


def test_finalized_census_reconciles_workload_roles_with_semantic_identities():
  census = _census(); census["records"][0]["workload_roles"] = ["generic"]
  with pytest.raises(ValueError, match="ambiguous identity"): validate_finalized_census(census)


def test_mr7_provider_response_envelope_declares_identity_fields_sha256():
  class FakeSession:
    def execute(self, request): return {"echo":"test"}
  session = FakeSession()
  rows = [{"protocol":PROTOCOL, "plan_sha256":"a"*64, "request_id":"test",
           "request":{"request_id":"test", "kind":"fake"}}]
  source, sink = io.StringIO(json.dumps(rows[0])+"\n"), io.StringIO()
  assert serve(source, sink, session) == 0
  response = json.loads(sink.getvalue().splitlines()[0])
  assert response["identity_fields"] == list(PROGRAM_IDENTITY_FIELDS)
  assert response["identity_fields_sha256"] == IDENTITY_FIELDS_SHA256
  assert len(response["identity_fields_sha256"]) == 64
