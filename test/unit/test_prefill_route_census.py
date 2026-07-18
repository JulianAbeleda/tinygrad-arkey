from types import SimpleNamespace

import pytest

from tinygrad.llm.model import _attach_selected_prefill_inventory
from tinygrad.llm.prefill_route_observer import (PrefillRouteAttachment, PrefillRouteExecution,
  notify_prefill_route, notify_prefill_route_execution)
from extra.qk.prefill_route_census import collect_prefill_route_census, collect_prefill_route_execution_census

def _attachment(invocation_id, route_id, tensor_identity=None):
  return PrefillRouteAttachment(invocation_id, route_id, tensor_identity or invocation_id, {"route": route_id}, {"target": "scan"})

def test_all_selected_policy_route_classes_are_censused_exactly():
  routes = {"direct": "direct-packed", "overlay": "resident-overlay", "graph": "graph-gemm",
            "bounded": "bounded-packed-tiles", "fallback": "direct-packed-fallback"}
  linears = [SimpleNamespace(_prefill_route_attachment=_attachment(key, route)) for key, route in routes.items()]
  with collect_prefill_route_census(tuple(routes)) as census:
    for linear in linears: notify_prefill_route(linear)
  artifact = census.artifact()
  assert artifact["status"] == "PASS" and artifact["complete"] is True
  assert {row["invocation_id"]: row["route_id"] for row in artifact["rows"]} == routes
  assert all(row["call_count"] == row["expected_call_count"] == 1 for row in artifact["rows"])

def test_missing_duplicate_and_unexpected_rows_fail_closed():
  with collect_prefill_route_census(("a", "b")) as census:
    notify_prefill_route(SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a")))
    notify_prefill_route(SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a")))
    notify_prefill_route(SimpleNamespace(_prefill_route_attachment=_attachment("unexpected", "route-x")))
  artifact = census.artifact()
  assert artifact["status"] == "FAIL" and artifact["complete"] is False
  assert "duplicate invocation_id" in artifact["blocker"]
  assert "unexpected invocation_id" in artifact["blocker"]
  assert "missing invocation_ids" in artifact["blocker"]

def test_decode_or_out_of_scope_calls_do_not_pollute_prefill_census():
  linear = SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a"))
  from tinygrad.llm.prefill_route_observer import prefill_route_scope
  with collect_prefill_route_census(("a",)) as census:
    with prefill_route_scope(False): notify_prefill_route(linear)
    notify_prefill_route(linear)
  assert census.artifact()["rows"][0]["call_count"] == 1

def test_inventory_attachment_uses_exact_tensor_identity_and_preserves_authorities():
  first, second = SimpleNamespace(), SimpleNamespace()
  model = SimpleNamespace(blk=[SimpleNamespace(ffn_gate=first, ffn_up=second)])
  inventory = {"rows": [{"invocation_id": "i-up", "tensor_identity": "blk.0.ffn_up.weight"}]}
  policy, facts = {"routes": {"i-up": "overlay-selected"}}, object()
  _attach_selected_prefill_inventory(model, inventory, policy, facts)
  assert not hasattr(first, "_prefill_route_attachment")
  attachment = second._prefill_route_attachment
  assert attachment.invocation_id == "i-up" and attachment.tensor_identity == "blk.0.ffn_up.weight"
  assert attachment.selected_policy is policy and attachment.scanned_target_facts is facts

def test_fixed_runtime_row_is_attached_and_required_by_full_census():
  controlled, output = SimpleNamespace(), SimpleNamespace()
  model = SimpleNamespace(blk=[SimpleNamespace(ffn_gate=controlled)], output=output)
  rows = [{"invocation_id": "controlled", "tensor_identity": "blk.0.ffn_gate.weight", "candidate_controlled": True},
          {"invocation_id": "fixed", "tensor_identity": "output.weight", "candidate_controlled": False,
           "fixed_route_id": "fixed-ggml-linear"}]
  routes = {"controlled": "overlay", "fixed": "fixed-ggml-linear"}
  _attach_selected_prefill_inventory(model, {"rows": rows}, {"routes": routes}, object())
  with collect_prefill_route_census(tuple(routes)) as census:
    notify_prefill_route(controlled); notify_prefill_route(output)
  artifact = census.artifact()
  assert artifact["complete"] is True
  assert {row["invocation_id"]: row["route_id"] for row in artifact["rows"]} == routes

def test_inventory_attachment_fails_for_missing_duplicate_or_policy_mismatch():
  model = SimpleNamespace(blk=[SimpleNamespace(ffn_gate=SimpleNamespace())])
  row = {"invocation_id": "i", "tensor_identity": "blk.0.ffn_gate.weight"}
  with pytest.raises(ValueError, match="policy and inventory"):
    _attach_selected_prefill_inventory(model, {"rows": [row]}, {"routes": {}}, object())
  with pytest.raises(ValueError, match="no exact runtime linear"):
    _attach_selected_prefill_inventory(model, {"rows": [{**row, "tensor_identity": "blk.0.ffn_up.weight"}]},
                                       {"routes": {"i": "route"}}, object())
  with pytest.raises(ValueError, match="duplicate selected"):
    _attach_selected_prefill_inventory(model, {"rows": [row, row]}, {"routes": {"i": "route"}}, object())

def _execution(invocation_id, route_id, candidate="candidate-a", program="binary:abc", fallback=False, reason=None):
  return PrefillRouteExecution(invocation_id, route_id, candidate, program, fallback, reason)

def test_actual_execution_census_records_exact_candidate_program_and_fallback():
  first = SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a", "a.weight"))
  second = SimpleNamespace(_prefill_route_attachment=_attachment("b", "route-b", "b.weight"))
  with collect_prefill_route_execution_census(
      ("a", "b"), expected_candidate_counts={"candidate-a": 2}, expected_fallback_count=1) as census:
    notify_prefill_route_execution(first, _execution("a", "route-a"))
    notify_prefill_route_execution(second, _execution(
      "b", "route-b", program="binary:def", fallback=True, reason="guard rejected optimized program"))
  artifact = census.artifact()
  assert artifact["status"] == "PASS" and artifact["complete"] is True
  assert artifact["observed_candidate_counts"] == {"candidate-a": 2}
  assert artifact["observed_fallback_counts"] == {"used": 1, "not_used": 1}
  rows = {row["invocation_id"]: row for row in artifact["rows"]}
  assert rows["a"]["attached_route_id"] == rows["a"]["executed_route_id"] == "route-a"
  assert rows["a"]["candidate_identity"] == "candidate-a" and rows["a"]["program_identity"] == "binary:abc"
  assert rows["b"]["fallback_used"] is True and rows["b"]["fallback_reason"] == "guard rejected optimized program"

def test_actual_execution_census_fails_on_duplicate_unexpected_and_count_drift():
  linear = SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a"))
  with collect_prefill_route_execution_census(
      ("a", "b"), expected_candidate_counts={"candidate-a": 2}, expected_fallback_count=0) as census:
    notify_prefill_route_execution(linear, _execution("a", "route-a"))
    notify_prefill_route_execution(linear, _execution("a", "route-a"))
    unexpected = SimpleNamespace(_prefill_route_attachment=_attachment("x", "route-x"))
    notify_prefill_route_execution(unexpected, _execution("x", "route-x", candidate="candidate-x"))
  artifact = census.artifact()
  assert artifact["status"] == "FAIL" and artifact["complete"] is False
  assert "duplicate execution invocation_id" in artifact["blocker"]
  assert "unexpected execution invocation_id" in artifact["blocker"]
  assert "unexpected execution candidate_identity" in artifact["blocker"]
  assert "missing execution invocation_ids" in artifact["blocker"]
  assert "candidate execution counts differ" in artifact["blocker"]

@pytest.mark.parametrize("event, blocker", [
  (_execution("other", "route-a"), "attachment-vs-execution invocation mismatch"),
  (_execution("a", "other-route"), "attachment-vs-execution route mismatch"),
  (_execution("a", "route-a", fallback=True), "fallback execution requires a non-empty reason"),
  (_execution("a", "route-a", reason="not actually a fallback"), "non-fallback execution must not report a fallback reason"),
])
def test_actual_execution_census_rejects_attachment_mismatch_and_invalid_fallback(event, blocker):
  linear = SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a"))
  with collect_prefill_route_execution_census(
      ("a",), expected_candidate_counts={"candidate-a": 1}, expected_fallback_count=int(event.fallback_used)) as census:
    notify_prefill_route_execution(linear, event)
  artifact = census.artifact()
  assert artifact["status"] == "FAIL" and blocker in artifact["blocker"]

def test_actual_execution_census_is_context_local_and_requires_exact_expectations():
  linear = SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a"))
  from tinygrad.llm.prefill_route_observer import prefill_route_scope
  with collect_prefill_route_execution_census(
      ("a",), expected_candidate_counts={"candidate-a": 1}, expected_fallback_count=0) as census:
    with prefill_route_scope(False): notify_prefill_route_execution(linear, _execution("a", "route-a"))
    notify_prefill_route_execution(linear, _execution("a", "route-a"))
  assert census.artifact()["rows"][0]["execution_count"] == 1
  with pytest.raises(ValueError, match="equal total expected"):
    collect_prefill_route_execution_census(
      ("a",), expected_candidate_counts={"candidate-a": 2}, expected_fallback_count=0).__enter__()
  with pytest.raises(ValueError, match="within total expected"):
    collect_prefill_route_execution_census(
      ("a",), expected_candidate_counts={"candidate-a": 1}, expected_fallback_count=2).__enter__()
