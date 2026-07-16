from types import SimpleNamespace

import pytest

from tinygrad.llm.model import _attach_selected_prefill_inventory
from tinygrad.llm.prefill_route_census import (PrefillRouteAttachment, collect_prefill_route_census,
                                                prefill_forward_scope, record_prefill_route)

def _attachment(invocation_id, route_id, tensor_identity=None):
  return PrefillRouteAttachment(invocation_id, route_id, tensor_identity or invocation_id, {"route": route_id}, {"target": "scan"})

def test_all_selected_policy_route_classes_are_censused_exactly():
  routes = {"direct": "direct-packed", "overlay": "resident-overlay", "graph": "graph-gemm",
            "bounded": "bounded-packed-tiles", "fallback": "direct-packed-fallback"}
  linears = [SimpleNamespace(_prefill_route_attachment=_attachment(key, route)) for key, route in routes.items()]
  with collect_prefill_route_census(tuple(routes)) as census, prefill_forward_scope():
    for linear in linears: record_prefill_route(linear)
  artifact = census.artifact()
  assert artifact["status"] == "PASS" and artifact["complete"] is True
  assert {row["invocation_id"]: row["route_id"] for row in artifact["rows"]} == routes
  assert all(row["call_count"] == row["expected_call_count"] == 1 for row in artifact["rows"])

def test_missing_duplicate_and_unexpected_rows_fail_closed():
  with collect_prefill_route_census(("a", "b")) as census, prefill_forward_scope():
    record_prefill_route(SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a")))
    record_prefill_route(SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a")))
    record_prefill_route(SimpleNamespace(_prefill_route_attachment=_attachment("unexpected", "route-x")))
  artifact = census.artifact()
  assert artifact["status"] == "FAIL" and artifact["complete"] is False
  assert "duplicate invocation_id" in artifact["blocker"]
  assert "unexpected invocation_id" in artifact["blocker"]
  assert "missing invocation_ids" in artifact["blocker"]

def test_decode_or_out_of_scope_calls_do_not_pollute_prefill_census():
  linear = SimpleNamespace(_prefill_route_attachment=_attachment("a", "route-a"))
  with collect_prefill_route_census(("a",)) as census:
    record_prefill_route(linear)
    with prefill_forward_scope(False): record_prefill_route(linear)
    with prefill_forward_scope(True): record_prefill_route(linear)
    record_prefill_route(linear)
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
  with collect_prefill_route_census(tuple(routes)) as census, prefill_forward_scope():
    record_prefill_route(controlled); record_prefill_route(output)
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
