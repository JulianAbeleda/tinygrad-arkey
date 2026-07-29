"""CPU-only contracts for selected prefill inventory attachment installation."""
from __future__ import annotations

import pytest

from tinygrad.llm.prefill_attachments import (PrefillDirectPackedBinding, PrefillRouteAttachment,
  attach_selected_prefill_inventory)
from tinygrad.llm.prefill_route_observer import (PrefillDirectPackedBinding as ObserverBinding,
  PrefillRouteAttachment as ObserverAttachment)
from tinygrad.llm.route_selection import RouteCandidatePolicy, RouteLifecycle


class _Linear: pass
class _Attention:
  def __init__(self): self.q, self.k = _Linear(), _Linear()
class _Block:
  def __init__(self): self.attn = _Attention()
class _Model:
  def __init__(self): self.blk = [_Block()]


def _inventory(rows): return {"rows": rows}
def _row(invocation_id, tensor, *, m=512, n=4096, k=4096, role="attn_qo"):
  return {"invocation_id": invocation_id, "tensor_identity": tensor, "role": role, "shape": {"m": m, "n": n, "k": k}}
def _policy():
  return {"routes": {"q": "route-q", "k": "route-k"},
          "bounded_packed_projection_proof": {"allocation_owner_identity": "owner"}}
def _direct(): return RouteCandidatePolicy("direct", RouteLifecycle.FALLBACK, "baseline")


def test_observer_reexports_exact_attachment_types():
  assert ObserverAttachment is PrefillRouteAttachment
  assert ObserverBinding is PrefillDirectPackedBinding


def test_selected_inventory_attaches_exact_fields_and_policy_identity():
  model, policy, facts = _Model(), _policy(), {"backend": "AMD"}
  attach_selected_prefill_inventory(model, _inventory([
    _row("q", "blk.0.attn.q.weight", role="attn_qo"), _row("k", "blk.0.attn.k.weight", n=1024, role="attn_kv")]),
    policy, facts, direct_packed_policy=_direct())
  q, k = model.blk[0].attn.q, model.blk[0].attn.k
  assert q._prefill_route_attachment == PrefillRouteAttachment("q", "route-q", "blk.0.attn.q.weight", policy, facts, "owner")
  assert q._prefill_route_attachment.selected_policy is policy
  assert q._prefill_direct_packed_binding == PrefillDirectPackedBinding("q", "prefill", "attn_qo", (512, 4096, 4096),
                                                                          RouteLifecycle.FALLBACK, "baseline")
  assert k._prefill_direct_packed_binding.shape == (512, 1024, 4096)


def test_attachment_validation_failure_is_atomic():
  model = _Model()
  with pytest.raises(ValueError, match="no exact runtime linear"):
    attach_selected_prefill_inventory(model, _inventory([
      _row("q", "blk.0.attn.q.weight"), _row("k", "blk.0.attn.missing.weight")]),
      _policy(), {}, direct_packed_policy=_direct())
  assert not hasattr(model.blk[0].attn.q, "_prefill_route_attachment")
  assert not hasattr(model.blk[0].attn.q, "_prefill_direct_packed_binding")


def test_direct_packed_policy_is_required():
  with pytest.raises(TypeError, match="direct_packed_policy"):
    attach_selected_prefill_inventory(_Model(), _inventory([]), {"routes": {}}, {}, direct_packed_policy=None)
