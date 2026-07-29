"""Immutable prefill attachment values and load-time inventory installation."""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad.llm.route_selection import RouteLifecycle


@dataclass(frozen=True)
class PrefillRouteAttachment:
  invocation_id: str
  route_id: str
  tensor_identity: str
  selected_policy: object
  scanned_target_facts: object
  allocation_owner_identity: str | None = None


@dataclass(frozen=True)
class PrefillDirectPackedBinding:
  """Exact selected-inventory ownership for the direct-packed prefill helper."""
  invocation_id: str
  phase: str
  role: str
  shape: tuple[int, int, int]
  lifecycle: RouteLifecycle = RouteLifecycle.PROMOTED
  quarantine_reason: str | None = None


def _runtime_linear(model, tensor_identity: str):
  if not isinstance(tensor_identity, str) or not tensor_identity.endswith(".weight"):
    raise ValueError(f"selected prefill tensor identity is not an exact runtime weight path: {tensor_identity!r}")
  obj = model
  try:
    for component in tensor_identity[:-7].split("."):
      obj = obj[int(component)] if component.isdigit() and isinstance(obj, (list, tuple)) else getattr(obj, component)
  except (AttributeError, IndexError, TypeError) as exc:
    raise ValueError(f"selected prefill tensor {tensor_identity!r} has no exact runtime linear") from exc
  return obj


def attach_selected_prefill_inventory(model, inventory:dict, policy, scanned_target_facts, *, direct_packed_policy) -> None:
  """Attach a caller-selected inventory atomically to its exact runtime linears.

  Policy selection is deliberately outside this module: ``direct_packed_policy``
  is required so attachment installation has no executor or route-policy edge.
  """
  if direct_packed_policy is None:
    raise TypeError("direct_packed_policy must be supplied")
  routes = dict(policy.get("routes", {})) if policy is not None else {}
  rows = inventory.get("rows", ())
  ids = [row.get("invocation_id") for row in rows]
  if len(ids) != len(set(ids)): raise ValueError("duplicate selected prefill inventory rows")
  if set(routes) != set(ids): raise ValueError("selected policy and inventory attachments differ")

  # Validate and construct every attachment before changing any runtime object.
  pending = []
  for row in rows:
    tensor_identity = row.get("tensor_identity")
    obj = _runtime_linear(model, tensor_identity)
    if hasattr(obj, "_prefill_route_attachment"): raise ValueError(f"duplicate runtime attachment for {tensor_identity!r}")
    invocation_id = row["invocation_id"]
    shape = row.get("shape")
    if not isinstance(shape, dict): raise ValueError(f"selected prefill row {tensor_identity!r} has no concrete shape")
    try: direct_shape = tuple(shape[axis] for axis in ("m", "n", "k"))
    except KeyError as exc: raise ValueError(f"selected prefill row {tensor_identity!r} has incomplete shape") from exc
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in direct_shape):
      raise ValueError(f"selected prefill row {tensor_identity!r} has invalid concrete shape")
    proof = policy.get("bounded_packed_projection_proof", {}) if hasattr(policy, "get") else {}
    owner_identity = proof.get("allocation_owner_identity") if hasattr(proof, "get") else None
    pending.append((obj, PrefillRouteAttachment(invocation_id, routes[invocation_id], tensor_identity, policy,
                                                 scanned_target_facts, owner_identity),
                    PrefillDirectPackedBinding(invocation_id, "prefill", str(row.get("role", "")), direct_shape,
                                               direct_packed_policy.lifecycle, direct_packed_policy.reason)))

  for obj, attachment, binding in pending:
    setattr(obj, "_prefill_route_attachment", attachment)
    setattr(obj, "_prefill_direct_packed_binding", binding)


__all__ = ["PrefillRouteAttachment", "PrefillDirectPackedBinding", "attach_selected_prefill_inventory"]
