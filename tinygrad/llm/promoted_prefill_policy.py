"""Production selection policy for the retained searched fp16 prefill graph."""
from __future__ import annotations

from typing import Mapping

from tinygrad.llm.prefill_candidate_runtime import (ROUTE_ID, canonical_candidate_set_identity,
                                                     promoted_candidate_registry)


def automatic_promoted_prefill_graph_policy(inventory: Mapping, scanned_device_facts: Mapping) -> dict | None:
  """Bind the promoted candidate set to one exact selected-GGUF inventory.

  This is structural admission only.  The model's existing memory planner must
  still admit the complete fp16 overlay before installing this policy.
  """
  rows = inventory.get("rows") if isinstance(inventory, Mapping) else None
  capabilities = scanned_device_facts.get("capabilities") if isinstance(scanned_device_facts, Mapping) else None
  if not isinstance(rows, list) or not isinstance(capabilities, Mapping): return None
  target = {"backend":scanned_device_facts.get("backend"), "arch":scanned_device_facts.get("architecture"),
            "wave_size":capabilities.get("wave_size")}
  inventory_identity = inventory.get("inventory_identity")
  if not isinstance(inventory_identity, str) or not inventory_identity: return None

  try: registry = promoted_candidate_registry()
  except (KeyError, TypeError, ValueError, OSError): return None
  candidate_set = registry.candidate_set.to_json()
  candidate_set_identity = canonical_candidate_set_identity(candidate_set)
  candidate_targets = {(admission.normalized_payload["workload"]["target"]["backend"],
                        admission.normalized_payload["workload"]["target"]["arch"],
                        admission.normalized_payload["workload"]["target"]["wave_size"])
                       for admission in registry.admissions}
  if (target["backend"], target["arch"], target["wave_size"]) not in candidate_targets: return None

  controlled, routes, bindings = [], {}, {}
  for row in rows:
    if not isinstance(row, Mapping): return None
    invocation_id = row.get("invocation_id")
    if not isinstance(invocation_id, str) or not invocation_id or invocation_id in routes: return None
    if row.get("candidate_controlled") is True:
      role, shape = row.get("role"), row.get("shape")
      if not isinstance(role, str) or not isinstance(shape, Mapping): return None
      try: exact_shape = tuple(shape[axis] for axis in ("m", "n", "k"))
      except KeyError: return None
      if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in exact_shape): return None
      admission = registry.get(role, exact_shape, target)
      if admission is None: return None
      key = (role, *exact_shape)
      prior = bindings.get(key)
      if prior is not None and prior.canonical_identity != admission.canonical_identity: return None
      bindings[key] = admission
      controlled.append(row)
      routes[invocation_id] = ROUTE_ID
    else:
      fixed_route_id = row.get("fixed_route_id")
      if not isinstance(fixed_route_id, str) or not fixed_route_id: return None
      routes[invocation_id] = fixed_route_id
  if not controlled: return None
  registry_keys = {(admission.normalized_payload["workload"]["role"],
                    *(admission.normalized_payload["workload"]["shape"][axis] for axis in ("m", "n", "k")))
                   for admission in registry.admissions}
  if set(bindings) != registry_keys: return None

  policy_rows = [{"phase":"prefill", "role":role, "quant":"fp16",
                  "shape":dict(zip(("m", "n", "k"), (m, n, k))), "target":dict(target),
                  "inventory_identity":inventory_identity, "candidate_set_identity":candidate_set_identity,
                  "candidate_identity":bindings[(role, m, n, k)].canonical_identity,
                  "selected_route":ROUTE_ID, "route_aliases":[ROUTE_ID]}
                 for role,m,n,k in sorted(bindings)]
  return {"strategy":"FULL_RESIDENT_OVERLAY", "candidate_id":ROUTE_ID, "routes":routes,
          "inventory_identity":inventory_identity,
          "provenance":"retained promoted candidate + live exact-fact admission", "measured":True,
          "selection_authority":"promoted_exact_candidate",
          "graph_gemm":{"candidate_set":candidate_set, "candidate_set_identity":candidate_set_identity,
                        "policy_rows":policy_rows, "compile_artifacts":{}}}


__all__ = ["automatic_promoted_prefill_graph_policy"]
