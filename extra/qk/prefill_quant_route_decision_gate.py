"""Quant-format-driven residual policy gate for generated prefill routes."""
from __future__ import annotations

from typing import Any

from tinygrad.llm import route_policy
from extra.qk import route_manifest
from extra.qk.generated_candidates import builtin_registry


def build(*, quant_format:str, direct_route_ids:tuple[str,...], candidate_prefix:str) -> dict[str,Any]:
  if not direct_route_ids: raise ValueError("at least one direct route id is required")
  direct_id = direct_route_ids[0]
  direct_row = route_manifest.ROUTES[direct_id]
  manifest_candidates = tuple(sorted(rid for rid in route_manifest.ROUTES
    if rid.startswith(candidate_prefix) and rid not in direct_route_ids))
  registered_candidates = sorted(c.route_id for c in builtin_registry().all()
    if c.route_id.startswith(candidate_prefix) and c.route_id not in direct_route_ids)
  supported = set(route_policy._SUPPORTED_QK_ROUTE_IDS)
  generated_supported = [rid for rid in sorted(set(manifest_candidates) | set(registered_candidates)) if rid in supported]
  loaded = route_policy.has_qk_route_policy()
  direct_selected = loaded and any(route_policy.qk_route_policy_selected(rid) for rid in direct_route_ids)
  blockers = []
  if not generated_supported: blockers.append(f"no generated {quant_format} prefill alternative is registered with route policy")
  if not loaded: blockers.append(f"no QK route policy is loaded for {quant_format} residual attribution")
  return {"schema":"prefill_quant_route_decision_gate.v1", "quant_format":quant_format,
    "verdict":"PREFILL_QUANT_ROUTE_DECISION_BLOCKED" if blockers else "PREFILL_QUANT_ROUTE_DECISION_READY",
    "direct_route":{"route_id":direct_id, "status":direct_row["status"],
      "provenance":route_manifest.route_provenance(direct_id), "env":dict(direct_row.get("env", {})),
      "rollback":dict(direct_row.get("rollback", {})), "roles":list(direct_row.get("roles", ())),
      "route_attribution":str(direct_row.get("route_attribution", ""))},
    "generated_route_inventory":{"manifest_routes":list(manifest_candidates), "registered_candidates":registered_candidates,
      "routes_supported_by_policy":generated_supported, "policy_loaded":loaded, "policy_selects_direct":direct_selected},
    "classified_blocker":bool(blockers), "blocker":" ".join(blockers) or None,
    "required_next":[f"Collect full-prefill attribution for {quant_format} before deciding keep/replace",
      f"If residual exceeds policy budget, register and measure a generated {quant_format} alternative before promotion"]}
