#!/usr/bin/env python3
"""Residual-policy gate for 14B Q6_K prefill route selection."""
from __future__ import annotations

import json
from typing import Any

from tinygrad.llm import route_policy
from extra.qk.generated_candidates import builtin_registry

from extra.qk import route_manifest

Q6_ROUTE_IDS = ("prefill_q6k_direct_generated",)
MMQ_Q6_ROUTE_PREFIX = "prefill_q6k"


def _direct_route_row() -> dict[str, Any]:
  return route_manifest.ROUTES["prefill_q6k_direct_generated"]


def _candidate_routes() -> list[str]:
  return sorted(c.route_id for c in builtin_registry().all() if c.route_id.startswith(MMQ_Q6_ROUTE_PREFIX)
                and c.route_id not in Q6_ROUTE_IDS)


def _manifest_q6_routes() -> tuple[str, ...]:
  return tuple(sorted(rid for rid in route_manifest.ROUTES if rid.startswith(MMQ_Q6_ROUTE_PREFIX)
                   and rid not in Q6_ROUTE_IDS))


def _policy_support() -> dict[str, Any]:
  supported = set(route_policy._SUPPORTED_QK_ROUTE_IDS)
  manifest_mmq = _manifest_q6_routes()
  candidate_mmq = _candidate_routes()
  return {
    "supported_ids": sorted(supported),
    "manifest_q6_mmq_routes": list(manifest_mmq),
    "candidate_q6_mmq_routes": candidate_mmq,
    "supported_q6_mmq": [rid for rid in sorted(set(manifest_mmq) | set(candidate_mmq)) if rid in supported],
    "policy_loaded": route_policy.has_qk_route_policy(),
  }


def _direct_route_policy_load() -> dict[str, Any]:
  if not route_policy.has_qk_route_policy():
    return {
      "loaded": False,
      "selected_direct": False,
    }

  direct_selected = any(route_policy.qk_route_policy_selected(rid) for rid in Q6_ROUTE_IDS)
  return {
    "loaded": True,
    "selected_direct": direct_selected,
  }


def build() -> dict[str, Any]:
  policy_state = _policy_support()
  direct_row = _direct_route_row()
  direct_candidates = _candidate_routes()
  direct_policy = _direct_route_policy_load()

  no_generated_q6_mmq = not policy_state["supported_q6_mmq"]
  blocker = None
  if no_generated_q6_mmq:
    blocker = "No generated Q6_K prefill MMQ route is currently registered in route-policy support or generated candidates; " \
              "policy cannot switch to generated MMQ for Q6_K residual today."

  if not direct_policy["loaded"]:
    if blocker is None:
      blocker = "No QK route policy is loaded for 14B Q6_K attribution; residual decision is intentionally unclassified."
    else:
      blocker += " No QK route policy is loaded for 14B Q6_K attribution."

  verdict = "PREFILL_14B_Q6_DECISION_BLOCKED_NO_GENERATED_Q6_MMQ"
  if blocker is None:
    verdict = "PREFILL_14B_Q6_DECISION_READY_KEEP_DIRECT_WITH_EVIDENCE"

  return {
    "schema": "prefill_14b_q6_decision_gate.v1",
    "route": "prefill_14b_q6_decision",
    "scope": "14B residual policy for Q6_K prefill is explicit only after generated candidate and residual attribution exist.",
    "verdict": verdict,
    "direct_route": {
      "route_id": "prefill_q6k_direct_generated",
      "status": direct_row["status"],
      "provenance": route_manifest.route_provenance("prefill_q6k_direct_generated"),
      "env": dict(direct_row.get("env", {})),
      "rollback": dict(direct_row.get("rollback", {})),
      "roles": list(direct_row.get("roles", ())),
      "route_attribution": str(direct_row.get("route_attribution", "")),
    },
    "generated_route_inventory": {
      "manifest_q6_routes": policy_state["manifest_q6_mmq_routes"],
      "registered_candidates": direct_candidates,
      "q6_mmq_routes_supported_by_policy": policy_state["supported_q6_mmq"],
      "policy_loaded": direct_policy["loaded"],
      "policy_selects_direct": direct_policy["selected_direct"],
    },
    "classified_blocker": blocker is not None,
    "blocker": blocker,
    "required_next": [
      "Collect 14B full-prefill attribution including Q6_K wall-share before deciding keep/expand",
      "If residual exceeds policy budget, add a generated Q6_K MMQ prefill route and policy selector before promotion",
    ],
  }


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if not out["classified_blocker"] else 1)
