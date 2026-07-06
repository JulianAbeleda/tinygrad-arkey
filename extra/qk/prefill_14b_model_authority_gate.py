#!/usr/bin/env python3
"""Policy authority gate for 14B Q4_K/Q8_1 MMQ prefill."""
from __future__ import annotations

import json
from typing import Any

from tinygrad.llm import route_policy
from tinygrad.llm.generated_candidates import builtin_registry

from extra.qk import route_manifest
from extra.qk.q4k_wmma_tile_lowering import QWEN3_14B_Q4K_ROLE_SHAPES

TARGET_ROUTE_IDS = (
  "prefill_q4k_int8_wmma_generated_research",
  "prefill_q4k_int8_wmma_tiled_research",
)


def _route_manifest_row(route_id: str) -> dict[str, Any]:
  return route_manifest.ROUTES[route_id]


def _policy_support() -> dict[str, Any]:
  supported = set(route_policy._SUPPORTED_QK_ROUTE_IDS)
  loaded = route_policy.has_qk_route_policy()
  route_support = {rid: rid in supported for rid in TARGET_ROUTE_IDS}
  return {
    "supported_ids": sorted(supported),
    "route_support": route_support,
    "supported_target_route_count": sum(route_support.values()),
    "policy_loaded": loaded,
  }


def _policy_selection_evidence() -> dict[str, Any]:
  if not route_policy.has_qk_route_policy():
    return {
      "loaded": False,
      "selected_roles": [],
      "fully_selectable": False,
      "routes_selected_any": [],
    }

  selected_roles = []
  routes_selected_any: set[str] = set()
  for role, _m, n, k in QWEN3_14B_Q4K_ROLE_SHAPES:
    selected_route: str | None = None
    for route_id in TARGET_ROUTE_IDS:
      if route_policy.qk_route_policy_selected(route_id, {"rows": n, "cols": k}):
        selected_route = route_id
        break
    if selected_route is not None:
      routes_selected_any.add(selected_route)
    selected_roles.append({
      "role": role,
      "rows": n,
      "cols": k,
      "selected_route": selected_route,
    })

  return {
    "loaded": True,
    "selected_roles": selected_roles,
    "routes_selected_any": sorted(routes_selected_any),
    "fully_selectable": len(selected_roles) == len(QWEN3_14B_Q4K_ROLE_SHAPES) and all(
      row["selected_route"] is not None for row in selected_roles),
  }


def _candidate_routes() -> list[str]:
  candidate_ids = [c.route_id for c in builtin_registry().all()]
  return [rid for rid in candidate_ids if rid in TARGET_ROUTE_IDS]


def build() -> dict[str, Any]:
  policy_state = _policy_support()
  selection = _policy_selection_evidence()
  route_rows = []
  for rid in TARGET_ROUTE_IDS:
    row = _route_manifest_row(rid)
    route_rows.append({
      "route_id": rid,
      "status": row["status"],
      "selector": row["selector"],
      "env": dict(row.get("env", {})),
      "rollback": dict(row.get("rollback", {})),
      "shape_guards": list(row.get("shape_guards", ())),
      "authorities": str(row.get("authority_gate", "")),
      "provenance": route_manifest.route_provenance(rid),
      "policy_supported": policy_state["route_support"][rid],
    })

  can_select = all(policy_state["route_support"].values())
  blocked = []
  if not can_select:
    blocked.append("QK_ROUTE_POLICY._SUPPORTED_QK_ROUTE_IDS is missing the MMQ prefill route ids: "
                  "'prefill_q4k_int8_wmma_generated_research' and/or "
                  "'prefill_q4k_int8_wmma_tiled_research'.")
  if not selection["loaded"]:
    blocked.append("No QK route policy is currently loaded; current prefill MMQ entry points remain env-driven via"
                  " PREFILL_Q4K_Q8=wmma/wmma_tiled, no policy decision evidence available")
  if not selection.get("fully_selectable"):
    blocked.append("Loaded policy does not select 14B MMQ/MMV prefill shapes across all representative Q4_K/Q8_1 packed rows.")

  verdict = "PREFILL_14B_MODEL_AUTHORITY_BLOCKED"
  if not blocked:
    verdict = "PREFILL_14B_MODEL_AUTHORITY_PASS"

  return {
    "schema": "prefill_14b_model_authority_gate.v1",
    "route": "prefill_14b_model_authority",
    "scope": "14B Q4_K/Q8_1 packed/MMQ route should be explicit via QK route policy and represent representative packed roles.",
    "verdict": verdict,
    "supported_route_policy_ids": policy_state["supported_ids"],
    "route_matrix": route_rows,
    "target_routes": TARGET_ROUTE_IDS,
    "representative_q4k_shapes": [
      {"role": role, "m": m, "n": n, "k": k} for role, m, n, k in QWEN3_14B_Q4K_ROLE_SHAPES
    ],
    "policy_evidence": {
      "candidate_routes_present": _candidate_routes(),
      "policy_loaded": selection["loaded"],
      "policy_selected_roles": selection["selected_roles"],
      "policy_selected_routes": selection["routes_selected_any"],
      "policy_selectable_count": len(selection["routes_selected_any"]),
    },
    "classified_blocker": bool(blocked),
    "blocker": None if not blocked else blocked,
    "required_next": [
      "Add q4k MMQ prefill route ids to tinygrad/llm/route_policy._SUPPORTED_QK_ROUTE_IDS and expose per-shape rows",
      "Add explicit QK_ROUTE_POLICY rows for representative 14B Q4_K/Q8_1 prefill (wmma/wmma_tiled variants), then rerun this gate",
    ],
  }


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if not out["classified_blocker"] else 1)
