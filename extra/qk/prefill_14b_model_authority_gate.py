#!/usr/bin/env python3
"""Policy authority gate for 14B Q4_K/Q8_1 prefill routes."""
from __future__ import annotations

import argparse
import json
from typing import Any

from tinygrad.llm import route_policy
from extra.qk.generated_candidates import builtin_registry

from extra.qk import route_manifest
from extra.qk.model_profiles import qwen3_14b_q4k_m_gfx1100_profile
from extra.qk.q4k_wmma_tile_lowering import q4k_prefill_role_shape_tuples

DEFAULT_TARGET_ROUTE_IDS = (
  "prefill_q4k_int8_wmma_generated_research",
  "prefill_q4k_int8_wmma_tiled_research",
)
HYBRID_MMQ_ATOM_ROUTE_ID = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"


def _representative_q4k_shapes() -> tuple[tuple[str, int, int, int], ...]:
  return q4k_prefill_role_shape_tuples(qwen3_14b_q4k_m_gfx1100_profile())


def _route_manifest_row(route_id: str) -> dict[str, Any]:
  return route_manifest.ROUTES[route_id]


def _policy_support(target_route_ids: tuple[str, ...]) -> dict[str, Any]:
  supported = set(route_policy._SUPPORTED_QK_ROUTE_IDS)
  loaded = route_policy.has_qk_route_policy()
  route_support = {rid: rid in supported for rid in target_route_ids}
  return {
    "supported_ids": sorted(supported),
    "route_support": route_support,
    "supported_target_route_count": sum(route_support.values()),
    "policy_loaded": loaded,
  }


def _policy_selection_evidence(target_route_ids: tuple[str, ...],
                               representative_shapes: tuple[tuple[str, int, int, int], ...]) -> dict[str, Any]:
  if not route_policy.has_qk_route_policy():
    return {
      "loaded": False,
      "selected_roles": [],
      "fully_selectable": False,
      "routes_selected_any": [],
    }

  selected_roles = []
  routes_selected_any: set[str] = set()
  for role, _m, n, k in representative_shapes:
    selected_route: str | None = None
    for route_id in target_route_ids:
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
    "fully_selectable": len(selected_roles) == len(representative_shapes) and all(
      row["selected_route"] is not None for row in selected_roles),
  }


def _candidate_routes(target_route_ids: tuple[str, ...]) -> list[str]:
  candidate_ids = [c.route_id for c in builtin_registry().all()]
  generated = [rid for rid in candidate_ids if rid in target_route_ids]
  manifest_only = [rid for rid in target_route_ids if rid in route_manifest.ROUTES and rid not in generated]
  return generated + manifest_only


def build(*, target_route_ids: tuple[str, ...] = DEFAULT_TARGET_ROUTE_IDS,
          representative_shapes: tuple[tuple[str, int, int, int], ...] | None = None,
          scope: str | None = None) -> dict[str, Any]:
  representative_shapes = representative_shapes or _representative_q4k_shapes()
  policy_state = _policy_support(target_route_ids)
  selection = _policy_selection_evidence(target_route_ids, representative_shapes)
  route_rows = []
  for rid in target_route_ids:
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
    missing = [rid for rid, ok in policy_state["route_support"].items() if not ok]
    blocked.append(f"QK_ROUTE_POLICY._SUPPORTED_QK_ROUTE_IDS is missing target prefill route ids: {missing}")
  if not selection["loaded"]:
    blocked.append("No QK route policy is currently loaded; no policy decision evidence available for target routes")
  if not selection.get("fully_selectable"):
    blocked.append("Loaded policy does not select target routes across all representative 14B Q4_K/Q8_1 packed rows.")

  verdict = "PREFILL_14B_MODEL_AUTHORITY_BLOCKED"
  if not blocked:
    verdict = "PREFILL_14B_MODEL_AUTHORITY_PASS"

  return {
    "schema": "prefill_14b_model_authority_gate.v1",
    "route": "prefill_14b_model_authority",
    "scope": scope or "14B Q4_K/Q8_1 packed route should be explicit via QK route policy and represent requested packed roles.",
    "verdict": verdict,
    "supported_route_policy_ids": policy_state["supported_ids"],
    "route_matrix": route_rows,
    "target_routes": target_route_ids,
    "representative_q4k_shapes": [
      {"role": role, "m": m, "n": n, "k": k} for role, m, n, k in representative_shapes
    ],
    "policy_evidence": {
      "candidate_routes_present": _candidate_routes(target_route_ids),
      "policy_loaded": selection["loaded"],
      "policy_selected_roles": selection["selected_roles"],
      "policy_selected_routes": selection["routes_selected_any"],
      "policy_selectable_count": len(selection["routes_selected_any"]),
    },
    "classified_blocker": bool(blocked),
    "blocker": None if not blocked else blocked,
    "required_next": [
      "Add target prefill route ids to tinygrad/llm/route_policy._SUPPORTED_QK_ROUTE_IDS and expose per-shape rows",
      "Add explicit QK_ROUTE_POLICY rows for representative 14B Q4_K/Q8_1 prefill shapes, then rerun this gate",
    ],
  }


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--target-route", action="append", default=[],
                  help="target route id; repeatable. Defaults to historical Q4K/Q8_1 WMMA research routes")
  ap.add_argument("--hybrid-atom-ffn-gate-up", action="store_true",
                  help="audit the 14B hybrid MMQ atom scaffold for ffn_gate_up only")
  return ap.parse_args()


if __name__ == "__main__":
  args = _parse_args()
  if args.hybrid_atom_ffn_gate_up:
    prof = qwen3_14b_q4k_m_gfx1100_profile()
    row = prof.role_shape("ffn_gate_up")
    out = build(target_route_ids=(HYBRID_MMQ_ATOM_ROUTE_ID,),
                representative_shapes=((row.role, row.M, row.N, row.K),),
                scope="14B hybrid MMQ atom route should be explicit via QK route policy for ffn_gate_up only.")
  else:
    out = build(target_route_ids=tuple(args.target_route) or DEFAULT_TARGET_ROUTE_IDS)
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if not out["classified_blocker"] else 1)
