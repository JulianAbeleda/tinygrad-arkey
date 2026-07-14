#!/usr/bin/env python3
"""Profile-driven policy authority gate for generated prefill routes."""
from __future__ import annotations

import argparse, json
from typing import Any

from tinygrad.llm import route_policy
from extra.qk import route_manifest
from extra.qk.generated_candidates import builtin_registry
from extra.qk.model_profiles import ModelProfile, prefill_role_shapes, profile_by_id


def _role_shapes(profile:ModelProfile) -> tuple[tuple[str,int,int,int], ...]:
  return tuple((row.role, row.M, row.N, row.K) for row in prefill_role_shapes(profile))


def build(profile:ModelProfile, *, target_route_ids:tuple[str,...],
          representative_shapes:tuple[tuple[str,int,int,int], ...]|None=None, scope:str|None=None) -> dict[str,Any]:
  """Require explicit policy selection for every requested profile role/shape."""
  shapes = representative_shapes or _role_shapes(profile)
  supported = set(route_policy._SUPPORTED_QK_ROUTE_IDS)
  route_support = {rid:rid in supported for rid in target_route_ids}
  loaded = route_policy.has_qk_route_policy()
  selected_roles, selected_routes = [], set()
  if loaded:
    for role, _m, n, k in shapes:
      selected = next((rid for rid in target_route_ids
                       if route_policy.qk_route_policy_selected(rid, {"rows":n, "cols":k})), None)
      if selected is not None: selected_routes.add(selected)
      selected_roles.append({"role":role, "rows":n, "cols":k, "selected_route":selected})
  fully_selectable = loaded and len(selected_roles) == len(shapes) and all(x["selected_route"] for x in selected_roles)
  blocked = []
  if not all(route_support.values()):
    blocked.append(f"route policy support is missing target route ids: {[x for x, ok in route_support.items() if not ok]}")
  if not loaded: blocked.append("no QK route policy is loaded")
  if not fully_selectable: blocked.append(f"loaded policy does not select target routes for every {profile.id} prefill role")
  candidate_ids = {c.route_id for c in builtin_registry().all()}
  route_rows = []
  for rid in target_route_ids:
    row = route_manifest.ROUTES[rid]
    route_rows.append({"route_id":rid, "status":row["status"], "selector":row["selector"],
      "env":dict(row.get("env", {})), "rollback":dict(row.get("rollback", {})),
      "shape_guards":list(row.get("shape_guards", ())), "authorities":str(row.get("authority_gate", "")),
      "provenance":route_manifest.route_provenance(rid), "policy_supported":route_support[rid]})
  return {"schema":"prefill_model_authority_gate.v1", "profile":profile.id, "route":"prefill_model_authority",
    "scope":scope or "Generated prefill routes must be explicit policy decisions for every requested profile role/shape.",
    "verdict":"PREFILL_MODEL_AUTHORITY_BLOCKED" if blocked else "PREFILL_MODEL_AUTHORITY_PASS",
    "supported_route_policy_ids":sorted(supported), "route_matrix":route_rows, "target_routes":target_route_ids,
    "representative_shapes":[{"role":r, "m":m, "n":n, "k":k} for r,m,n,k in shapes],
    "policy_evidence":{"candidate_routes_present":[rid for rid in target_route_ids
      if rid in candidate_ids or rid in route_manifest.ROUTES], "policy_loaded":loaded,
      "policy_selected_roles":selected_roles, "policy_selected_routes":sorted(selected_routes),
      "policy_selectable_count":len(selected_routes)},
    "classified_blocker":bool(blocked), "blocker":blocked or None,
    "required_next":["Register every target route with the centralized route policy authority",
      f"Add explicit policy rows for every {profile.id} prefill role/shape, then rerun this gate"]}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--profile", required=True)
  ap.add_argument("--target-route", action="append", required=True)
  args = ap.parse_args()
  out = build(profile_by_id(args.profile), target_route_ids=tuple(args.target_route))
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if not out["classified_blocker"] else 1)


if __name__ == "__main__": main()
