#!/usr/bin/env python3
"""Compatibility entry point for the generic Q6_K prefill route decision gate."""
from __future__ import annotations

import json
from typing import Any

from extra.qk.prefill_quant_route_decision_gate import build as build_quant_route_decision

Q6_ROUTE_IDS = ("prefill_q6k_direct_generated",)
MMQ_Q6_ROUTE_PREFIX = "prefill_q6k"


def build() -> dict[str,Any]:
  out = build_quant_route_decision(quant_format="Q6_K", direct_route_ids=Q6_ROUTE_IDS,
                                   candidate_prefix=MMQ_Q6_ROUTE_PREFIX)
  inventory = out["generated_route_inventory"]
  inventory["manifest_q6_routes"] = inventory.pop("manifest_routes")
  inventory["q6_mmq_routes_supported_by_policy"] = inventory.pop("routes_supported_by_policy")
  if not inventory["q6_mmq_routes_supported_by_policy"]:
    suffix = " No QK route policy is loaded for 14B Q6_K attribution." if not inventory["policy_loaded"] else ""
    out["blocker"] = "No generated Q6_K prefill MMQ route is currently registered in route-policy support or generated candidates; " \
      f"policy cannot switch to generated MMQ for Q6_K residual today.{suffix}"
  out.update(schema="prefill_14b_q6_decision_gate.v1", route="prefill_14b_q6_decision",
    scope="14B residual policy for Q6_K prefill requires generated-candidate and full-prefill attribution evidence.",
    verdict=("PREFILL_14B_Q6_DECISION_BLOCKED_NO_GENERATED_Q6_MMQ" if out["classified_blocker"]
             else "PREFILL_14B_Q6_DECISION_READY_KEEP_DIRECT_WITH_EVIDENCE"))
  return out


if __name__ == "__main__":
  report = build()
  print(json.dumps(report, indent=2))
  raise SystemExit(0 if not report["classified_blocker"] else 1)
