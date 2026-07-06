#!/usr/bin/env python3
"""Gate for the Q4_K/Q8_1 full-role tiled WMMA lowering contract.

This is a structural gate only: it proves bounded full-role role geometry exists as data and that
the selected scheduler-owned WMMA surface is available according to the checked-in artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from extra.qk.q4k_wmma_tile_lowering import (
  SCHEDULER_OWNED_TILE_LOOP_BLOCKER,
  describe_qwen3_14b_q4k_full_role_lowering,
  build_scheduler_owned_tile_loop_contract,
)

SCHEMA = "q4k-wmma-full-role-contract-gate.v1"
SURFACE_ARTIFACT = Path("bench/q4k-wmma-scheduler-surface/latest.json")
LIFECYCLE_ARTIFACT = Path("bench/q4k-wmma-tiled-lifecycle/latest.json")
NO_HAND_ARTIFACT = Path("bench/q4k-wmma-tiled-no-hand-kernel/latest.json")
ROLE_SHAPE_EXEC_ARTIFACT = Path("bench/q4k-wmma-tiled-role-shape-exec/latest.json")


def _load_json(path: Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except FileNotFoundError:
    return {"missing": str(path)}


def build_report() -> dict[str, Any]:
  spec = describe_qwen3_14b_q4k_full_role_lowering(wmma_surface="shaped_wmma_tile")
  surface = _load_json(SURFACE_ARTIFACT)
  lifecycle = _load_json(LIFECYCLE_ARTIFACT)
  no_hand = _load_json(NO_HAND_ARTIFACT)

  surface_ok = surface.get("verdict") == "Q4K_WMMA_SCHEDULER_SURFACE_SHAPED_READY"
  lifecycle_ok = lifecycle.get("verdict") == "Q4K_WMMA_TILED_LIFECYCLE_PASS"
  no_hand_ok = no_hand.get("verdict") == "Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS"
  roles = spec.to_json()["roles"]
  bounded_roles = all(r["bounds"]["bounded_raw_ok"] and r["bounds"]["live_raw_elems"] <= 256 for r in roles)
  scheduler_loop_contract = build_scheduler_owned_tile_loop_contract(spec.roles, route_id=spec.route_id)
  role_shape_exec = _load_json(ROLE_SHAPE_EXEC_ARTIFACT)
  role_shape_loop_validation = {
    "artifact": str(ROLE_SHAPE_EXEC_ARTIFACT),
    "available": role_shape_exec.get("verdict") not in {None, "missing"},
    "verdict": role_shape_exec.get("verdict"),
    "remaining_blocker": role_shape_exec.get("remaining_blocker"),
    "attempted_count": role_shape_exec.get("attempted_count"),
    "executor_verified": role_shape_exec.get("verdict") in {"Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING",
                                                          "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_LIFECYCLE"},
  }

  return {
    "schema": SCHEMA,
    "route_id": spec.route_id,
    "verdict": "Q4K_WMMA_FULL_ROLE_CONTRACT_PASS" if surface_ok and lifecycle_ok and no_hand_ok and bounded_roles
      else "Q4K_WMMA_FULL_ROLE_CONTRACT_BLOCKED",
    "contract": spec.to_json(),
    "evidence": {
      "surface_artifact": str(SURFACE_ARTIFACT),
      "surface_verdict": surface.get("verdict"),
      "surface_ok": surface_ok,
      "selected_surface": surface.get("selected_surface"),
      "lifecycle_artifact": str(LIFECYCLE_ARTIFACT),
      "lifecycle_verdict": lifecycle.get("verdict"),
      "lifecycle_ok": lifecycle_ok,
      "no_hand_artifact": str(NO_HAND_ARTIFACT),
      "no_hand_verdict": no_hand.get("verdict"),
      "no_hand_ok": no_hand_ok,
      "bounded_roles": bounded_roles,
      "role_shape_exec_validation": role_shape_loop_validation,
      "scheduler_owned_tile_loop": scheduler_loop_contract,
    },
    "remaining_blocker": SCHEDULER_OWNED_TILE_LOOP_BLOCKER if scheduler_loop_contract["required"] else None,
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  args = ap.parse_args(argv)
  report = build_report()
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
