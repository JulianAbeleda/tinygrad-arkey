#!/usr/bin/env python3
"""Full-role execution classifier for the Q4_K/Q8_1 tiled WMMA route."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extra.qk.q4k_wmma_tile_lowering import QWEN3_14B_Q4K_ROLE_SHAPES, describe_int8_wmma_tile_lowering
from extra.qk.q4k_wmma_tiled_lifecycle_gate import build as lifecycle_build

ARTIFACT = Path("bench/q4k-wmma-tiled-role-shape-exec/latest.json")

def _role_row(role:str, m:int, n:int, k:int) -> dict[str, Any]:
  spec = describe_int8_wmma_tile_lowering(m, n, k, role=role, m_tile=16, n_tile=16, group_tile=1)
  plan = spec.to_json()
  return {"role": role, "m": m, "n": n, "k": k, "groups": spec.groups,
          "tile": {"m_tile": spec.m_tile, "n_tile": spec.n_tile, "group_tile": spec.group_tile,
                   "live_raw_elems": spec.live_raw_elems,
                   "forbidden_full_raw_elems": spec.forbidden_full_raw_elems},
          "lowering_plan": plan,
          "exec": {"attempted": False, "class": "blocked.lifecycle_missing",
                   "compile_ms": None, "runtime_ms": None, "kernel_count": None,
                   "graph_node_count": None, "wmma_present": None}}


def build(lifecycle:dict[str, Any]|None=None) -> dict[str, Any]:
  lifecycle = lifecycle if lifecycle is not None else lifecycle_build()
  lifecycle_pass = lifecycle["verdict"] == "Q4K_WMMA_TILED_LIFECYCLE_PASS"
  rows = [_role_row(*shape) for shape in QWEN3_14B_Q4K_ROLE_SHAPES]
  for row in rows:
    row["exec"]["class"] = "blocked.scheduler_owned_tile_loop_missing" if lifecycle_pass else "blocked.lifecycle_missing"
  verdict = "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING" if lifecycle_pass else \
    "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_LIFECYCLE"
  return {"schema": "q4k_wmma_tiled_role_shape_exec_gate.v1",
          "scope": "synthetic execution gate for all 14B Q4_K/Q8_1 wmma_tiled prefill role shapes",
          "verdict": verdict,
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "lifecycle": {"verdict": lifecycle["verdict"], "class": lifecycle["class"]},
          "roles": rows,
          "classified_blocker": True,
          "blocker": "role execution is intentionally not attempted until a scheduler-owned tile_m/tile_n/group loop exists",
          "distinction_from_classifier": "q4k_wmma_tiled_role_shape enumerates/selects shapes; this gate is reserved for actual synthetic execution metrics"}


if __name__ == "__main__":
  out = build()
  ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
  ARTIFACT.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_PASS" else 1)
