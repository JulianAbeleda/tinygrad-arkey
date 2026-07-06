#!/usr/bin/env python3
"""Distinct full-role execution gate for the Q4_K/Q8_1 tiled WMMA route."""
from __future__ import annotations

import json
from typing import Any

from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill
from extra.qk.q4k_wmma_tiled_lifecycle_gate import build as lifecycle_build

ROLE_SHAPES = (
  ("attn_kv", 512, 1024, 5120),
  ("attn_qo", 512, 5120, 5120),
  ("ffn_down", 512, 5120, 17408),
  ("ffn_gate_up", 512, 17408, 5120),
)


def _role_row(role:str, m:int, n:int, k:int) -> dict[str, Any]:
  spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role=role, m_tile=16, n_tile=16, group_tile=1)
  return {"role": role, "m": m, "n": n, "k": k, "groups": spec.groups,
          "tile": {"m_tile": spec.m_tile, "n_tile": spec.n_tile, "group_tile": spec.group_tile,
                   "live_raw_elems": spec.live_raw_elems,
                   "forbidden_full_raw_elems": spec.forbidden_full_raw_elems},
          "exec": {"attempted": False, "class": "blocked.lifecycle_missing",
                   "compile_ms": None, "runtime_ms": None, "kernel_count": None,
                   "graph_node_count": None, "wmma_present": None}}


def build(lifecycle:dict[str, Any]|None=None) -> dict[str, Any]:
  lifecycle = lifecycle if lifecycle is not None else lifecycle_build()
  lifecycle_blocked = lifecycle["verdict"] == "Q4K_WMMA_TILED_LIFECYCLE_BLOCKED_MULTI_TILE_LOWERING"
  lifecycle_pass = lifecycle["verdict"] == "Q4K_WMMA_TILED_LIFECYCLE_PASS"
  rows = [_role_row(*shape) for shape in ROLE_SHAPES]
  for row in rows:
    row["exec"]["class"] = "blocked.full_role_lowering_missing" if lifecycle_pass else "blocked.lifecycle_missing"
  verdict = "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING" if lifecycle_pass else \
    "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_LIFECYCLE" if lifecycle_blocked else "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_FAIL"
  return {"schema": "q4k_wmma_tiled_role_shape_exec_gate.v1",
          "scope": "synthetic execution gate for all 14B Q4_K/Q8_1 wmma_tiled prefill role shapes",
          "verdict": verdict,
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "lifecycle": {"verdict": lifecycle["verdict"], "class": lifecycle["class"]},
          "roles": rows,
          "classified_blocker": verdict in ("Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_LIFECYCLE",
                                            "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING"),
          "blocker": "role execution is intentionally not attempted until full-role synthetic lowering exists",
          "distinction_from_classifier": "q4k_wmma_tiled_role_shape enumerates/selects shapes; this gate is reserved for actual synthetic execution metrics"}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_PASS" else 1)
