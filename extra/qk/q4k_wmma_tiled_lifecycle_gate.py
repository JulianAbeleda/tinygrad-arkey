#!/usr/bin/env python3
"""Lifecycle gate for the Q4_K/Q8_1 tiled WMMA route."""
from __future__ import annotations

import json
from typing import Any

from extra.qk.q4k_wmma_tiled_microgate import build as microgate_build
from extra.qk.q4k_wmma_tiled_surface_gate import build as surface_build


def build(surface:dict[str, Any]|None=None, microgate:dict[str, Any]|None=None) -> dict[str, Any]:
  surface = surface if surface is not None else surface_build()
  microgate = microgate if microgate is not None else microgate_build()
  surface_ok = surface["verdict"] == "Q4K_WMMA_TILED_SURFACE_TC_MATCHER_SELECTED"
  microgate_ok = microgate["verdict"] == "Q4K_WMMA_TILED_MICROGATE_PASS"
  blocker = "one-kernel multi-output-tile lifecycle lowering is not implemented"
  verdict = "Q4K_WMMA_TILED_LIFECYCLE_BLOCKED_MULTI_TILE_LOWERING" if surface_ok and microgate_ok \
    else "Q4K_WMMA_TILED_LIFECYCLE_FAIL"
  return {"schema": "q4k_wmma_tiled_lifecycle_gate.v1",
          "scope": "M=32,N=32,K=256 four-output-tile generated lifecycle for Q4_K/Q8_1 tiled WMMA",
          "verdict": verdict,
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "target_shape": {"m": 32, "n": 32, "k": 256, "output_tiles": 4,
                           "m_tile": 16, "n_tile": 16, "group_tile": 1},
          "surface": {"ok": surface_ok, "verdict": surface["verdict"],
                      "selected_surface": surface.get("selected_surface")},
          "one_tile_numeric": {"ok": microgate_ok, "verdict": microgate["verdict"],
                               "has_iu8_wmma": microgate["probe"]["has_iu8_wmma"]},
          "implemented": False,
          "class": "blocked.multi_tile_lifecycle_missing" if verdict.endswith("MULTI_TILE_LOWERING") else "fail",
          "classified_blocker": verdict == "Q4K_WMMA_TILED_LIFECYCLE_BLOCKED_MULTI_TILE_LOWERING",
          "blocker": blocker,
          "required_next": ["lower four 16x16 output tiles from one generated route lifecycle",
                            "keep raw_i32 bounded at m_tile*n_tile*group_tile",
                            "report kernel_count/compile_ms/runtime_ms and prove iu8 WMMA remains present",
                            "do not concatenate one-tile Tensor wrappers as the final lifecycle"]}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_LIFECYCLE_PASS" else 1)
