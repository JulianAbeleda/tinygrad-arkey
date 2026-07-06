#!/usr/bin/env python3
"""WMMA surface-selection gate for the Q4_K/Q8_1 tiled prefill route."""
from __future__ import annotations

import json, pathlib
from typing import Any

from extra.qk.q4k_wmma_tiled_lowering_feasibility import _run_probe as _run_tc_matcher_probe
from extra.qk.q4k_wmma_tiled_no_hand_kernel_gate import build as no_hand_build

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _has(path:str, needle:str) -> bool:
  p = ROOT / path
  return p.exists() and needle in p.read_text()


def _route_local_shaped_wmma_producers() -> list[str]:
  producers: list[str] = []
  for path in ("extra/qk/prefill_int8_wmma_spec.py", "tinygrad/llm/prefill_routes.py",
               "extra/qk/q4k_wmma_tile_lowering.py"):
    p = ROOT / path
    if p.exists() and "Ops.SHAPED_WMMA" in p.read_text():
      producers.append(path)
  return producers


def _tc_matcher_surface() -> dict[str, Any]:
  probe = _run_tc_matcher_probe()
  ok = probe["returncode"] == 0 and probe["has_iu8_wmma"] and probe["max_abs_ok"]
  return {"name": "tc_matcher_tile",
          "owner": "tinygrad Tensor.matmul(dtype=int) -> tinygrad/codegen/opt/tc.py -> HIPRenderer WMMA",
          "class": "selected" if ok else "blocked.probe_failed",
          "reason": "Existing tensor-core matcher lowers the bounded int8 tile to iu8 WMMA." if ok
                    else "Bounded int8 tile did not prove iu8 WMMA lowering.",
          "requires_env": {"DEV": "AMD", "TC": "1", "TC_OPT": "1"},
          "full_role_status": "needs lifecycle/role-shape execution gate; one tile is proven, full loop ownership is not",
          "probe": probe}


def _shaped_wmma_surface() -> dict[str, Any]:
  lowerer_present = _has("tinygrad/schedule/rangeify.py", "lower_shaped_wmma")
  spec_present = _has("tinygrad/uop/spec.py", "Ops.SHAPED_WMMA")
  producers = _route_local_shaped_wmma_producers()
  ok = lowerer_present and spec_present and bool(producers)
  return {"name": "shaped_wmma_tile",
          "owner": "tinygrad/schedule/rangeify.py lower_shaped_wmma",
          "class": "available" if ok else "blocked.no_q4k_producer",
          "reason": "Route-local reusable Q4_K SHAPED_WMMA producer exists." if ok
                    else "Infrastructure exists, but no reusable Q4_K route producer constructs Ops.SHAPED_WMMA.",
          "infrastructure": {"rangeify_lowerer": lowerer_present, "uop_spec": spec_present},
          "route_local_producers": producers}


def build() -> dict[str, Any]:
  surfaces = [_tc_matcher_surface(), _shaped_wmma_surface()]
  no_hand = no_hand_build()
  selected = next((s for s in surfaces if s["class"] == "selected"), None)
  no_hand_ok = no_hand["verdict"] == "Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS"
  verdict = "Q4K_WMMA_TILED_SURFACE_TC_MATCHER_SELECTED" if selected and selected["name"] == "tc_matcher_tile" and no_hand_ok \
    else "Q4K_WMMA_TILED_SURFACE_BLOCKED"
  return {"schema": "q4k_wmma_tiled_surface_gate.v1",
          "scope": "select the generated WMMA surface for bounded Q4_K/Q8_1 tiled prefill",
          "verdict": verdict,
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "surface": selected["name"] if selected else None,
          "has_ops_shaped_wmma": any(s["name"] == "shaped_wmma_tile" and s["infrastructure"]["uop_spec"]
                                    for s in surfaces),
          "has_ops_wmma_after_rangeify": any(s["name"] == "shaped_wmma_tile" and s["infrastructure"]["rangeify_lowerer"]
                                             for s in surfaces),
          "has_iu8_wmma_isa_or_source": any(s["name"] == "tc_matcher_tile" and s["probe"]["has_iu8_wmma"]
                                            for s in surfaces),
          "numeric_ok": any(s["name"] == "tc_matcher_tile" and s["probe"]["max_abs_ok"] for s in surfaces),
          "no_route_local_builtin": no_hand_ok and not any(f["kind"] == "source.route_local_wmma_builtin"
                                                           for f in no_hand["findings"]),
          "no_route_local_asm": no_hand_ok and not any(f["kind"] == "source.inline_asm"
                                                       for f in no_hand["findings"]),
          "live_raw_elems": 16 * 16 * 1,
          "no_hand_kernel": no_hand,
          "selected_surface": selected["name"] if selected else None,
          "surfaces": surfaces,
          "next_required": "build the lifecycle/full-role execution lowering on the selected TC matcher surface without route-local WMMA source or fallback"}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_SURFACE_TC_MATCHER_SELECTED" else 1)
