#!/usr/bin/env python3
"""P6 lowering-bind decision for decode online-state+PV tile."""
from __future__ import annotations

import json, pathlib, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
P5 = ROOT / "bench/qk-decode-attention-online-state-pv-tile/latest.json"
MANIFEST = ROOT / "bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json"
OUT = ROOT / "bench/qk-decode-attention-online-state-pv-p6-lowering-bind"


def _exists(rel: str) -> bool:
  return (ROOT / rel).exists()


def _load(path: pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def build() -> dict[str, Any]:
  p5, manifest = _load(P5), _load(MANIFEST)
  sig = p5["online_state_pv_tile"]["signature"]
  programs = sig["generated_attention_programs"]
  p5_clean = p5.get("verdict") == "ONLINE_STATE_PV_TILE_STRUCTURAL_ROUTE_CLEAN"
  lowerings = {
    "fdot2_lowering": _exists("extra/qk_fdot2_lowering.py"),
    "warp_reduce_lowering": _exists("extra/qk_warp_reduce_lowering.py"),
    "lane_partition_reduce": _exists("extra/qk_lane_partition_reduce.py"),
  }
  target_matrix = [
    {
      "target": "cross_lane_m",
      "required_site": "lane-sharded partial max for the same (h,s) split",
      "current_site": "serial j loop computes full m redundantly per d lane inside flash_online_state_pv_tile_whole_cache_32_128",
      "lowering_available": lowerings["warp_reduce_lowering"],
      "bindable_now": False,
      "decision": "needs token/dot sharding before cross-lane max is meaningful"
    },
    {
      "target": "cross_lane_l",
      "required_site": "lane-sharded partial denominator for the same (h,s) split",
      "current_site": "serial j loop computes full l redundantly per d lane; only d==Hd is stored",
      "lowering_available": lowerings["warp_reduce_lowering"],
      "bindable_now": False,
      "decision": "needs token sharding before cross-lane add is meaningful"
    },
    {
      "target": "cross_lane_accD",
      "required_site": "multiple lanes contribute partial PV to the same output D",
      "current_site": "each local d lane owns one output D and performs the full token loop alone",
      "lowering_available": lowerings["lane_partition_reduce"],
      "bindable_now": False,
      "decision": "needs a second lane dimension or token-sharded D ownership before cross-lane PV combine exists"
    },
    {
      "target": "packed_dot_inside_tile",
      "required_site": "q.k score production inside or directly fused with online-state tile",
      "current_site": "score remains external in flash_score_whole_cache_32_128",
      "lowering_available": lowerings["fdot2_lowering"],
      "bindable_now": False,
      "decision": "needs score/tile fusion or token/dot-sharded score path before packed dot can close the primitive"
    }
  ]
  if not p5_clean:
    verdict = "ONLINE_STATE_PV_TILE_P6_FAIL__P5_NOT_CLEAN"
  elif any(r["bindable_now"] for r in target_matrix):
    verdict = "ONLINE_STATE_PV_TILE_P6_BINDABLE_LOWERING_FOUND"
  elif all(lowerings.values()):
    verdict = "ONLINE_STATE_PV_TILE_P6_NEEDS_TOKEN_SHARDED_REWRITE"
  else:
    verdict = "ONLINE_STATE_PV_TILE_P6_BLOCKED_BY_MISSING_LOWERING"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "search_space_id": manifest["search_space_id"],
    "inputs": {"p5": str(P5.relative_to(ROOT)), "manifest": str(MANIFEST.relative_to(ROOT))},
    "route_signature": programs,
    "lowerings_available": lowerings,
    "current_p5_dataflow": {
      "workgroups": "Hkv*S",
      "local_lane": "d owns PV dimension plus l/m state columns",
      "token_loop": "j serial inside each d lane",
      "m_l_state": "inside tile, but redundantly computed per d lane",
      "score": "external flash_score_whole_cache_32_128"
    },
    "target_matrix": target_matrix,
    "decision": {
      "next_scope": "P7 token-sharded online-state tile",
      "next_program_name": "flash_online_state_pv_tile_xlane_whole_cache_32_128",
      "why": "P5 moved m/l state into the tile, but every local lane still performs the full serial token loop. Cross-lane lowerings need lane-sharded partials; packed-dot needs score production inside/directly fused with the tile.",
      "do_not_do_next": [
        "global WARP_REDUCE_LOWERING without a lane-sharded site",
        "standalone score fdot2 rerun as the main path",
        "metadata fusion",
        "combine-only optimization"
      ]
    }
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-state-pv-p6-lowering-bind-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("ONLINE_STATE_PV_TILE_P6_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
