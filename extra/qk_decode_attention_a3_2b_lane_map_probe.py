#!/usr/bin/env python3
"""A3.2b scoped attention lane-map readiness probe."""
from __future__ import annotations

import json, os, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-a3-2b-lane-map"


def _exists(rel: str) -> bool:
  return (ROOT / rel).exists()


def build() -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture

  os.environ["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
  os.environ["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  os.environ["DECODE_ATTN_SCORE_VDOT2"] = "0"
  os.environ["WARP_REDUCE_LOWERING"] = "0"
  route = capture("a2")
  names = route["route_fire"]["program_node_names"]
  score_programs = [n for n in names if n.startswith("flash_score_whole_cache")]
  has_xlane_score = any(n.startswith("flash_score_whole_cache_xlane") for n in score_programs)
  a32 = ROOT / "bench/qk-decode-attention-a3-2-cross-lane/latest.json"
  global_blocker = None
  if a32.exists():
    try: global_blocker = json.loads(a32.read_text()).get("verdict")
    except Exception: global_blocker = "unreadable"
  static = {
    "lane_partition_reduce_exists": _exists("extra/qk_lane_partition_reduce.py"),
    "warp_reduce_lowering_exists": _exists("extra/qk_warp_reduce_lowering.py"),
    "amd_warp_reduce_exists": _exists("extra/amd_warp_reduce.py"),
    "gemv_g3_lanemap_example_exists": _exists("extra/qk_gemv_g3_codegen_lowering.py"),
    "a3_2_global_blocker_artifact": str(a32.relative_to(ROOT)) if a32.exists() else None,
    "a3_2_global_blocker_verdict": global_blocker,
  }
  route_clean = (
    route["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and
    route["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and
    route["route_counts"]["owned_flash_combine"] == 0 and
    not route["materialization"]["E_49152_present"] and
    bool(route["materialization"]["selected_route_buffer_identity"])
  )
  if not route_clean:
    verdict = "A3_2B_FAIL__A2_ROUTE_NOT_CLEAN"
  elif has_xlane_score:
    verdict = "A3_2B_ATTENTION_LANE_MAP_ALREADY_WIRED"
  elif all(static[k] for k in ("lane_partition_reduce_exists", "warp_reduce_lowering_exists", "amd_warp_reduce_exists", "gemv_g3_lanemap_example_exists")):
    verdict = "A3_2B_ATTENTION_LANE_MAP_NOT_WIRED"
  else:
    verdict = "A3_2B_BLOCKED_BY_MISSING_LANE_PRIMITIVES"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "route_clean": route_clean,
    "score_programs": score_programs,
    "has_xlane_score_program": has_xlane_score,
    "static_primitives": static,
    "route": route,
    "decision": (
      "Implement DECODE_ATTN_SCORE_XLANE=1 with explicit UOp.special/lane_partition_reduce_sum score kernel."
      if verdict == "A3_2B_ATTENTION_LANE_MAP_NOT_WIRED" else
      "Do not proceed until the verdict is classified."
    ),
  }


def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-2b-lane-map-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("A3_2B_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
