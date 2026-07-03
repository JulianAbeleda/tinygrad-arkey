#!/usr/bin/env python3
"""Hard gate for decode physical primitive visibility gaps."""
from __future__ import annotations
import json, pathlib, time
from typing import Any
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-primitive-space"
DET = OUT / "latest.json"

REQUIRED = ["TileMemory.lds_tile", "DotLowering.v_dot2", "CrossLane.reduce_broadcast", "LaneMap.score_reuse_across_output_columns"]


def build() -> dict[str, Any]:
  if not DET.exists():
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": "PRIMITIVE_GAP_FAIL__MISSING_DETECTOR_ARTIFACT"}
  d = json.loads(DET.read_text())
  missing = d.get("summary", {}).get("missing_generated_physical_primitives", [])
  rows = d.get("rows", [])
  generated = [r for r in rows if r.get("kernel", "").startswith("flash_fused_score_state_pv_tile")]
  owned = [r for r in rows if r.get("kernel", "").startswith("owned_flash_tile_gqa_whole")]
  p1 = d.get("summary", {}).get("p1_crosslane_probe", {})
  pall = d.get("summary", {}).get("all_primitives_bundle", {})
  p1_crosslane = bool(p1.get("crosslane_visible"))
  pall_visible = bool(pall.get("all_visible"))
  if not generated:
    verdict = "PRIMITIVE_GAP_FAIL__NO_GENERATED_FUSED_TILE"
  elif not owned:
    verdict = "PRIMITIVE_GAP_FAIL__NO_OWNED_COMPARATOR"
  elif pall_visible and all(x in missing for x in REQUIRED):
    verdict = "PRIMITIVE_GAP_PARTIAL__ALL_PRIMITIVES_VISIBLE_NOT_IN_FUSED_ROUTE"
  elif p1_crosslane and all(x in missing for x in REQUIRED):
    verdict = "PRIMITIVE_GAP_PARTIAL__P1_LANEMAP_CROSSLANE_VISIBLE_NOT_IN_FUSED_ROUTE"
  elif all(x in missing for x in REQUIRED):
    verdict = "PRIMITIVE_GAP_CONFIRMED__PHYSICAL_TILE_PRIMITIVES_ABSENT"
  elif missing:
    verdict = "PRIMITIVE_GAP_PARTIAL__SOME_PHYSICAL_PRIMITIVES_ABSENT"
  else:
    verdict = "PRIMITIVE_GAP_READY__SEARCHABLE_PRIMITIVES_PRESENT"
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "detector_artifact": str(DET.relative_to(ROOT)),
    "required_primitives": REQUIRED,
    "missing_primitives": missing,
    "p1_crosslane_probe": p1,
    "all_primitives_bundle": pall,
    "decision": "All missing primitive classes are emit/detect-visible, but the fused route still lacks them; next step is a single fused route integration candidate using the bundle." if "ALL_PRIMITIVES" in verdict else "P1 proves cross-lane score reduction is emit/detect visible, but the fused route still lacks it; next step is route integration or P2 v_dot2 probe." if "P1_LANEMAP" in verdict else "Do not keep tuning the current fused route. Build/search lowering support for the missing primitives before rerunning W==D." if "ABSENT" in verdict else "Primitive visibility no longer blocks search; proceed to candidate lowering gates."
  }


if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("primitive_gap"))
