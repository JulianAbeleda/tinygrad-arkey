#!/usr/bin/env python3
"""Decode primitive detector for BubbleBeam/FutureSight physical primitive visibility."""
from __future__ import annotations
import json, pathlib, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-primitive-space"
ATTR = ROOT / "bench/qk-decode-attention-fused-score-state-pv-attribution/latest.json"
P1 = ROOT / "bench/qk-decode-primitive-space/p1_crosslane_latest.json"
PALL = ROOT / "bench/qk-decode-primitive-space/all_primitives_latest.json"


def _load(path: pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text()) if path.exists() else {}


def _kernel_row(name: str, arm: str, kr: dict[str, Any], work: dict[str, Any]) -> dict[str, Any]:
  flags = kr.get("primitive_flags", {})
  is_generated_fused = name.startswith("flash_fused_score_state_pv_tile")
  is_owned_tile = name.startswith("owned_flash_tile_gqa_whole")
  qk_redundancy = work.get("qk_dot_reductions_per_workgroup") if is_generated_fused else None
  return {
    "arm": arm,
    "kernel": name,
    "resources": {"vgpr": kr.get("vgpr"), "sgpr": kr.get("sgpr"), "lds": kr.get("lds"), "scratch": kr.get("scratch")},
    "primitive_flags": flags,
    "LaneMap": {
      "qk_owner": "per_output_column" if is_generated_fused else "per_kv_tile" if is_owned_tile else "unknown",
      "d_owner": "local_column" if is_generated_fused else "lane_group" if is_owned_tile else "unknown",
      "gqa_owner": "register_g_vector" if is_generated_fused else "shared_tile" if is_owned_tile else "unknown",
      "lane_group_width": None,
      "score_reuse": "none_across_output_columns" if is_generated_fused else "shared" if is_owned_tile else "unknown",
      "search_visibility": "inferred_absent" if is_generated_fused else "hardcoded_manual" if is_owned_tile else "unknown"
    },
    "TileMemory": {
      "kv_staging": "lds_tile" if flags.get("has_lds") else "global_direct",
      "lds_bytes": kr.get("lds"),
      "cooperative_load": bool(flags.get("has_lds") and flags.get("has_vector_global_load")),
      "barrier_count": None,
      "reuse_scope": "tile" if flags.get("has_lds") else "none_detected",
      "search_visibility": "detected_only" if is_generated_fused else "hardcoded_manual" if is_owned_tile else "unknown"
    },
    "DotLowering": {
      "lowering": "v_dot2" if flags.get("has_v_dot2") else "scalar_fma_or_generic_valu",
      "packed_inputs": bool(flags.get("has_v_dot2")),
      "dequant_placement": "unknown",
      "native_instruction": "v_dot2" if flags.get("has_v_dot2") else None,
      "search_visibility": "inferred_absent" if is_generated_fused and not flags.get("has_v_dot2") else "hardcoded_manual" if is_owned_tile else "detected_only"
    },
    "CrossLane": {
      "reduce": bool(flags.get("has_cross_lane")),
      "broadcast": bool(flags.get("has_cross_lane")),
      "instruction_family": "ds_bpermute/ds_permute" if flags.get("has_cross_lane") else None,
      "width": None,
      "search_visibility": "inferred_absent" if is_generated_fused and not flags.get("has_cross_lane") else "hardcoded_manual" if is_owned_tile else "detected_only"
    },
    "WorkShape": {
      "workgroups": work.get("fused_tile_workgroups") if is_generated_fused else work.get("baseline_owned_workgroups_approx") if is_owned_tile else None,
      "global_axes": None,
      "local_axes": ["W"] if is_generated_fused else None,
      "qk_redundancy_factor": qk_redundancy,
      "score_reuse": "repeated_per_local_output_column" if is_generated_fused else "shared_or_manual_tile" if is_owned_tile else "unknown",
      "search_visibility": "detected_only" if is_generated_fused else "hardcoded_manual" if is_owned_tile else "unknown"
    }
  }


def build() -> dict[str, Any]:
  attr = _load(ATTR)
  if not attr:
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": "PRIMITIVE_DETECTOR_FAIL__MISSING_ATTRIBUTION", "source": str(ATTR.relative_to(ROOT))}
  work = attr.get("work_shape", {})
  rows = []
  for arm_name, arm in attr.get("arms", {}).items():
    for name, kr in arm.get("kernel_resources", {}).items():
      rows.append(_kernel_row(name, arm_name, kr, work))
  p1 = _load(P1)
  pall = _load(PALL)
  p1_pass = str(p1.get("verdict", "")).startswith("P1_CROSSLANE_PASS")
  pall_pass = pall.get("verdict") == "PALL_PRIMITIVES_VISIBLE__ROUTE_INTEGRATION_NEXT"
  generated = [r for r in rows if r["kernel"].startswith("flash_fused_score_state_pv_tile")]
  owned = [r for r in rows if r["kernel"].startswith("owned_flash_tile_gqa_whole")]
  missing = []
  if generated:
    g = generated[0]
    if not g["primitive_flags"].get("has_lds"): missing.append("TileMemory.lds_tile")
    if not g["primitive_flags"].get("has_v_dot2"): missing.append("DotLowering.v_dot2")
    if not g["primitive_flags"].get("has_cross_lane"): missing.append("CrossLane.reduce_broadcast")
    if (g["WorkShape"].get("qk_redundancy_factor") or 0) > 16: missing.append("LaneMap.score_reuse_across_output_columns")
  verdict = "PRIMITIVE_DETECTOR_READY" if rows else "PRIMITIVE_DETECTOR_FAIL__NO_KERNEL_ROWS"
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "source_attribution": str(ATTR.relative_to(ROOT)),
    "rows": rows,
    "summary": {
      "generated_fused_tile_present": bool(generated),
      "owned_tile_present": bool(owned),
      "missing_generated_physical_primitives": missing,
      "search_visibility_gap": [m for m in missing],
      "p1_crosslane_probe": {
        "available": bool(p1),
        "path": str(P1.relative_to(ROOT)),
        "verdict": p1.get("verdict"),
        "crosslane_visible": p1_pass,
        "kernel_flags": next(iter(p1.get("probe", {}).get("kernels", {}).values()), {}).get("primitive_flags", {}) if p1 else {}
      },
      "all_primitives_bundle": {
        "available": bool(pall),
        "path": str(PALL.relative_to(ROOT)),
        "verdict": pall.get("verdict"),
        "all_visible": pall_pass,
        "checks": pall.get("checks", {})
      }
    }
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"primitive-detector-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("PRIMITIVE_DETECTOR_FAIL") else 1

if __name__ == "__main__":
  raise SystemExit(main())
