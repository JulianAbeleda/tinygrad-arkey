#!/usr/bin/env python3
"""P4 exhaustive codegen-target decision for decode online-PV tile."""
from __future__ import annotations

import json, pathlib, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
P2 = ROOT / "bench/qk-decode-attention-online-pv-tile/latest.json"
P3 = ROOT / "bench/qk-decode-attention-online-pv-lanemap/latest.json"
MANIFEST = ROOT / "bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json"
OUT = ROOT / "bench/qk-decode-attention-online-pv-p4-codegen-decision"


def _exists(rel: str) -> bool:
  return (ROOT / rel).exists()


def _load(rel: pathlib.Path) -> dict[str, Any]:
  return json.loads(rel.read_text())


def _target_matrix() -> list[dict[str, Any]]:
  return [
    {
      "target": "score_dot",
      "current_owner": "flash_score_whole_cache_32_128",
      "lowering_available": _exists("extra/qk_fdot2_lowering.py"),
      "bindable_now": False,
      "prior_result": "A3_1_VDOT2_SCORE_NO_TRANSFER",
      "classification": "prior_no_transfer_as_standalone_score_program",
      "reason": "fdot2 can be attempted on the separated score program, but A3.1 already showed no material W==D transfer; primitive-complete path needs score inside/directly fused with the online tile lifecycle."
    },
    {
      "target": "per_split_m",
      "current_owner": "flash_max_32",
      "lowering_available": _exists("extra/qk_warp_reduce_lowering.py"),
      "bindable_now": False,
      "prior_result": "A3_6_TILE_SCORE_MAX_NO_TRANSFER",
      "classification": "no_in_tile_lane_owned_reduction_site",
      "reason": "cross-lane max lowering exists for lane reductions, but current per-split max remains an external program and is not a lane-owned online update inside flash_online_pv_tile_whole_cache."
    },
    {
      "target": "online_l_denominator",
      "current_owner": "flash_den_32 plus denominator lane contribution in online tile",
      "lowering_available": _exists("extra/qk_warp_reduce_lowering.py"),
      "bindable_now": False,
      "prior_result": "A3_10_TILE_PROB_PARTIAL_PV_NO_TRANSFER",
      "classification": "partial_contribution_only_not_online_state",
      "reason": "the tile emits a denominator contribution in d==Hd, but global l/den remains external; there is no in-tile online l state/reduction site to lower yet."
    },
    {
      "target": "pv_accD",
      "current_owner": "flash_online_pv_tile_whole_cache_32_128 register c[G]",
      "lowering_available": _exists("extra/qk_lane_partition_reduce.py"),
      "bindable_now": False,
      "prior_result": "A3_9_TILE_PARTIAL_PV_NO_TRANSFER / A3_10 regression",
      "classification": "accumulator_present_but_no_cross_lane_combine_site",
      "reason": "PV accumulation is inside the tile, but the current GQA register loop does not create lane-owned partials that require a cross-lane combine; adding a lowering has no site to bind."
    },
    {
      "target": "final_combine",
      "current_owner": "flash_combine_32_128",
      "lowering_available": _exists("extra/qk_warp_reduce_lowering.py"),
      "bindable_now": False,
      "prior_result": "two_kernel_combine_lever_refuted",
      "classification": "not_next_speed_target",
      "reason": "combine must remain lifecycle-accounted, but cheaper/fused combine was already audited as non-actionable for bounded speed; it is not the primitive-complete tile unlock."
    },
    {
      "target": "lds_staging",
      "current_owner": "none in generated online tile",
      "lowering_available": True,
      "bindable_now": False,
      "prior_result": "decode_LDS_trap_prior",
      "classification": "requires_dataflow_and_resource_plan",
      "reason": "LDS exists natively, but decode T=1 LDS staging can reduce occupancy or duplicate cache-served reads; it should only be introduced after online m/l state and lane ownership create a real reuse target."
    }
  ]


def build() -> dict[str, Any]:
  p2, p3, manifest = _load(P2), _load(P3), _load(MANIFEST)
  matrix = _target_matrix()
  p2_clean = p2.get("verdict") == "ONLINE_PV_TILE_STRUCTURAL_ROUTE_CLEAN"
  p3_ready = p3.get("verdict") == "ONLINE_PV_TILE_P3_LANEMAP_READY"
  programs = p3.get("programs", [])
  lowerings = {
    "fdot2_lowering": _exists("extra/qk_fdot2_lowering.py"),
    "warp_reduce_lowering": _exists("extra/qk_warp_reduce_lowering.py"),
    "lane_partition_reduce": _exists("extra/qk_lane_partition_reduce.py"),
  }
  bindable = [r for r in matrix if r["bindable_now"]]
  if not (p2_clean and p3_ready):
    verdict = "ONLINE_PV_TILE_P4_FAIL__PREREQ_NOT_READY"
  elif bindable:
    verdict = "ONLINE_PV_TILE_P4_BINDABLE_CODEGEN_TARGET_FOUND"
  elif all(r["lowering_available"] for r in matrix if r["target"] in ("score_dot", "per_split_m", "online_l_denominator", "pv_accD")):
    verdict = "ONLINE_PV_TILE_P4_NEEDS_DATAFLOW_REWRITE_BEFORE_CODEGEN"
  else:
    verdict = "ONLINE_PV_TILE_P4_BLOCKED_BY_CODEGEN"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "search_space_id": manifest["search_space_id"],
    "inputs": {
      "p2": str(P2.relative_to(ROOT)),
      "p3": str(P3.relative_to(ROOT)),
      "manifest": str(MANIFEST.relative_to(ROOT))
    },
    "route_signature": programs,
    "lowerings_available": lowerings,
    "target_matrix": matrix,
    "decision": {
      "next_scope": "P5 online-state tile rewrite",
      "next_program_name": "flash_online_state_pv_tile_whole_cache_32_128",
      "why": "Current lowerings exist but have no useful in-tile reduction/dot site to bind. Move m/l state into the tile first, then attach cross-lane and packed-dot lowerings to real primitive-complete dataflow.",
      "do_not_do_next": [
        "another metadata-only fusion",
        "standalone score fdot2 rerun as the main path",
        "combine-only optimization",
        "blind LDS staging without online-state reuse target"
      ]
    }
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-pv-p4-codegen-decision-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("ONLINE_PV_TILE_P4_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
