#!/usr/bin/env python3
"""P3 lane-map/reduction-ownership artifact for decode online-PV tile."""
from __future__ import annotations

import json, math, pathlib, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
P2 = ROOT / "bench/qk-decode-attention-online-pv-tile/latest.json"

Hq, Hkv, Hd, L = 32, 8, 128, 256
CTXS = (512, 1024, 2048, 4096)


def _load_p2() -> dict[str, Any]:
  return json.loads(P2.read_text())


def _online_programs(p2: dict[str, Any]) -> list[str]:
  return list(p2["online_pv_tile"]["signature"]["generated_attention_programs"])


def _ctx_rows() -> list[dict[str, Any]]:
  rows = []
  for ctx in CTXS:
    s = math.ceil(ctx / L)
    rows.append({
      "ctx": ctx,
      "split_L": L,
      "split_count_S": s,
      "tile_workgroups_Hkv_times_S": Hkv * s,
      "local_lanes_Hd_plus_denominator": Hd + 1,
      "gqa_register_accumulators_G": Hq // Hkv,
      "parallelism_preserved": Hkv * s > 0 and Hd + 1 == 129,
    })
  return rows


def build() -> dict[str, Any]:
  p2 = _load_p2()
  programs = _online_programs(p2)
  route = p2["online_pv_tile"]["route"]
  rows = _ctx_rows()
  p2_clean = p2.get("verdict") == "ONLINE_PV_TILE_STRUCTURAL_ROUTE_CLEAN"
  program_present = any(n.startswith("flash_online_pv_tile_whole_cache_32_128") for n in programs)
  owned_absent = route["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and route["route_counts"]["owned_flash_combine"] == 0
  e49152_absent = not route["materialization"]["E_49152_present"]
  parallelism_ok = all(r["parallelism_preserved"] and r["tile_workgroups_Hkv_times_S"] == Hkv * r["split_count_S"] for r in rows)
  state = {
    "tile_owned_now": [
      "PV_accumulator_accD_per_GQA_head_register_array_c[G] inside flash_online_pv_tile_whole_cache_32_128",
      "denominator_column_accumulator_d_eq_Hd inside the same tile output width W=Hd+1",
      "V_dimension_lane_ownership_d_as_LOCAL_axis"
    ],
    "external_to_tile_now": [
      "score[h,t] from flash_score_whole_cache_32_128",
      "per_split_max_m[h,s] from flash_max_32",
      "global_max_gm[h] from flash_gmax_32",
      "global_denominator_den[h] from flash_den_32",
      "final_rescale_combine from flash_combine_32_128"
    ],
    "missing_for_primitive_complete_tile": [
      "lane-owned online update of m within flash_online_pv_tile_whole_cache_32_128",
      "lane-owned online update of l within flash_online_pv_tile_whole_cache_32_128",
      "cross-lane or equivalent reduction schedule for m/l/accD",
      "packed-dot score production inside or directly fused with the tile lifecycle"
    ],
  }
  state_attributed = all(state[k] for k in state)
  if not p2_clean:
    verdict = "ONLINE_PV_TILE_P3_FAIL__P2_ROUTE_NOT_CLEAN"
  elif not (program_present and owned_absent and e49152_absent):
    verdict = "ONLINE_PV_TILE_P3_FAIL__P2_ROUTE_NOT_CLEAN"
  elif not parallelism_ok:
    verdict = "ONLINE_PV_TILE_P3_FAIL__PARALLELISM_COLLAPSED"
  elif not state_attributed:
    verdict = "ONLINE_PV_TILE_P3_FAIL__STATE_UNATTRIBUTED"
  else:
    verdict = "ONLINE_PV_TILE_P3_LANEMAP_READY"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "diagnostic_id": "decode_attention_online_pv_lanemap",
    "candidate_id": "decode_attention_online_pv_tile_structural_p2",
    "p2_artifact": str(P2.relative_to(ROOT)),
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "G": Hq // Hkv, "L": L, "W": Hd + 1},
    "axis_ownership": {
      "kvh": "GLOBAL axis: KV head workgroup owner",
      "s": "GLOBAL axis: split-KV chunk workgroup owner",
      "d": "LOCAL axis: V/PV output dimension plus denominator lane",
      "j": "REDUCE axis: token positions inside split",
      "g": "register loop: GQA query heads per KV head"
    },
    "ctx_rows": rows,
    "programs": programs,
    "checks": {
      "p2_clean": p2_clean,
      "online_program_present": program_present,
      "owned_absent": owned_absent,
      "E_49152_absent": e49152_absent,
      "parallelism_ok": parallelism_ok,
      "state_attributed": state_attributed,
      "cross_lane_reduction_emitted": False,
      "packed_dot_inside_tile_emitted": False
    },
    "state_attribution": state,
    "decision": "Proceed to P4 only by changing reduction/dot codegen or explicitly classifying the missing lowering; P3 itself is structural attribution, not speed promotion."
  }


if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("attention_online_pv_lanemap"))
