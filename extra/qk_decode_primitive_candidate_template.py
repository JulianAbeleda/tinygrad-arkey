#!/usr/bin/env python3
"""Generate decode-attention physical primitive candidate templates from the search contract."""
from __future__ import annotations
import itertools, json, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
CONTRACT = OUT / "search_contract.json"

PRIORITY = [
  {"qk_owner": "per_query_head", "d_owner": "lane_group", "dot_lowering": "scalar_fma", "kv_staging": "global_direct", "score_broadcast": "cross_lane", "gqa_reuse": "register_g_vector"},
  {"qk_owner": "per_kv_tile", "d_owner": "lane_group", "dot_lowering": "v_dot2", "kv_staging": "global_direct", "score_broadcast": "cross_lane", "gqa_reuse": "register_g_vector"},
  {"qk_owner": "per_kv_tile", "d_owner": "lane_group", "dot_lowering": "v_dot2", "kv_staging": "lds_tile", "score_broadcast": "cross_lane", "gqa_reuse": "shared_tile"}
]


def build():
  c = json.loads(CONTRACT.read_text())
  candidates = []
  for i, knobs in enumerate(PRIORITY, 1):
    cid = f"{c['candidate_id_prefix']}_p{i}"
    missing_lowering = []
    if knobs["score_broadcast"] == "cross_lane": missing_lowering.append("LaneMap/CrossLane score broadcast lowering")
    if knobs["dot_lowering"] == "v_dot2": missing_lowering.append("v_dot2 packed-dot lowering in decode attention")
    if knobs["kv_staging"] == "lds_tile": missing_lowering.append("TileMemory LDS K/V cooperative load + barrier lowering")
    candidates.append({
      "candidate_id": cid,
      "knobs": knobs,
      "intended_fix": [k for k, v in knobs.items() if v not in ("per_output_column", "global_column", "scalar_fma", "global_direct", "none")],
      "required_lowering_support": missing_lowering,
      "gates": c["required_gates"],
      "kill_condition": "Stop if emitted ISA/resource artifact does not show the intended primitive flags or q.k redundancy remains tied to local output columns.",
      "risk": "low" if i == 1 else "medium" if i == 2 else "high"
    })
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "schema": "qk_decode_primitive_candidate_templates_v1", "source_contract": str(CONTRACT.relative_to(ROOT)), "candidates": candidates}


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "candidate_templates.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
