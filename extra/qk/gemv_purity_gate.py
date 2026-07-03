#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / 'bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_latest.json'


def build() -> dict:
  src = json.loads(SRC.read_text())
  rows = src['rows']
  ctxs = [str(c) for c in src['ctxs']]
  owned_warp = all(rows[c]['program_counts']['owned']['owned_gateup'] > 0 for c in ctxs)
  lane_bridge = all(rows[c]['program_counts']['bubblebeam_futuresight']['lane_partition_gateup'] > 0 for c in ctxs)
  g3_under_futuresight = all(rows[c]['program_counts']['bubblebeam_futuresight'].get('g3_lanemap_gateup', 0) > 0 for c in ctxs)
  full_gemv_g3_under_futuresight = all(rows[c]['program_counts']['bubblebeam_futuresight'].get('g3_lanemap_gateup', 0) > 0 and
                                      rows[c]['program_counts']['bubblebeam_futuresight'].get('g3_lanemap_down', 0) > 0 and
                                      rows[c]['program_counts']['bubblebeam_futuresight'].get('g3_lanemap_proj', 0) > 0 and
                                      rows[c]['program_counts']['bubblebeam_futuresight'].get('owned_down', 0) == 0 and
                                      rows[c]['program_counts']['bubblebeam_futuresight'].get('owned_proj', 0) == 0 for c in ctxs)
  owned_under_futuresight = any(rows[c]['program_counts']['bubblebeam_futuresight']['owned_gateup'] > 0 for c in ctxs)
  generated_arm = 'generated_skeleton' if 'generated_skeleton' in rows[ctxs[0]]['program_counts'] else None
  g2_arm = 'g2_lanemap' if 'g2_lanemap' in rows[ctxs[0]]['program_counts'] else None
  g3_arm = 'g3_lanemap_codegen' if 'g3_lanemap_codegen' in rows[ctxs[0]]['program_counts'] else None
  generated_skeleton = bool(generated_arm) and all(rows[c]['program_counts'][generated_arm]['lane_partition_gateup'] == 0 and rows[c]['program_counts'][generated_arm]['owned_gateup'] == 0 for c in ctxs)
  g2_lanemap = bool(g2_arm) and all(rows[c]['program_counts'][g2_arm]['lane_partition_gateup'] == 0 and rows[c]['program_counts'][g2_arm]['owned_gateup'] == 0 for c in ctxs)
  g3_lanemap = bool(g3_arm) and all(rows[c]['program_counts'][g3_arm].get('g3_lanemap_gateup', 0) > 0 and rows[c]['program_counts'][g3_arm]['lane_partition_gateup'] == 0 and rows[c]['program_counts'][g3_arm]['owned_gateup'] == 0 for c in ctxs)
  scheduler_generated = all(rows[c]['program_counts']['bubblebeam_futuresight']['lane_partition_gateup'] == 0 and (rows[c]['program_counts']['bubblebeam_futuresight']['scheduler_programs'] > 0 or rows[c]['program_counts']['bubblebeam_futuresight'].get('g3_lanemap_gateup', 0) > 0) for c in ctxs)
  tokens_match = bool(src.get('tokens_match_all_ctx'))
  tok_s = {c: rows[c]['tok_s']['bubblebeam_futuresight'] for c in ctxs}
  generated_tok_s = {c: rows[c]['tok_s'][generated_arm] for c in ctxs} if generated_arm else {}
  g2_tok_s = {c: rows[c]['tok_s'][g2_arm] for c in ctxs} if g2_arm else {}
  g3_tok_s = {c: rows[c]['tok_s'][g3_arm] for c in ctxs} if g3_arm else {}
  g3_promotable = bool(src.get('g3_lanemap_verdict') == 'G3_LANEMAP_PROMOTABLE' and g3_lanemap and tokens_match)
  if tokens_match and full_gemv_g3_under_futuresight and not owned_under_futuresight and not lane_bridge:
    verdict = 'GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV'
  elif tokens_match and g3_under_futuresight and not owned_under_futuresight and not lane_bridge:
    verdict = 'GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3'
  elif tokens_match and lane_bridge and not owned_under_futuresight:
    verdict = 'GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE'
  elif tokens_match and scheduler_generated:
    verdict = 'GEMV_PURE_SEARCH_GENERATED'
  else:
    verdict = 'GEMV_PURITY_GATE_FAIL'
  classification = ('BubbleBeam/FutureSight routes all tracked Q4_K GEMV roles (gate/up, down, projection) to generated G3 LaneMap programs; no lane-partition bridge or owned Q4_K GEMV fires under BubbleBeam.'
                    if verdict == 'GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV' else
                    'BubbleBeam/FutureSight routes to the generated G3 LaneMap gate/up program; no lane-partition bridge or owned warp gate/up fires under BubbleBeam.'
                    if verdict == 'GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3' else
                    'BubbleBeam/FutureSight is search-selected but not search-generated while it routes through qk_q4k_lane_partition_gemv custom_kernel bridge.')
  out = {
    'date': '2026-06-25',
    'timestamp': time.strftime('%Y%m%d-%H%M%S'),
    'verdict': verdict,
    'source_artifact': str(SRC.relative_to(ROOT)),
    'route_flags': {
      'owned_warp_gemv_used_in_owned_arm': owned_warp,
      'lane_partition_custom_bridge_used_in_bubblebeam_arm': lane_bridge,
      'g3_lanemap_generated_used_in_bubblebeam_arm': g3_under_futuresight,
      'g3_lanemap_full_q4k_gemv_used_in_bubblebeam_arm': full_gemv_g3_under_futuresight,
      'owned_warp_gemv_used_in_bubblebeam_arm': owned_under_futuresight,
      'scheduler_generated_route_used_in_bubblebeam_arm': scheduler_generated,
      'generated_skeleton_arm_present': bool(generated_arm),
      'generated_skeleton_route_used': generated_skeleton,
      'g2_lanemap_arm_present': bool(g2_arm),
      'g2_lanemap_route_clean': g2_lanemap,
      'g3_lanemap_arm_present': bool(g3_arm),
      'g3_lanemap_route_clean': g3_lanemap,
      'g3_lanemap_promotable': g3_promotable,
    },
    'tokens_match': tokens_match,
    'tok_s_by_ctx': tok_s,
    'generated_skeleton_tok_s_by_ctx': generated_tok_s,
    'g2_lanemap_tok_s_by_ctx': g2_tok_s,
    'g2_lanemap_verdict': src.get('g2_lanemap_verdict'),
    'g3_lanemap_tok_s_by_ctx': g3_tok_s,
    'g3_lanemap_verdict': src.get('g3_lanemap_verdict'),
    'classification': classification,
  }
  return out

if __name__ == '__main__':
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("gemv_purity"))
