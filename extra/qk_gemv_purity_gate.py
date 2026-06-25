#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_latest.json'
OUT = ROOT / 'bench/qk-gemv-purity-gate'


def main() -> int:
  src = json.loads(SRC.read_text())
  rows = src['rows']
  ctxs = [str(c) for c in src['ctxs']]
  owned_warp = all(rows[c]['program_counts']['owned']['owned_gateup'] > 0 for c in ctxs)
  lane_bridge = all(rows[c]['program_counts']['bubblebeam_futuresight']['lane_partition_gateup'] > 0 for c in ctxs)
  owned_under_futuresight = any(rows[c]['program_counts']['bubblebeam_futuresight']['owned_gateup'] > 0 for c in ctxs)
  generated_arm = 'generated_skeleton' if 'generated_skeleton' in rows[ctxs[0]]['program_counts'] else None
  generated_skeleton = bool(generated_arm) and all(rows[c]['program_counts'][generated_arm]['lane_partition_gateup'] == 0 and rows[c]['program_counts'][generated_arm]['owned_gateup'] == 0 for c in ctxs)
  scheduler_generated = all(rows[c]['program_counts']['bubblebeam_futuresight']['lane_partition_gateup'] == 0 and rows[c]['program_counts']['bubblebeam_futuresight']['scheduler_programs'] > 0 for c in ctxs)
  tokens_match = bool(src.get('tokens_match_all_ctx'))
  tok_s = {c: rows[c]['tok_s']['bubblebeam_futuresight'] for c in ctxs}
  generated_tok_s = {c: rows[c]['tok_s'][generated_arm] for c in ctxs} if generated_arm else {}
  if tokens_match and lane_bridge and not owned_under_futuresight:
    verdict = 'GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE'
  elif tokens_match and scheduler_generated:
    verdict = 'GEMV_PURE_SEARCH_GENERATED'
  else:
    verdict = 'GEMV_PURITY_GATE_FAIL'
  out = {
    'date': '2026-06-25',
    'timestamp': time.strftime('%Y%m%d-%H%M%S'),
    'verdict': verdict,
    'source_artifact': str(SRC.relative_to(ROOT)),
    'route_flags': {
      'owned_warp_gemv_used_in_owned_arm': owned_warp,
      'lane_partition_custom_bridge_used_in_bubblebeam_arm': lane_bridge,
      'owned_warp_gemv_used_in_bubblebeam_arm': owned_under_futuresight,
      'scheduler_generated_route_used_in_bubblebeam_arm': scheduler_generated,
      'generated_skeleton_arm_present': bool(generated_arm),
      'generated_skeleton_route_used': generated_skeleton,
    },
    'tokens_match': tokens_match,
    'tok_s_by_ctx': tok_s,
    'generated_skeleton_tok_s_by_ctx': generated_tok_s,
    'classification': 'BubbleBeam/FutureSight is search-selected but not search-generated while it routes through qk_q4k_lane_partition_gemv custom_kernel bridge.',
  }
  OUT.mkdir(parents=True, exist_ok=True)
  latest = OUT / 'latest.json'
  stamped = OUT / f"gemv-purity-gate-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + '\n')
  stamped.write_text(json.dumps(out, indent=2) + '\n')
  print(json.dumps(out, indent=2))
  return 0 if verdict != 'GEMV_PURITY_GATE_FAIL' else 1

if __name__ == '__main__':
  raise SystemExit(main())
