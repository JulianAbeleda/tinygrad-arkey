#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / 'bench/qk-decode-eval/candidates.json'
SEARCH_SPACES = ROOT / 'bench/qk-search-spaces'
ALLOWED_STATUS = {
  'manual_oracle_not_search_generated',
  'search_selected_custom_bridge',
  'search_generated',
  'search_space_incomplete',
  'search_blocked_by_codegen',
  'search_blocked_by_runtime',
}
MANUAL_FLAGS = {
  'Q4K_GEMV_WARP', 'Q4K_GEMV_WARP_DOWN', 'Q4K_GEMV_WARP_PROJ',
  'Q8_FFN_HANDWRITTEN',
}


def main() -> int:
  spaces = {json.loads(p.read_text())['search_space_id'] for p in SEARCH_SPACES.glob('*.json')}
  data = json.loads(CANDIDATES.read_text())
  errors: list[str] = []
  for c in data.get('candidates', []):
    cid = c.get('id', '<missing-id>')
    sid = c.get('search_space_id')
    status = c.get('search_generation_status')
    if not sid: errors.append(f'{cid}: missing search_space_id')
    elif sid not in spaces: errors.append(f'{cid}: unknown search_space_id {sid!r}')
    if not status: errors.append(f'{cid}: missing search_generation_status')
    elif status not in ALLOWED_STATUS: errors.append(f'{cid}: invalid search_generation_status {status!r}')
    if 'purity_status' not in c: errors.append(f'{cid}: missing purity_status')
    if 'excluded_primitives' not in c: errors.append(f'{cid}: missing excluded_primitives')
    env = c.get('env', {}) or {}
    manual_env = sorted(k for k in env if k in MANUAL_FLAGS and str(env[k]) not in ('0', 'False', 'false'))
    if status == 'search_generated' and sid == 'manual_oracle_not_search_generated':
      errors.append(f'{cid}: search_generated cannot use manual_oracle_not_search_generated')
    if status == 'search_generated' and manual_env:
      errors.append(f'{cid}: search_generated but enables manual/custom flags {manual_env}')
  if errors:
    print('SEARCH_PROVENANCE_FAIL')
    for e in errors: print(f'- {e}')
    return 1
  print(json.dumps({'verdict': 'SEARCH_PROVENANCE_PASS', 'candidates': len(data.get('candidates', [])), 'search_spaces': sorted(spaces)}, indent=2))
  return 0

if __name__ == '__main__':
  raise SystemExit(main())
