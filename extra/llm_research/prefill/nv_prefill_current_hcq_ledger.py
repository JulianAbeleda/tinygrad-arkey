#!/usr/bin/env python3
"""Current-route HCQ ledger adapter for the 18-role Q4-V composed graph."""
import argparse, collections, json, pathlib
from decimal import Decimal
from nv_prefill_hcq_exact_accounting import _primary, _specialize_roles, _interval_partition

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--profile',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
  entries=[json.loads(x) for x in pathlib.Path(a.profile).read_text().splitlines() if x.strip()]
  if len(entries)%6: raise ValueError('profile is not six graph segments per invocation')
  groups=[entries[i:i+6] for i in range(0,len(entries),6)]
  rows=[{**e,'segment':s,'segment_index':i} for s,p in enumerate(groups[-1]) for i,e in enumerate(p['entries'])]
  _specialize_roles(rows)
  layer=0; seen_q=False
  for r in rows:
    if r['primary']=='q':
      if seen_q: layer+=1
      seen_q=True
    r['layer']=layer if layer<36 else None
  counts=collections.Counter(r['primary'] for r in rows)
  expected={'q':36,'k':36,'v':18,'o':36,'gate':36,'up':36,'down':36,'flash_score_reduction':36}
  if any(counts[k]!=v for k,v in expected.items()): raise ValueError(f'role census mismatch: {dict(counts)}')
  active=collections.defaultdict(Decimal)
  for r in rows: active[r['primary']]+=Decimal(str(r['duration']))
  unknown=sum(1 for r in rows if _primary(r)[0] is None)
  timeline=_interval_partition(rows)
  payload={'schema':'tinygrad.nv_prefill_current_hcq_ledger.v1','selected_invocation':len(groups)-1,
    'available_invocations':len(groups),'segments':6,'launches':len(rows),'unknown_launches':unknown,
    'launch_counts':dict(sorted(counts.items())),'active_us':{k:str(v) for k,v in sorted(active.items())},
    'timeline':timeline,'entries':rows,'classification_basis':'named historical HCQ classifier, current 18-role Q4-V population; layers anchored by Q'}
  if unknown or sum(counts.values())!=len(rows): raise ValueError('classification closure failed')
  p=pathlib.Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'status':'PASS','launches':len(rows),'unknown_launches':unknown,'timeline':timeline},indent=2))
if __name__=='__main__': main()
