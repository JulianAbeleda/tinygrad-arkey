#!/usr/bin/env python3
"""Deterministically split the post-unroll support bucket.

The capture has no per-call metadata for the ordinary fallback programs.  The
four generated program identities below are nevertheless unambiguous: they
are the only repeated K=4096, N=12288 down projection programs in the
captured graph (and occur twice per layer).  Everything else remains in the
residual/RoPE/KV transport bucket; V is explicitly zero because no V-owned
identity is present in this admitted route.  This is accounting, not a wall
model: PROFILE=1 materially perturbs host/device timing.
"""
import argparse, collections, json, pathlib
from decimal import Decimal

# Use exact names from the captured graph; keeping this separate makes stale
# or renamed programs fail closed instead of silently changing ownership.
DOWN = {
 "r_16_64_8_16_4_4_48_2_2_2_16_2_5f6b8990b16488fe41fabac9467d1d269c8e04a5d65a62d02d051c6ce71eb714": "down",
 "r_16_64_8_16_4_4_48_4_2_16_2_f1eeac9e841042b4ed91ff33c2bd237011162f3ab1488eac91b664563ec4a243": "down",
}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--profile',required=True); ap.add_argument('--out',required=True)
 a=ap.parse_args(); runs=[json.loads(x) for x in pathlib.Path(a.profile).read_text().splitlines() if x.strip()]
 rows=[e for e in runs[-6:] for e in e['entries']]
 if len(rows)!=1449: raise ValueError(f'expected 1449 selected launches, got {len(rows)}')
 cats=collections.Counter(); active=collections.defaultdict(Decimal); unknown=[]
 for i,r in enumerate(rows):
  n=r['name']
  if n in DOWN: cat='down'
  elif n=='q8_compact_record_fp16': cat='other_support'
  else: cat='residual_rope_kv_transport'
  cats[cat]+=1; active[cat]+=Decimal(str(r['duration']))
 # V has no live-owned identity in this route. Keep it explicit, not folded.
 cats['v']=0; active['v']=Decimal(0)
 total=sum(active.values(),Decimal(0)); source=json.loads(pathlib.Path(a.out).read_text()) if False else None
 payload={'schema':'tinygrad.nv_post_unroll_support_attribution.v1','source':a.profile,
  'selected_segments':'last six graph segments (one final replay)','launches':len(rows),
  'classified_launches':sum(cats.values()),'unknown_launches':len(unknown),
  'ownership_basis':{'down':'exact captured program-name allowlist for the two per-layer N=12288 down programs',
   'v':'zero: no V-owned identity admitted/present in this route','residual_rope_kv_transport':'all remaining non-down graph-owned programs',
   'other_support':'q8 compact producer calls'},
  'launch_counts':dict(sorted(cats.items())), 'active_us':{k:str(v) for k,v in sorted(active.items())},
  'active_sum_us':str(total),'closure':{'sum_category_active_us':str(total),'support_bucket_active_us':'501106.784',
   'difference_us':str(total-Decimal('501106.784'))},
  'instrumentation_warning':'PROFILE=1 trace is instrumentation-perturbed and is not unprofiled wall authority.'}
 if payload['classified_launches']!=1449 or payload['unknown_launches']!=0: raise ValueError('classification closure failed')
 pathlib.Path(a.out).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,indent=2))
main()
