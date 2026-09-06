#!/usr/bin/env python3
"""Current-route HCQ ledger adapter for the pp512 composed graph."""
import argparse, collections, json, pathlib
from decimal import Decimal
from nv_prefill_hcq_exact_accounting import _primary, _interval_partition

CURRENT_QO_ID = "4de2a30ea73fa03dcbfb788035f41c6b418acc34e4b0862840bb140170f1e72d"
CURRENT_Q6_V_ID = "47c05cc9e63ce8aa27f30b2132e96728a8b259019b5e79e0396ae94f31c3b655"
CURRENT_Q4_V_NAME = "r_8_32_32_2_2_2_2_2_2_64_2_2_2_16b68e06c3ae78a7a2268b570f8b1b25a2d6565bbe49eaee035f420e67629580"
GATE_STREAMK_NAMES = ("q4_qo_streamk", "q4k_imma_fixup_active")
Q6_DOWN_NAMES = (
  "nv_q6_oracle_broad_cta_serial_q6_tile8_fragments_q6_phase_metadata_combined_publish_factor_da_oracle_publisher_fp32_legacy_ssa_vector_both_segments_in_cta_streamk_s0",
  "nv_q6_destination_major_fixup",
)
PRODUCER_NAMES = frozenset(("q8_compact_record_fp16", "q8_compact_record_fp16_k12288",
  "q8_compact_record_fp16_q6_attn_v", "q8_streamk_record_fp16_q6_ffn_down"))
FIXUP_NAMES = frozenset(("q4k_imma_fixup_active", "nv_q6_destination_major_fixup"))


def _lifecycle_kind(row:dict) -> str:
  if row["name"] in PRODUCER_NAMES: return "projection_producer"
  if row["name"] in FIXUP_NAMES: return "projection_fixup"
  if row["primary"] in ("q", "k", "v", "o", "gate", "up", "down"): return "projection_main"
  if row["role"] == "support": return "unresolved_support"
  return row["primary"]

def _specialize_current(rows:list[dict]) -> None:
  counters=collections.Counter()
  previous_name=None
  for row in rows:
    name=row["name"]; ident=(row.get("metadata") or {}).get("canonical_identity")
    if ident == CURRENT_QO_ID: primary,tag="qo","qo_main"
    elif ident == CURRENT_Q6_V_ID: primary,tag="v","q6_v_main"
    elif name == CURRENT_Q4_V_NAME and ident is None: primary,tag="v","q4_v_main"
    elif name == "q4_down_streamk" or (name == "q4k_imma_fixup_active" and previous_name == "q4_down_streamk"):
      primary,tag="down","q4_down_main" if name == "q4_down_streamk" else "q4_down_fixup"
    elif name in GATE_STREAMK_NAMES: primary,tag="gate_up",name
    elif name in Q6_DOWN_NAMES: primary,tag="down",name
    elif name == "q6k_v_four_warp_fp16_direct_151936_4096": primary,tag="vocabulary","generated_vocabulary_main"
    else: primary,tag=_primary(row)
    if primary == "qo":
      row["primary"]=row["role"]="q" if counters[tag]%2 == 0 else "o";counters[tag]+=1
    elif primary == "gate_up":
      row["primary"]=row["role"]="gate" if counters[tag]%2 == 0 else "up";counters[tag]+=1
    else: row["primary"],row["role"]=primary,tag
    if tag in ("q4_down_main","q4_down_fixup"): counters[tag]+=1
    previous_name=name
  expected={"qo_main":72,"q4_qo_streamk":72,"q4k_imma_fixup_active":72}
  if counters["q4_down_main"] or counters["q4_down_fixup"]: expected.update({"q4_down_main":18,"q4_down_fixup":18})
  if any(counters[k] != v for k,v in expected.items()): raise ValueError(f"incomplete current dense role census: {dict(counters)}")

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--profile',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
  entries=[json.loads(x) for x in pathlib.Path(a.profile).read_text().splitlines() if x.strip()]
  if len(entries)%6: raise ValueError('profile is not six graph segments per invocation')
  groups=[entries[i:i+6] for i in range(0,len(entries),6)]
  rows=[{**e,'segment':s,'segment_index':i} for s,p in enumerate(groups[-1]) for i,e in enumerate(p['entries'])]
  _specialize_current(rows)
  layer=0; seen_q=False
  for r in rows:
    if r['primary']=='q':
      if seen_q: layer+=1
      seen_q=True
    r['layer']=layer if layer<36 else None
  counts=collections.Counter(r['primary'] for r in rows)
  q4_down=sum(r["name"]=="q4_down_streamk" for r in rows)
  expected={'q':36,'k':36,'v':36,'o':36,'gate':72,'up':72,'down':72 if q4_down else 54,'flash_score_reduction':36}
  if any(counts[k]!=v for k,v in expected.items()): raise ValueError(f'role census mismatch: {dict(counts)}')
  active=collections.defaultdict(Decimal)
  for r in rows: active[r['primary']]+=Decimal(str(r['duration']))
  lifecycle=collections.defaultdict(lambda:{"launches":0,"command_interval_us":Decimal(0)})
  for r in rows:
    kind=r["lifecycle_kind"]=_lifecycle_kind(r)
    lifecycle[kind]["launches"]+=1
    lifecycle[kind]["command_interval_us"]+=Decimal(str(r['duration']))
  lifecycle={kind:{**value,"command_interval_us":str(value["command_interval_us"])} for kind,value in lifecycle.items()}
  unknown=sum(1 for r in rows if _primary(r)[0] is None)
  # The historical classifier has a catch-all support category. Zero unknown
  # rows therefore proves interval accounting, not complete semantic attribution.
  unresolved_support=[r for r in rows if r["lifecycle_kind"] == "unresolved_support"]
  timeline=_interval_partition(rows)
  payload={'schema':'tinygrad.nv_prefill_current_hcq_ledger.v1','selected_invocation':len(groups)-1,
    'available_invocations':len(groups),'segments':6,'launches':len(rows),'unknown_launches':unknown,
    'launch_counts':dict(sorted(counts.items())),'active_us':{k:str(v) for k,v in sorted(active.items())},
    'timeline':timeline,'entries':rows,'classification_basis':'named historical HCQ classifier, current pp512 composed population; layers anchored by Q',
    'unresolved_support_launches':len(unresolved_support),
    'unresolved_support_us':str(sum((Decimal(str(r['duration'])) for r in unresolved_support),Decimal(0))),
    'timing_boundary':'HCQ command intervals, not CUPTI kernel-active duration',
    'lifecycle_components':lifecycle,
    'semantic_closure':not unresolved_support and not unknown}
  if unknown or sum(counts.values())!=len(rows): raise ValueError('classification closure failed')
  p=pathlib.Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'status':'PASS','launches':len(rows),'unknown_launches':unknown,'timeline':timeline},indent=2))
if __name__=='__main__': main()
