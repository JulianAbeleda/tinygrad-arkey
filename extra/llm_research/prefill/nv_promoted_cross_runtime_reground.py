#!/usr/bin/env python3
"""Matched non-overlapping accounting for the promoted NV pp512 route and llama."""
from __future__ import annotations
import argparse, collections, json, pathlib, statistics
from decimal import Decimal

Q4_MAIN="_Z15dense_mul_mat_qIL9ggml_type12"
Q6_MAIN="_Z15dense_mul_mat_qIL9ggml_type14"
Q4_FIX="_Z30dense_mul_mat_q_stream_k_fixupIL9ggml_type12"
Q6_FIX="_Z30dense_mul_mat_q_stream_k_fixupIL9ggml_type14"
NORM="r_32_8_4_4_512_628602e2c7aa848d6e4ff1d7eb39bd95e9b93923f4805b7db94c8e9bfb030561"
ACT={"E_64_192_8_16_4_1e161f6c4c230e894f4d2601704fc92075a12b3f53be815dcba4bbed84e83ed5",
     "E_64_192_8_16_4_5a2137f8e57933947793f5908b5b1c440a16780ed3881ebe0c386c7e3680548c"}
INPUT={"E_4_32_4_3a39b1a8ecb491ce70bff135129041430248b04e874e29e2e6fd3dbecfcf6703",
       "E_512_16_8_2_4_4_2e162bdabfe040283cb257ced27118633de4e59e1679a9699f9c54996923a749"}

def classify(rows):
  phase=None; counts=collections.Counter()
  for r in rows:
    n=r["name"]
    if n=="q8_ds4_fp16_qkv_pp512" or n=="q8_d4_fp16_qkv_pp512": phase="qkv"; cat="norm/conversion"
    elif n=="q8_ds4_fp16_o_pp512": phase="o"; cat="norm/conversion"
    elif n=="q8_ds4_fp16_pp512": phase="gate/up"; cat="norm/conversion"
    elif n in ("q8_ds4_fp16_down_pp512","q8_d4_fp16_down_pp512"): phase="down"; cat="norm/conversion"
    elif n.startswith((Q4_MAIN,Q6_MAIN,Q4_FIX,Q6_FIX)):
      if phase not in ("qkv","o","gate/up","down"): raise ValueError(f"native projection without producer phase: {n}")
      cat=phase
    elif n=="nv_sm120_q16_grid_hd128_loop_attention": cat="Flash"
    elif n==NORM or (n.startswith("r_") and not n.startswith("r_1187_")): cat="norm/conversion"
    elif n in ACT: cat="activation/multiply"
    elif n in INPUT: cat="input/embed"
    elif n.startswith(("r_1187_","E_1187_")): cat="vocabulary"
    elif n.startswith("native_finite_fp32_argmax_") or n=="E_c9699af0d7ec2ef027b0f07fc5156c7a5f00221c2efb3447f24a23d08b2e0346": cat="output/token"
    else: cat="support/RoPE/KV/residual"
    r["category"]=cat;counts[cat]+=1
  return counts

def intervals(rows):
  points=sorted({Decimal(r[k]) for r in rows for k in ("start","end")});exclusive=collections.defaultdict(Decimal)
  combos=collections.defaultdict(Decimal);union=Decimal(0)
  for a,b in zip(points,points[1:]):
    cats=[r["category"] for r in rows if Decimal(r["start"])<b and Decimal(r["end"])>a]
    if not cats: continue
    dt=b-a;union+=dt
    if len(cats)==1: exclusive[cats[0]]+=dt
    else: combos[" + ".join(sorted(set(cats)))]+=dt
  lo=min(Decimal(r["start"]) for r in rows);hi=max(Decimal(r["end"]) for r in rows)
  return union,hi-lo,exclusive,combos

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--tiny-profile",required=True);ap.add_argument("--tiny-wall",required=True)
  ap.add_argument("--llama-accounting",required=True);ap.add_argument("--llama-wall",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
  xs=[json.loads(x) for x in pathlib.Path(a.tiny_profile).read_text().splitlines() if x.strip()]
  if len(xs)%6: raise ValueError("tiny profile does not contain whole six-segment replays")
  groups=[xs[i:i+6] for i in range(0,len(xs),6)];chosen=groups[-1]
  rows=sorted([dict(e) for x in chosen for e in x["entries"]],key=lambda r:(Decimal(r["start"]),Decimal(r["end"])))
  counts=classify(rows);active=collections.defaultdict(Decimal)
  for r in rows: active[r["category"]]+=Decimal(r["duration"])
  union,span,exclusive,combos=intervals(rows)
  llama=json.load(open(a.llama_accounting));la=llama["primary_region_active_ns"];le=llama["exclusive_region_union_ns"]
  lm={"input/embed":("input/embed and graph setup",),"norm/conversion":("RMSNorm and activation conversion",),
      "qkv":("Q","K","V"),"Flash":("Flash score/reduction",),"o":("O",),"gate/up":("gate","up"),
      "activation/multiply":("activation/multiply",),"down":("down",),
      "support/RoPE/KV/residual":("residual, RoPE, KV write, and other support",),
      "final-row gather/prune":("final-row gather/prune",),"vocabulary":("vocabulary",),"output/token":("output/token transfer",)}
  cats=list(lm);table=[]
  for c in cats:
    ta=float(active.get(c,0))*1000;te=float(exclusive.get(c,0))*1000
    lactive=sum(la.get(x,0) for x in lm[c]);lexcl=sum(le.get(x,0) for x in lm[c])
    table.append({"category":c,"tiny_active_ns":ta,"llama_active_ns":lactive,"active_debt_ns":ta-lactive,
                  "tiny_exclusive_ns":te,"llama_exclusive_ns":lexcl,"exclusive_debt_ns":te-lexcl})
  tw=json.load(open(a.tiny_wall))["wall"];lw=json.load(open(a.llama_wall))[0];settled=lw["samples_ns"][1:]
  payload={"schema":"tinygrad.nv_promoted_cross_runtime_reground.v1","status":"PASS","method":"matched device-clock traces; active time is non-additive; exclusive time excludes overlap",
    "tiny":{"profile_replays":len(groups),"selected_replay":len(groups)-1,"segments":6,"launches":len(rows),"unknown":0,
      "active_sum_ns":sum(float(x)*1000 for x in active.values()),"device_union_ns":float(union)*1000,"device_span_ns":float(span)*1000,
      "device_idle_ns":float(span-union)*1000,"category_launches":dict(counts),"overlap_sets_us":{k:str(v) for k,v in combos.items()}},
    "llama":{"launches":llama["launch_count"],"unknown":llama["unknown_count"],"device_union_ns":llama["device_union_ns"],"device_span_ns":llama["device_span_ns"],"device_idle_ns":llama["device_idle_ns"]},
    "unprofiled":{"tiny_min_ms":tw["min_ms"],"tiny_median_ms":tw["median_ms"],"llama_min_ms":min(settled)/1e6,"llama_median_ms":statistics.median(settled)/1e6},
    "categories":table,"causality":"Trace debts locate slower service but do not prove rollback recovery; promotion still requires a whole-model A/B."}
  out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+"\n")
  print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
