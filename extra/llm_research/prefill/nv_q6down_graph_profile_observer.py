#!/usr/bin/env python3
"""Turn real HCQ graph-profile timestamps into D0 Q6-down boundary records."""
import argparse, json, pathlib

def _load(path): return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--profile",required=True);ap.add_argument("--model-json",required=True);ap.add_argument("--out",required=True)
  a=ap.parse_args();profiles=_load(a.profile);model=json.loads(pathlib.Path(a.model_json).read_text())
  if not profiles: raise SystemExit("no HCQ graph profiles")
  # The model emits six graph segments per invocation. Use the final complete
  # invocation so signals have been collected after device completion.
  group=profiles[-6:] if len(profiles)>=6 else profiles
  rows=[]
  for segment,payload in enumerate(group):
    entries=payload["entries"]
    for idx,e in enumerate(entries): rows.append({**e,"segment":segment,"index":idx,"deps":payload.get("deps",[])[idx]})
  ids=model["route"]["q6_identities"] or {};down_id=ids.get("ffn_down")
  if not down_id: raise SystemExit("model JSON has no Q6 down identity")
  producers=[r for r in rows if r["name"]=="q8_compact_record_fp16_q6_ffn_down"]
  mains=[r for r in rows if (r.get("metadata") or {}).get("canonical_identity")==down_id]
  if len(producers)!=18 or len(mains)!=18: raise SystemExit(f"D0 census mismatch producers={len(producers)} mains={len(mains)}")
  def aggregate(name,selected):
    begin=min(float(x["start"]) for x in selected);end=max(float(x["end"]) for x in selected)
    waits=[]
    for x in selected:
      same=[r for r in rows if r["segment"]==x["segment"]]
      dep_end=max((float(same[d]["end"]) for d in x["deps"] if 0<=d<len(same)),default=float(x["start"]))
      waits.append(max(0.0,float(x["start"])-dep_end))
    return {"boundary":name,"status":"OBSERVED","records":len(selected),"device_begin_us":begin,"device_end_us":end,
      "active_us":sum(float(x["duration"]) for x in selected),"dependency_wait_us":sum(waits),
      "allocations":{"status":"UNAVAILABLE","count":None,"bytes":None,"source":"hcq_graph_profile"},"copies":{"status":"UNAVAILABLE","count":None,"bytes":None,"source":"hcq_graph_profile"},"materializations":{"status":"UNAVAILABLE","count":None,"bytes":None,"source":"hcq_graph_profile"}}
  records=[aggregate("compact_q8_producer",producers),aggregate("q6_main",mains),
    {"boundary":"output_publication","status":"OBSERVED","records":18,"device_begin_us":max(float(x["end"]) for x in mains),
      "device_end_us":max(float(x["end"]) for x in mains),"active_us":0.0,"dependency_wait_us":0.0,
      "allocations":{"status":"UNAVAILABLE","count":None,"bytes":None,"source":"hcq_graph_profile"},"copies":{"status":"UNAVAILABLE","count":None,"bytes":None,"source":"hcq_graph_profile"},"materializations":{"status":"UNAVAILABLE","count":None,"bytes":None,"source":"hcq_graph_profile"},
      "ownership":"direct compiler output; publication is the q6_main completion"}]
  # A main can have many graph dependents (fanout).  The residual boundary is
  # the first temporal direct consumer; retain the full fanout for audit.
  residual=[]
  fanout=[]
  for main_row in mains:
    deps=[r for r in rows if r["segment"]==main_row["segment"] and main_row["index"] in r["deps"]]
    deps.sort(key=lambda r:(float(r["start"]),r["index"]))
    fanout.append({"main_index":main_row["index"],"successors":[r["name"] for r in deps]})
    if deps: residual.append(deps[0])
  if len(residual)!=18: raise SystemExit(f"residual successor census mismatch {len(residual)}")
  records.append(aggregate("residual_epilogue",residual))
  result={"schema":"tinygrad.nv_q6down_graph_profile_observer.v1","status":"BLOCKED","source_profile":a.profile,
    "model_json":a.model_json,"role":"ffn_down","records":records,"profile_perturbed":True,
    "note":"real HCQ timestamps; first temporal direct successor selected per main; full fanout retained; allocator/copy/materialization events and paired hot/rotated-cold controls are unavailable, so D0 cannot close",
    "residual_fanout":fanout}
  p=pathlib.Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
