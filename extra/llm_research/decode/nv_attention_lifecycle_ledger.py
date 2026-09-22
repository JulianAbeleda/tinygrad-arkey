#!/usr/bin/env python3
"""Build a same-clock installed attention edge ledger from HCQ profile JSONL."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, statistics, subprocess
from collections import Counter

HASH=re.compile(r"_[0-9a-f]{40,64}$")
SCORE="flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"
COMBINE="flash_fused_gmax_combine_f16_32_128"
O="q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096"

def canon(name:str)->str:return HASH.sub("",str(name)).strip()

def complete_replays(lines:list[dict])->tuple[tuple[int,...],list[list[dict]]]:
  sizes=[len(x.get("entries",[])) for x in lines]
  tails=Counter(sizes[i+3] for i in range(len(sizes)-3) if tuple(sizes[i:i+3])==(32,64,128))
  if not tails:raise RuntimeError("no production replay signature")
  group=(32,64,128,tails.most_common(1)[0][0]);out=[];i=0
  while i+4<=len(lines):
    if tuple(sizes[i:i+4])==group:out.append(lines[i:i+4]);i+=4
    else:i+=1
  return group,out

def pct(xs:list[float],q:float)->float:return sorted(xs)[min(len(xs)-1,int(q*len(xs)))]
def stats(xs:list[float])->dict:
  return {"min":min(xs),"median":statistics.median(xs),"mean":statistics.mean(xs),"p95":pct(xs,.95),"max":max(xs)}

def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--profile-jsonl",type=pathlib.Path,required=True)
  ap.add_argument("--full-token-dag",type=pathlib.Path);ap.add_argument("--warmup",type=int,default=3)
  ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  lines=[json.loads(x) for x in a.profile_jsonl.read_text().splitlines() if x.strip()]
  group,replays=complete_replays(lines);steady=replays[a.warmup:]
  rows=[];viol=[];token_sums=[]
  for ri,replay in enumerate(steady):
    sums={"ready_to_score":0.,"score_to_combine":0.,"combine_to_o":0.};layers=0
    for gi,g in enumerate(replay):
      es,ds=g.get("entries",[]),g.get("deps",[])
      for si,s in enumerate(es):
        if canon(s.get("name",""))!=SCORE:continue
        layers+=1
        # One layer can straddle a graph split. Its dependency is accounted
        # from the pre-split full-token DAG below, not guessed from timestamps.
        if si+2>=len(es):continue
        c,o=es[si+1],es[si+2]
        if canon(c.get("name",""))!=COMBINE or canon(o.get("name",""))!=O:
          viol.append({"kind":"chain_not_contiguous","replay":ri,"graph":gi,"index":si,
            "next":[canon(c.get("name","")),canon(o.get("name",""))]});continue
        sdeps=ds[si];cdeps=ds[si+1];odeps=ds[si+2]
        if not sdeps:viol.append({"kind":"score_has_no_deps","replay":ri,"graph":gi,"index":si});continue
        ready=max(sdeps,key=lambda x:float(es[x]["end"]));ready_node=es[ready]
        waits={"ready_to_score":float(s["start"])-float(ready_node["end"]),
          "score_to_combine":float(c["start"])-float(s["end"]),
          "combine_to_o":float(o["start"])-float(c["end"])}
        direct={"ready_to_score":ready in sdeps,"score_to_combine":si in cdeps,"combine_to_o":si+1 in odeps}
        if not all(direct.values()):viol.append({"kind":"missing_direct_dep","replay":ri,"graph":gi,"index":si,"direct":direct})
        for k,v in waits.items():sums[k]+=v
        rows.append({"replay":ri,"graph":gi,"layer_in_token":layers-1,"score_index":si,
          "ready_producer":canon(ready_node.get("name","")),"ready_producer_index":ready,
          "wait_us":waits,"direct_dependency":direct,
          "body_us":{"ready":float(ready_node["duration"]),"score":float(s["duration"]),
            "combine":float(c["duration"]),"o":float(o["duration"])}})
    sums["layers"]=layers;token_sums.append(sums)
  if not rows:raise RuntimeError("no attention chains")
  edges={k:[r["wait_us"][k] for r in rows] for k in ("ready_to_score","score_to_combine","combine_to_o")}
  cross={"count":0,"direct_raw_ready_dependencies":False,"score_nodes":0}
  if a.full_token_dag:
    dag=json.loads(a.full_token_dag.read_text());nodes={int(n["id"]):n for n in dag["nodes"]}
    score_ids=[i for i,n in nodes.items() if canon(n.get("name",""))==SCORE]
    cross_edges=[e for e in dag.get("edges",[]) if e.get("crosses_group") and int(e["to"]) in score_ids and e.get("kind")=="RAW"]
    cross_scores={int(e["to"]) for e in cross_edges}
    cross={"count":len(cross_scores),"score_nodes":len(score_ids),"raw_edge_count":len(cross_edges),
      "dag_path":str(a.full_token_dag),"dag_sha256":hashlib.sha256(a.full_token_dag.read_bytes()).hexdigest(),
      "dag_node_count":len(nodes),"dag_edge_count":len(dag.get("edges",[])),
      "dag_cross_group_edge_count":sum(bool(e.get("crosses_group")) for e in dag.get("edges",[])),
      "dag_unknown_dep_node_count":int(dag.get("summary",{}).get("unknown_dep_node_count",-1)),
      "direct_raw_ready_dependencies":len(score_ids)==36 and len(cross_scores)==1 and len(cross_edges)==2,
      "rows":[{"from":int(e["from"]),"to":int(e["to"]),"producer":canon(nodes[int(e["from"])]["name"]),
        "consumer":canon(nodes[int(e["to"])]["name"]),"spans":e.get("spans",[])} for e in cross_edges]}
  result={"schema":"tinygrad.nv_attention_lifecycle_ledger.v1",
    "commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "profile_jsonl":str(a.profile_jsonl),"profile_sha256":hashlib.sha256(a.profile_jsonl.read_bytes()).hexdigest(),
    "group_sizes":list(group),"complete_replays":len(replays),"warmup_replays":a.warmup,"steady_replays":len(steady),
    "within_group_chain_count":len(rows),"cross_group_chain":cross,"violations":viol,
    "invariants":{"within_group_chains_per_token":len(rows)//len(steady),
      "all_36_chains_accounted":len(rows)==35*len(steady) and cross["direct_raw_ready_dependencies"],
      "all_within_group_direct":not viol,"all_waits_nonnegative":all(v>=0 for xs in edges.values() for v in xs)},
    "edge_wait_us":{k:stats(v) for k,v in edges.items()},
    "per_token_edge_wait_sum_us":{k:stats([x[k] for x in token_sums]) for k in edges},
    "ready_producer_population":dict(sorted(Counter(r["ready_producer"] for r in rows).items())),
    "verdict":{"ready_to_score":"REQUIRES_ATTRIBUTION" if max(edges["ready_to_score"])>0 else "ZERO_WAIT",
      "score_to_combine":"REQUIRES_ATTRIBUTION" if max(edges["score_to_combine"])>0 else "ZERO_WAIT",
      "combine_to_o":"REQUIRES_ATTRIBUTION" if max(edges["combine_to_o"])>0 else "ZERO_WAIT"},
    "rows":rows[:36]}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2,sort_keys=True))
  return 0 if not viol and result["invariants"]["all_36_chains_accounted"] else 1

if __name__=="__main__":raise SystemExit(main())
