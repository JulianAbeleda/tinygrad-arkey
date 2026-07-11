#!/usr/bin/env python3
"""Exact 2x2 pure-host UOp depth and fanout structure audit."""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from collections import Counter
from pathlib import Path
from typing import Any
from tinygrad import dtypes
from tinygrad.uop.ops import Ops,UOp
from extra.qk.mmq_invocation_v1 import _identity

SCHEMA="tinygrad.mmq_host_structure_audit.v1"
DEPTHS=(32,64);FANOUTS=(256,512);ROUNDS_MIN=30
EXPECTED_HISTOGRAM={"ADD":1243,"CONST":1,"GROUP":1,"SINK":1}
EXPECTED_EVENTS={"raw_add":1243,"group":1,"sink":1}

def generated_id(depth:int,fanout:int)->str:
  if depth not in DEPTHS or fanout not in FANOUTS:raise ValueError("unsupported structure cell")
  return f"generated_noncandidate.mmq_host_structure.depth{depth}.fanout{fanout}"

def _raw_add(a:UOp,b:UOp,tag:int)->UOp:return UOp(Ops.ADD,dtypes.float32,(a,b),arg=tag)

def _tree(node_count:int,base:UOp,tag_start:int)->tuple[UOp,list[UOp],int]:
  # An odd node count is exactly N leaves plus N-1 balanced reducers.
  leaves=(node_count+1)//2;nodes=[_raw_add(base,base,tag_start+i) for i in range(leaves)];all_nodes=list(nodes);tag=tag_start+leaves
  while len(nodes)>1:
    nxt=[]
    for idx in range(0,len(nodes),2):
      if idx+1==len(nodes):nxt.append(nodes[idx])
      else:
        nxt.append(_raw_add(nodes[idx],nodes[idx+1],tag));all_nodes.append(nxt[-1]);tag+=1
    nodes=nxt
  assert len(all_nodes)==node_count
  return nodes[0],all_nodes,tag

def build_structure(depth:int,fanout:int)->tuple[UOp,dict[str,Any]]:
  generated_id(depth,fanout)
  base=UOp.const(dtypes.float32,1.0);chain_len=depth-3;chain=[];root=base;tag=0
  for _ in range(chain_len):root=_raw_add(root,base,tag);chain.append(root);tag+=1
  remaining=1243-chain_len;assert remaining%2==0
  tree_a,nodes_a,tag=_tree(remaining//2,base,tag);tree_b,nodes_b,tag=_tree(remaining//2,base,tag)
  all_adds=chain+nodes_a+nodes_b;assert len(all_adds)==1243
  required=[root,tree_a,tree_b];required_set=set(required)
  candidates=[node for node in all_adds if node not in required_set]
  unique=required+candidates[:fanout-len(required)]
  assert len(unique)==fanout
  group_sources=unique+[unique[idx%len(unique)] for idx in range(633-len(unique))]
  group=UOp.group(*group_sources);sink=UOp.sink(group)
  return sink,{"builder_events":{"raw_add":len(all_adds),"group":1,"sink":1},"group_unique_sources":len(set(group.src)),
    "group_total_source_edges":len(group.src),"chain_adds":chain_len,"balanced_tree_nodes_each":remaining//2}

def structural_metrics(root:UOp)->dict[str,Any]:
  nodes=root.toposort();depths={}
  for node in nodes:depths[node]=1+max((depths[src] for src in node.src),default=0)
  edges=sum(len(node.src) for node in nodes);hist=dict(sorted(Counter(node.op.name for node in nodes).items()))
  return {"uops":len(nodes),"edges":edges,"shared_extra_edges":edges-(len(nodes)-1),"dependency_depth":depths[root],
          "opcode_histogram":hist,"repr_sha256":hashlib.sha256(repr(root).encode()).hexdigest()}

def structure_admission()->dict[str,Any]:
  actual={};failures=[]
  for depth in DEPTHS:
    for fanout in FANOUTS:
      root,events=build_structure(depth,fanout);metrics=structural_metrics(root);cell=f"depth{depth}.fanout{fanout}"
      actual[cell]={**metrics,**events}
      expected={"uops":1246,"edges":3120,"shared_extra_edges":1875,"dependency_depth":depth,"opcode_histogram":EXPECTED_HISTOGRAM,
                "builder_events":EXPECTED_EVENTS,"group_unique_sources":fanout,"group_total_source_edges":633}
      for key,value in expected.items():
        if actual[cell].get(key)!=value:failures.append(f"{cell}.{key}: expected {value!r}, actual {actual[cell].get(key)!r}")
  return {"status":"admitted" if not failures else "rejected","contract":{"uops":1246,"edges_range":[3050,3200],
    "shared_extra_edges_range":[1800,1950],"exact_edges":3120,"exact_shared_extra_edges":1875,"opcode_histogram":EXPECTED_HISTOGRAM,
    "builder_events":EXPECTED_EVENTS},"actual":actual,"failure_audit":failures}

def _clock(depth:int,fanout:int)->tuple[int,str]:
  start=time.perf_counter_ns();root,_=build_structure(depth,fanout);elapsed=time.perf_counter_ns()-start
  return elapsed,hashlib.sha256(repr(root).encode()).hexdigest()
def _summary(samples:list[int],overhead:int)->dict[str,Any]:
  median=float(statistics.median(samples));return {"samples_ns":samples,"median_ns":median,"min_ns":min(samples),"max_ns":max(samples),
    "overhead_corrected_median_ns":max(0.0,median-overhead)}

def _factorial_fit(rows:list[dict[str,Any]])->dict[str,Any]:
  import numpy as np
  x=np.asarray([[1,r["depth"],r["fanout"],r["depth"]*r["fanout"]] for r in rows],float)
  y=np.asarray([r["construction"]["overhead_corrected_median_ns"] for r in rows]);coef=np.linalg.solve(x,y)
  return {"terms":["intercept","depth","fanout","depth_x_fanout"],"coefficients_ns":[float(v) for v in coef],
    "saturated_design":True,"degrees_of_freedom":0,"r2":1.0,"interpretation":"bounded four-cell contrast only"}

def run_audit(*,rounds=30,warmups=5,seed=20260711,system_snapshot_id=None)->dict[str,Any]:
  if rounds<ROUNDS_MIN:raise ValueError("structure audit requires rounds >= 30")
  admission=structure_admission()
  if admission["status"]!="admitted":raise RuntimeError("structure admission failed: "+"; ".join(admission["failure_audit"]))
  overhead_samples=[]
  for _ in range(2000):start=time.perf_counter_ns();overhead_samples.append(time.perf_counter_ns()-start)
  overhead=int(statistics.median(overhead_samples));cells=[(d,f) for d in DEPTHS for f in FANOUTS]
  for cell in cells:
    for _ in range(warmups):_clock(*cell)
  order=[cell for cell in cells for _ in range(rounds)];random.Random(seed).shuffle(order);samples={cell:[] for cell in cells}
  for cell in order:
    elapsed,identity=_clock(*cell)
    if identity!=admission["actual"][f"depth{cell[0]}.fanout{cell[1]}"]["repr_sha256"]:raise RuntimeError("structure identity drift")
    samples[cell].append(elapsed)
  rows=[]
  for depth,fanout in cells:
    key=f"depth{depth}.fanout{fanout}";rows.append({"generated_id":generated_id(depth,fanout),"candidate_id":None,"depth":depth,"fanout":fanout,
      "structural_identity":admission["actual"][key],"construction":_summary(samples[(depth,fanout)],overhead)})
  return {"schema":SCHEMA,"provenance_class":"generated_host_microbenchmark",**_identity(system_snapshot_id),"candidate_ids":[],"candidate_binaries":[],
    "candidate_timings":[],"structure_admission":admission,"protocol":{"rounds":rounds,"warmups":warmups,"seed":seed,
      "randomized_interleaved_order":[list(x) for x in order],"timed_scope":"pure Python UOp construction only","device_time_policy":"no device or schedule execution"},
    "instrumentation_overhead":_summary(overhead_samples,0),"rows":rows,"factorial_contrast":_factorial_fit(rows),"production_dispatch_changed":False}

def main():
  p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--rounds",type=int,default=30);p.add_argument("--warmups",type=int,default=5);p.add_argument("--seed",type=int,default=20260711);p.add_argument("--system-snapshot-id");a=p.parse_args();r=run_audit(rounds=a.rounds,warmups=a.warmups,seed=a.seed,system_snapshot_id=a.system_snapshot_id);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
