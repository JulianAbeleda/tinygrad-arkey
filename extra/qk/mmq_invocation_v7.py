#!/usr/bin/env python3
"""Host-only writeback builder topology probe with fixed canonical output."""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,sys,time
from collections import Counter
from pathlib import Path
from typing import Any
from tinygrad import dtypes
from tinygrad.uop.ops import KernelInfo,Ops,UOp
from extra.qk.mmq_host_structure_audit import EXPECTED_HISTOGRAM,build_structure,structural_metrics
from extra.qk.mmq_invocation_v1 import _identity

SCHEMA="tinygrad.mmq_invocation_v7.writeback_builder.v1"
STYLES=("simple","candidate_shaped");ITERATIONS=(1,256)
CHANNELS=("operand","equality","index","store","hash_canonicalization","residual","total")

def generated_id(style:str,iterations:int)->str:
  if style not in STYLES or iterations not in ITERATIONS:raise ValueError("unsupported invocation-v7 cell")
  return f"generated_noncandidate.mmq_invocation_v7.{style}.wb{iterations}"

def _operand_helper(style:str,iterations:int)->list[UOp]:
  if style=="simple":return [UOp.const(dtypes.float32,float(i)) for i in range(iterations)]
  value=UOp.const(dtypes.float32,0.0)
  for step in range(32):value=value+UOp.const(dtypes.float32,float(step)+0.125)
  return [value]*iterations

def _equality_helper(style:str,iterations:int)->list[UOp]:
  if style=="simple":return [UOp(Ops.CMPEQ,dtypes.bool,(UOp.const(dtypes.int32,i),UOp.const(dtypes.int32,i)),arg=i) for i in range(iterations)]
  row,batch=UOp.special(16,"gidx0"),UOp.special(16,"gidx1");out=[]
  for i in range(iterations):
    mi,ni=divmod(i%256,16);out.append(batch.eq(mi)&row.eq(ni))
  return out

def _index_helper(style:str,iterations:int)->list[UOp]:
  if style=="simple":
    out=UOp.placeholder((256,),dtypes.float32,70);return [out[i%256] for i in range(iterations)]
  out=UOp.placeholder((16,16),dtypes.float32,70);return [out[divmod(i%256,16)] for i in range(iterations)]

def _store_helper(indices:list[UOp],values:list[UOp],gates:list[UOp])->list[UOp]:
  return [UOp(Ops.STORE,dtypes.void,(idx,value,gate),arg=i) for i,(idx,value,gate) in enumerate(zip(indices,values,gates))]

def _hash_canonicalization_helper()->tuple[UOp,str]:
  sink=build_structure(32,256)[0];return sink,hashlib.sha256(repr(sink).encode()).hexdigest()

HELPERS={"operand":_operand_helper,"equality":_equality_helper,"index":_index_helper,"store":_store_helper,
         "hash_canonicalization":_hash_canonicalization_helper}

def attempted_topology(style:str,iterations:int)->tuple[UOp,dict[str,Any]]:
  values=_operand_helper(style,iterations);gates=_equality_helper(style,iterations);indices=_index_helper(style,iterations)
  stores=_store_helper(indices,values,gates);sink=UOp.sink(*stores);nodes=sink.toposort();depths={}
  for node in nodes:depths[node]=1+max((depths[src] for src in node.src),default=0)
  hist=dict(sorted(Counter(node.op.name for node in nodes).items()));edges=sum(len(n.src) for n in nodes)
  return sink,{"uops":len(nodes),"edges":edges,"dependency_depth":depths[sink],"opcode_histogram":hist,
    "store_attempts":iterations,"equality_attempts":iterations if style=="simple" else 2*iterations,"index_attempts":iterations,
    "index_dimensions":1 if style=="simple" else 2,"operand_builds":iterations if style=="simple" else 1,
    "shared_value_store_fanout":1 if style=="simple" else iterations,"repr_sha256":hashlib.sha256(repr(sink).encode()).hexdigest()}

EXPECTED_TOPOLOGY:dict[str,dict[str,Any]]={
  "simple.wb1":{"uops":8,"edges":8,"dependency_depth":4,"opcode_histogram":{"CMPEQ":1,"CONST":3,"INDEX":1,"PARAM":1,"SINK":1,"STORE":1},
    "store_attempts":1,"equality_attempts":1,"index_attempts":1,"index_dimensions":1,"operand_builds":1,"shared_value_store_fanout":1,
    "repr_sha256":"77ebbd8ca02cd933ec5196790f8c87a9e9c0d8146fded5c6b8dc2bebbd094f88"},
  "simple.wb256":{"uops":1538,"edges":2048,"dependency_depth":4,"opcode_histogram":{"CMPEQ":256,"CONST":768,"INDEX":256,"PARAM":1,"SINK":1,"STORE":256},
    "store_attempts":256,"equality_attempts":256,"index_attempts":256,"index_dimensions":1,"operand_builds":256,"shared_value_store_fanout":1,
    "repr_sha256":"b7d68e9d69ef5d38cebf13e6dacdad96aac54f0c9880577d7a5cdc5d3bf6ea6c"},
  "candidate_shaped.wb1":{"uops":81,"edges":85,"dependency_depth":35,"opcode_histogram":{"ADD":32,"AND":1,"CMPNE":4,"CONST":37,"INDEX":1,"PARAM":1,"RESHAPE":1,"SINK":1,"SPECIAL":2,"STORE":1},
    "store_attempts":1,"equality_attempts":2,"index_attempts":1,"index_dimensions":2,"operand_builds":1,"shared_value_store_fanout":1,
    "repr_sha256":"6c16aeced4511270d282fd6b5b61f2429e7fe7015cdab7f7209747f33d6293d1"},
  "candidate_shaped.wb256":{"uops":921,"edges":2500,"dependency_depth":35,"opcode_histogram":{"ADD":32,"AND":256,"CMPNE":64,"CONST":52,"INDEX":256,"PARAM":1,"RESHAPE":1,"SINK":1,"SPECIAL":2,"STORE":256},
    "store_attempts":256,"equality_attempts":512,"index_attempts":256,"index_dimensions":2,"operand_builds":1,"shared_value_store_fanout":256,
    "repr_sha256":"c69e22f235de4daaf551dcd6acf82130b69bfe2902173a6745f49601551af081"},
}

def topology_admission()->dict[str,Any]:
  actual={};failures=[]
  for style in STYLES:
    for iterations in ITERATIONS:
      key=f"{style}.wb{iterations}";actual[key]=attempted_topology(style,iterations)[1]
      if actual[key]!=EXPECTED_TOPOLOGY.get(key):failures.append(f"{key}: exact topology contract mismatch")
  return {"status":"admitted" if not failures else "rejected","expected":EXPECTED_TOPOLOGY,"actual":actual,"failure_audit":failures}

def _timed(fn):start=time.perf_counter_ns();value=fn();return value,time.perf_counter_ns()-start
def _execute(style:str,iterations:int,overhead:int):
  total_start=time.perf_counter_ns();values,t_operand=_timed(lambda:_operand_helper(style,iterations));gates,t_eq=_timed(lambda:_equality_helper(style,iterations))
  indices,t_index=_timed(lambda:_index_helper(style,iterations));stores,t_store=_timed(lambda:_store_helper(indices,values,gates))
  (sink,identity),t_hash=_timed(_hash_canonicalization_helper);total=time.perf_counter_ns()-total_start
  raw={"operand":t_operand,"equality":t_eq,"index":t_index,"store":t_store,"hash_canonicalization":t_hash}
  channels={name:max(0,value-overhead) for name,value in raw.items()};channels["residual"]=max(0,total-sum(raw.values()));channels["total"]=max(0,total-overhead)
  return sink,identity,channels

def profile_crosscheck(style:str,iterations:int)->dict[str,Any]:
  counts={name:0 for name in HELPERS};codes={fn.__code__:name for name,fn in HELPERS.items()}
  def profiler(frame,event,arg):
    if event=="call" and frame.f_code in codes:counts[codes[frame.f_code]]+=1
  sys.setprofile(profiler)
  try:_execute(style,iterations,0)
  finally:sys.setprofile(None)
  expected={name:1 for name in HELPERS}
  return {"status":"PASS" if counts==expected else "FAIL","expected_calls":expected,"profiled_calls":counts,
    "attempts":attempted_topology(style,iterations)[1],"method":"untimed sys.setprofile exact helper code objects"}

def _summary(samples:list[int]):return {"samples_ns":samples,"median_ns":float(statistics.median(samples)),"min_ns":min(samples),"max_ns":max(samples)}
def _fit(rows:list[dict[str,Any]],channel:str):
  import numpy as np
  x=[];y=[]
  for r in rows:
    a=int(r["style"]=="candidate_shaped");b=int(r["writeback_iterations"]==256);x.append([1,a,b,a*b]);y.append(r["channels"][channel]["median_ns"])
  coef=np.linalg.solve(np.asarray(x,float),np.asarray(y,float))
  return {"terms":["intercept","candidate_shaped","wb256","candidate_x_wb"],"coefficients_ns":[float(v) for v in coef],
    "saturated_design":True,"degrees_of_freedom":0,"r2":1.0}

def run_probe(*,rounds=30,warmups=5,seed=20260711,system_snapshot_id=None):
  if rounds<30:raise ValueError("invocation-v7 requires rounds >= 30")
  admission=topology_admission()
  if admission["status"]!="admitted":raise RuntimeError("topology admission failed: "+"; ".join(admission["failure_audit"]))
  canonical,canonical_hash=_hash_canonicalization_helper();metrics=structural_metrics(canonical)
  if metrics["uops"]!=1246 or metrics["opcode_histogram"]!=EXPECTED_HISTOGRAM:raise RuntimeError("canonical graph admission failed")
  overhead_samples=[]
  for _ in range(2000):start=time.perf_counter_ns();overhead_samples.append(time.perf_counter_ns()-start)
  overhead=int(statistics.median(overhead_samples));cells=[(s,w) for s in STYLES for w in ITERATIONS];cross={generated_id(*c):profile_crosscheck(*c) for c in cells}
  if any(v["status"]!="PASS" for v in cross.values()):raise RuntimeError("profile crosscheck failed")
  for cell in cells:
    for _ in range(warmups):_execute(*cell,overhead)
  order=[cell for cell in cells for _ in range(rounds)];random.Random(seed).shuffle(order);samples={cell:{c:[] for c in CHANNELS} for cell in cells}
  for cell in order:
    sink,identity,channels=_execute(*cell,overhead)
    if identity!=canonical_hash or structural_metrics(sink)["repr_sha256"]!=metrics["repr_sha256"]:raise RuntimeError("canonical identity drift")
    for channel in CHANNELS:samples[cell][channel].append(channels[channel])
  rows=[]
  for style,iterations in cells:
    cell=(style,iterations);rows.append({"generated_id":generated_id(*cell),"candidate_id":None,"style":style,"writeback_iterations":iterations,
      "topology":admission["actual"][f"{style}.wb{iterations}"],"profile_crosscheck":cross[generated_id(*cell)],
      "channels":{channel:_summary(samples[cell][channel]) for channel in CHANNELS},"final_graph_identity":metrics})
  return {"schema":SCHEMA,"provenance_class":"generated_host_microbenchmark",**_identity(system_snapshot_id),"candidate_ids":[],"candidate_binaries":[],"candidate_timings":[],
    "topology_admission":admission,"canonical_graph_contract":{"uops":1246,"opcode_histogram":EXPECTED_HISTOGRAM,"repr_sha256":canonical_hash},
    "timer":{"clock":"perf_counter_ns","overhead_samples":2000,"overhead":_summary(overhead_samples)},
    "protocol":{"rounds":rounds,"warmups":warmups,"seed":seed,"randomized_interleaved_order":[list(x) for x in order],"host_only":True,"device_launches":0},
    "rows":rows,"factorial_contrasts":{channel:_fit(rows,channel) for channel in CHANNELS},"production_dispatch_changed":False}

def main():
  p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--rounds",type=int,default=30);p.add_argument("--warmups",type=int,default=5);p.add_argument("--seed",type=int,default=20260711);p.add_argument("--system-snapshot-id");a=p.parse_args();r=run_probe(rounds=a.rounds,warmups=a.warmups,seed=a.seed,system_snapshot_id=a.system_snapshot_id);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
