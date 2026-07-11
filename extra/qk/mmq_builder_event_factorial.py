#!/usr/bin/env python3
"""Eight-cell host-only builder-event factorial with canonical fixed output."""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,sys,time
from pathlib import Path
from typing import Any,Callable
import numpy as np
from tinygrad import dtypes
from tinygrad.uop.ops import Ops,UOp
from extra.qk.mmq_host_structure_audit import EXPECTED_HISTOGRAM,build_structure,structural_metrics
from extra.qk.mmq_invocation_v1 import _identity

SCHEMA="tinygrad.mmq_builder_event_factorial.v1"
QG_POINTS=(0,8);REDUCE_POINTS=(0,1);WRITEBACK_POINTS=(1,256)
CHANNELS=("group","quant","reduce","eq","store","canonicalization","residual","total")

def cell_id(qg:int,reduce:int,writeback:int)->str:return f"generated_noncandidate.builder_events.qg{qg}.reduce{reduce}.wb{writeback}"
def _group_helper(index:int)->UOp:return UOp.group(UOp.const(dtypes.float32,float(index)))
def _quant_helper(index:int)->UOp:
  value=UOp.const(dtypes.uint32,index+1);return ((value>>(index%8))&UOp.const(dtypes.uint32,0xF)).cast(dtypes.float32)
def _reduce_helper(index:int)->UOp:
  value=UOp.const(dtypes.float32,float(index))
  for step in range(8):value=value+UOp.const(dtypes.float32,float(step)+0.25)
  return value
def _eq_helper(index:int)->UOp:return UOp(Ops.CMPEQ,dtypes.bool,(UOp.const(dtypes.int32,index),UOp.const(dtypes.int32,index)),arg=index)
def _store_helper(index:int)->UOp:
  out=UOp.placeholder((256,),dtypes.float32,80);return out[index%256].store(UOp.const(dtypes.float32,float(index)))
def _canonicalization_helper()->UOp:return build_structure(32,256)[0]

HELPERS={"group":_group_helper,"quant":_quant_helper,"reduce":_reduce_helper,"eq":_eq_helper,"store":_store_helper,
         "canonicalization":_canonicalization_helper}

def _timer_overhead(samples:int=2000)->list[int]:
  out=[]
  for _ in range(samples):start=time.perf_counter_ns();out.append(time.perf_counter_ns()-start)
  return out
def _timed(fn:Callable[[],Any])->tuple[Any,int]:start=time.perf_counter_ns();value=fn();return value,time.perf_counter_ns()-start

def _execute(qg:int,reduce:int,writeback:int,overhead:int)->tuple[UOp,dict[str,int]]:
  total_start=time.perf_counter_ns();raw={}
  _,raw["group"]=_timed(lambda:[_group_helper(i) for i in range(qg)])
  _,raw["quant"]=_timed(lambda:[_quant_helper(i) for i in range(qg)])
  _,raw["reduce"]=_timed(lambda:[_reduce_helper(i) for i in range(reduce)])
  _,raw["eq"]=_timed(lambda:[_eq_helper(i) for i in range(2*writeback)])
  _,raw["store"]=_timed(lambda:[_store_helper(i) for i in range(writeback)])
  sink,raw["canonicalization"]=_timed(_canonicalization_helper)
  total=time.perf_counter_ns()-total_start
  corrected={name:max(0,value-overhead) for name,value in raw.items()};residual=max(0,total-sum(raw.values()))
  return sink,{**corrected,"residual":residual,"total":max(0,total-overhead)}

def profile_crosscheck(qg:int,reduce:int,writeback:int)->dict[str,Any]:
  counts={name:0 for name in HELPERS};codes={fn.__code__:name for name,fn in HELPERS.items()}
  def profiler(frame,event,arg):
    if event=="call" and frame.f_code in codes:counts[codes[frame.f_code]]+=1
  sys.setprofile(profiler)
  try:_execute(qg,reduce,writeback,0)
  finally:sys.setprofile(None)
  expected={"group":qg,"quant":qg,"reduce":reduce,"eq":2*writeback,"store":writeback,"canonicalization":1}
  return {"status":"PASS" if counts==expected else "FAIL","expected_calls":expected,"profiled_calls":counts,
          "method":"untimed sys.setprofile filtered by exact helper code objects"}

def _summary(samples:list[int])->dict[str,Any]:return {"samples_ns":samples,"median_ns":float(statistics.median(samples)),"min_ns":min(samples),"max_ns":max(samples)}
def _fit(rows:list[dict[str,Any]],channel:str)->dict[str,Any]:
  x=[];y=[]
  for row in rows:
    a=int(row["quant_group_calls"]==8);b=row["reduce_calls"];c=int(row["writeback_iterations"]==256)
    x.append([1,a,b,c,a*b,a*c,b*c,a*b*c]);y.append(row["channels"][channel]["median_ns"])
  coef=np.linalg.solve(np.asarray(x,float),np.asarray(y,float))
  return {"terms":["intercept","qg8","reduce1","wb256","qg_x_reduce","qg_x_wb","reduce_x_wb","qg_x_reduce_x_wb"],
          "coefficients_ns":[float(v) for v in coef],"saturated_design":True,"degrees_of_freedom":0,"r2":1.0}

def run_factorial(*,rounds=30,warmups=5,seed=20260711,system_snapshot_id=None)->dict[str,Any]:
  if rounds<30:raise ValueError("builder-event factorial requires rounds >= 30")
  overhead_samples=_timer_overhead();overhead=int(statistics.median(overhead_samples));cells=[(q,r,w) for q in QG_POINTS for r in REDUCE_POINTS for w in WRITEBACK_POINTS]
  canonical=_canonicalization_helper();canonical_metrics=structural_metrics(canonical)
  if canonical_metrics["uops"]!=1246 or canonical_metrics["opcode_histogram"]!=EXPECTED_HISTOGRAM:raise RuntimeError("canonical sink admission failed")
  crosschecks={cell_id(*cell):profile_crosscheck(*cell) for cell in cells}
  if any(row["status"]!="PASS" for row in crosschecks.values()):raise RuntimeError("profile call-count crosscheck failed")
  for cell in cells:
    for _ in range(warmups):_execute(*cell,overhead)
  order=[cell for cell in cells for _ in range(rounds)];random.Random(seed).shuffle(order);samples={cell:{channel:[] for channel in CHANNELS} for cell in cells}
  for cell in order:
    sink,channels=_execute(*cell,overhead);metrics=structural_metrics(sink)
    if metrics["repr_sha256"]!=canonical_metrics["repr_sha256"]:raise RuntimeError("canonical sink identity drift")
    for channel in CHANNELS:samples[cell][channel].append(channels[channel])
  rows=[]
  for qg,reduce,writeback in cells:
    cell=(qg,reduce,writeback);rows.append({"generated_id":cell_id(*cell),"candidate_id":None,"quant_group_calls":qg,"reduce_calls":reduce,
      "writeback_iterations":writeback,"expected_attempts":{"eq":2*writeback,"store":writeback},"profile_crosscheck":crosschecks[cell_id(*cell)],
      "channels":{channel:_summary(samples[cell][channel]) for channel in CHANNELS},"final_sink_identity":canonical_metrics})
  return {"schema":SCHEMA,"provenance_class":"generated_host_microbenchmark",**_identity(system_snapshot_id),"candidate_ids":[],"candidate_binaries":[],"candidate_timings":[],
    "factorial":{"quant_group_calls":list(QG_POINTS),"reduce_calls":list(REDUCE_POINTS),"writeback_iterations":list(WRITEBACK_POINTS)},
    "canonical_sink_contract":{"uops":1246,"opcode_histogram":EXPECTED_HISTOGRAM,"identity":canonical_metrics},
    "timer":{"clock":"perf_counter_ns","overhead_samples":2000,"overhead":_summary(overhead_samples),"subtraction":"one median empty-pair per timed channel"},
    "protocol":{"rounds":rounds,"warmups":warmups,"seed":seed,"randomized_interleaved_order":[list(x) for x in order],"host_only":True,"device_launches":0},
    "rows":rows,"factorial_coefficients":{channel:_fit(rows,channel) for channel in CHANNELS},"production_dispatch_changed":False}

def main():
  p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--rounds",type=int,default=30);p.add_argument("--warmups",type=int,default=5);p.add_argument("--seed",type=int,default=20260711);p.add_argument("--system-snapshot-id");a=p.parse_args();r=run_factorial(rounds=a.rounds,warmups=a.warmups,seed=a.seed,system_snapshot_id=a.system_snapshot_id);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
