#!/usr/bin/env python3
"""Three-cell exact-opcode host probe for predicate scaffolding."""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from collections import Counter
from pathlib import Path
from typing import Any,Callable
import numpy as np
from tinygrad import Tensor,dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import KernelInfo,Ops,UOp
from extra.qk.mmq_bounded_harness import ACTIVATION_LAYOUT_MMQ_DS4,_finite_q4k_bytes,_q8_activation_inputs
from extra.qk.mmq_invocation_v1 import _identity
from extra.qk.mmq_q4k_q8_atom import _as_u32_words

SCHEMA="tinygrad.mmq_invocation_v5_scaffolding.exact_histogram.v1"
SCAFFOLD_POINTS=(0,128,256);PHASES=("uop_construction","schedule_creation","warmed_compile_cache_lookup")

def generated_id(cmplts:int)->str:
  if cmplts not in SCAFFOLD_POINTS:raise ValueError("unsupported CMPLT scaffold point")
  return f"generated_noncandidate.mmq_invocation_v5_scaffolding.base1246.cmplt{cmplts}"

def _raw_cmp(op:Ops,value:UOp,tag:int)->UOp:return UOp(op,dtypes.bool,(value,value),arg=tag)

def _kernel(steps:int,cmplts:int)->Callable[...,UOp]:
  def kernel(out:UOp,words:UOp,values:UOp,scales:UOp,sums:UOp)->UOp:
    row,batch,lane=UOp.special(16,"gidx0"),UOp.special(16,"gidx1"),UOp.special(32,"lidx0")
    value=words[row].cast(dtypes.float32)+values[batch*256+lane].cast(dtypes.float32)
    value=value+scales[batch*8].cast(dtypes.float32)+sums[batch*8].cast(dtypes.float32)
    for step in range(steps):
      operand=words[(row+step+1)%288].cast(dtypes.float32);value=value*UOp.const(dtypes.float32,1.000001+(step%17)*1e-7)+operand
    stores=[out[batch,row].store(value,gate=lane.eq(0))]
    grouped=[_raw_cmp(Ops.CMPNE,batch,1000+group) for group in range(64)]
    ownership=[]
    for pred in range(256):ownership.append(_raw_cmp(Ops.CMPLT if pred<cmplts else Ops.CMPEQ,row,2000+pred))
    for site in range(255):
      mi,ni=divmod(site,16);gate=grouped[site//4]&ownership[site]
      if site==0:gate=gate&ownership[255]
      stores.append(out[mi,ni].store(value,gate=gate))
    return UOp.group(*stores).sink(arg=KernelInfo(name=generated_id(cmplts).replace(".","_"),opts_to_apply=()))
  return kernel

def _sink(steps:int,cmplts:int)->UOp:
  return _kernel(steps,cmplts)(UOp.placeholder((16,16),dtypes.float32,0),UOp.placeholder((288,),dtypes.uint32,1),
    UOp.placeholder((4096,),dtypes.int8,2),UOp.placeholder((128,),dtypes.float32,3),UOp.placeholder((128,),dtypes.float32,4))

def _resolve_steps()->tuple[int,int]:
  candidates=[]
  for steps in range(200):
    count=len(_sink(steps,0).toposort());candidates.append((abs(count-1246),steps,count))
    if count>=1246:break
  _,steps,count=min(candidates);return steps,count

def _hist(sink:UOp)->dict[str,int]:return dict(sorted(Counter(node.op.name for node in sink.toposort()).items()))

# Filled from the admitted formulation and deliberately strict across every opcode.
EXPECTED_HISTOGRAMS:dict[int,dict[str,int]]={
  0:{"ADD":46,"AND":256,"CAST":16,"CMPEQ":256,"CMPNE":66,"CONST":36,"FLOORMOD":14,"GROUP":1,"INDEX":274,
     "MUL":16,"PARAM":5,"RESHAPE":1,"SINK":1,"SPECIAL":3,"STORE":256},
  128:{"ADD":46,"AND":256,"CAST":16,"CMPEQ":128,"CMPLT":128,"CMPNE":66,"CONST":36,"FLOORMOD":14,"GROUP":1,"INDEX":274,
       "MUL":16,"PARAM":5,"RESHAPE":1,"SINK":1,"SPECIAL":3,"STORE":256},
  256:{"ADD":46,"AND":256,"CAST":16,"CMPLT":256,"CMPNE":66,"CONST":36,"FLOORMOD":14,"GROUP":1,"INDEX":274,
       "MUL":16,"PARAM":5,"RESHAPE":1,"SINK":1,"SPECIAL":3,"STORE":256},
}

def topology_admission(steps:int)->dict[str,Any]:
  actual={point:_hist(_sink(steps,point)) for point in SCAFFOLD_POINTS}
  expected=EXPECTED_HISTOGRAMS
  admitted=bool(expected) and actual==expected
  failures=[]
  for point in SCAFFOLD_POINTS:
    for op in sorted(set(actual[point])|set(expected.get(point,{}))):
      if actual[point].get(op,0)!=expected.get(point,{}).get(op,0):failures.append(f"cmplt{point}.{op}: expected {expected.get(point,{}).get(op,0)}, actual {actual[point].get(op,0)}")
  baseline=_hist(_base_sink(steps))
  deltas={point:{op:actual[point].get(op,0)-baseline.get(op,0) for op in ("STORE","INDEX","AND","CMPNE","CMPLT")} for point in SCAFFOLD_POINTS}
  contract=all(deltas[p]["STORE"]==255 and deltas[p]["INDEX"]==255 and deltas[p]["AND"]==256 and deltas[p]["CMPNE"]==64 and deltas[p]["CMPLT"]==p for p in SCAFFOLD_POINTS)
  return {"status":"admitted" if admitted and contract else "rejected","full_expected_histograms":expected,"full_actual_histograms":actual,
    "tracked_deltas":deltas,"tracked_contract":contract,"failure_audit":failures+([] if contract else ["tracked core delta contract failed"])}

def _base_sink(steps:int)->UOp:
  def base(out:UOp,words:UOp,values:UOp,scales:UOp,sums:UOp)->UOp:
    row,batch,lane=UOp.special(16,"gidx0"),UOp.special(16,"gidx1"),UOp.special(32,"lidx0")
    value=words[row].cast(dtypes.float32)+values[batch*256+lane].cast(dtypes.float32);value=value+scales[batch*8].cast(dtypes.float32)+sums[batch*8].cast(dtypes.float32)
    for step in range(steps):value=value*UOp.const(dtypes.float32,1.000001+(step%17)*1e-7)+words[(row+step+1)%288].cast(dtypes.float32)
    return out[batch,row].store(value,gate=lane.eq(0)).sink(arg=KernelInfo(name="generated_noncandidate_base",opts_to_apply=()))
  return base(UOp.placeholder((16,16),dtypes.float32,0),UOp.placeholder((288,),dtypes.uint32,1),UOp.placeholder((4096,),dtypes.int8,2),
    UOp.placeholder((128,),dtypes.float32,3),UOp.placeholder((128,),dtypes.float32,4))

def _clock(fn:Callable[[],Any])->tuple[Any,int]:start=time.perf_counter_ns();value=fn();return value,time.perf_counter_ns()-start
def _summary(samples:list[int],overhead:int)->dict[str,Any]:
  median=float(statistics.median(samples));return {"samples_ns":samples,"median_ns":median,"min_ns":min(samples),"max_ns":max(samples),"overhead_corrected_median_ns":max(0.0,median-overhead)}
def _linear_fit(rows:list[dict[str,Any]],phase:str)->dict[str,Any]:
  x=np.asarray([[1,r["cmplt_scaffolding"]] for r in rows],float);y=np.asarray([r["phases"][phase]["overhead_corrected_median_ns"] for r in rows]);coef=np.linalg.lstsq(x,y,rcond=None)[0];pred=x@coef
  return {"intercept_ns":float(coef[0]),"per_cmplt_ns":float(coef[1]),"r2":1-float(np.sum((y-pred)**2))/float(np.sum((y-y.mean())**2))}

def run_probe(*,rounds=30,warmups=5,seed=20260711,system_snapshot_id=None):
  if rounds<30:raise ValueError("scaffolding probe requires rounds >= 30")
  steps,total=_resolve_steps();admission=topology_admission(steps)
  if admission["status"]!="admitted":raise RuntimeError("full histogram admission failed: "+"; ".join(admission["failure_audit"]))
  q4=_finite_q4k_bytes(16,256,seed);ds4=_q8_activation_inputs(16,256,seed+1,ACTIVATION_LAYOUT_MMQ_DS4).ds4_activation;assert ds4
  resident=[Tensor(_as_u32_words(q4),dtype=dtypes.uint32,device="AMD").realize(),Tensor(np.ascontiguousarray(ds4.values.reshape(-1)),dtype=dtypes.int8,device="AMD").realize(),Tensor(np.ascontiguousarray(ds4.scales.reshape(-1)),dtype=dtypes.float32,device="AMD").realize(),Tensor(np.ascontiguousarray(ds4.sums.reshape(-1)),dtype=dtypes.float32,device="AMD").realize()]
  state={}
  for point in SCAFFOLD_POINTS:
    sink=_sink(steps,point);program=to_program(sink,Device["AMD"].renderer);source=program.src[3].arg;state[point]={"sink_uops":len(sink.toposort()),"source_bytes":len(source.encode()),"source_lines":len(source.splitlines()),"rendered_statements":source.count(";"),"source_sha256":hashlib.sha256(source.encode()).hexdigest(),"program_key":program.key.hex()}
  overhead_samples=[]
  for _ in range(2000):start=time.perf_counter_ns();overhead_samples.append(time.perf_counter_ns()-start)
  overhead=int(statistics.median(overhead_samples));samples={p:{phase:[] for phase in PHASES} for p in SCAFFOLD_POINTS}
  for p in SCAFFOLD_POINTS:
    for _ in range(warmups):lazy=Tensor.empty(16,16,dtype=dtypes.float32,device="AMD").custom_kernel(*resident,fxn=_kernel(steps,p))[0];linear,_=lazy.linear_with_vars();compile_linear(linear)
  order=[p for p in SCAFFOLD_POINTS for _ in range(rounds)];random.Random(seed).shuffle(order)
  for p in order:
    lazy,elapsed=_clock(lambda:Tensor.empty(16,16,dtype=dtypes.float32,device="AMD").custom_kernel(*resident,fxn=_kernel(steps,p))[0]);samples[p]["uop_construction"].append(elapsed)
    scheduled,elapsed=_clock(lambda:lazy.linear_with_vars());samples[p]["schedule_creation"].append(elapsed);linear,_=scheduled
    _,elapsed=_clock(lambda:compile_linear(linear));samples[p]["warmed_compile_cache_lookup"].append(elapsed)
  rows=[{"generated_id":generated_id(p),"candidate_id":None,"cmplt_scaffolding":p,"target_total_uops":1246,"achieved_total_uops":state[p]["sink_uops"],"identity":state[p],"full_opcode_histogram":admission["full_actual_histograms"][p],"phases":{phase:_summary(samples[p][phase],overhead) for phase in PHASES}} for p in SCAFFOLD_POINTS]
  return {"schema":SCHEMA,"provenance_class":"generated_host_microbenchmark",**_identity(system_snapshot_id),"candidate_ids":[],"candidate_binaries":[],"candidate_timings":[],"topology_admission":admission,"protocol":{"rounds":rounds,"warmups":warmups,"randomized_interleaved_order":order,"device_time_policy":"host-only; no launch"},"instrumentation_overhead":_summary(overhead_samples,0),"rows":rows,"scaffolding_fits":{phase:_linear_fit(rows,phase) for phase in PHASES},"production_dispatch_changed":False}

def main():
  p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--rounds",type=int,default=30);p.add_argument("--warmups",type=int,default=5);p.add_argument("--seed",type=int,default=20260711);p.add_argument("--system-snapshot-id");a=p.parse_args();r=run_probe(rounds=a.rounds,warmups=a.warmups,seed=a.seed,system_snapshot_id=a.system_snapshot_id);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
