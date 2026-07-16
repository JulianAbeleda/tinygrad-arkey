#!/usr/bin/env python3
"""Exact-operation admission probe for generated MMQ false-site topology."""
from __future__ import annotations

import argparse, hashlib, json, random, statistics, time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.q4k_q8_fixture import ACTIVATION_LAYOUT_MMQ_DS4, make_finite_q4k_bytes, make_q8_activation_inputs
from extra.qk.mmq_invocation_v1 import _identity
from extra.qk.mmq_q4k_q8_atom import _as_u32_words

SCHEMA="tinygrad.mmq_invocation_v4.exact_topology.v1"
TOPOLOGIES=("baseline","one_and_255","two_and_255")
PHASES=("uop_construction","schedule_creation","warmed_compile_cache_lookup")

def generated_id(topology:str)->str:
  if topology not in TOPOLOGIES:raise ValueError("unsupported invocation-v4 topology")
  return f"generated_noncandidate.mmq_invocation_v4.base405.{topology}"

def _raw_ne(value:UOp,offset:int)->UOp:
  expr=value+UOp.const(dtypes.int32,offset)
  return UOp(Ops.CMPNE,dtypes.bool,(expr,expr))

def _kernel(steps:int,topology:str)->Callable[...,UOp]:
  name=generated_id(topology).replace(".","_")
  def kernel(out:UOp,words:UOp,values:UOp,scales:UOp,sums:UOp)->UOp:
    row,batch,lane=UOp.special(16,"gidx0"),UOp.special(16,"gidx1"),UOp.special(32,"lidx0")
    value=words[row].cast(dtypes.float32)+values[batch*256+lane].cast(dtypes.float32)
    value=value+scales[batch*8].cast(dtypes.float32)+sums[batch*8].cast(dtypes.float32)
    for step in range(steps):
      operand=words[(row+step+1)%288].cast(dtypes.float32)
      value=value*UOp.const(dtypes.float32,1.000001+(step%17)*1e-7)+operand
    stores=[out[batch,row].store(value,gate=lane.eq(0))]
    sites=0 if topology=="baseline" else 255
    for site in range(sites):
      mi,ni=divmod(site,16)
      gate=_raw_ne(batch,1000+site)&_raw_ne(row,2000+site)
      if topology=="two_and_255":gate=gate&_raw_ne(lane,3000+site)
      stores.append(out[mi,ni].store(value,gate=gate))
    return UOp.group(*stores).sink(arg=KernelInfo(name=name,opts_to_apply=()))
  return kernel

def _sink(steps:int,topology:str)->UOp:
  return _kernel(steps,topology)(UOp.placeholder((16,16),dtypes.float32,0),UOp.placeholder((288,),dtypes.uint32,1),
    UOp.placeholder((4096,),dtypes.int8,2),UOp.placeholder((128,),dtypes.float32,3),UOp.placeholder((128,),dtypes.float32,4))

def _resolve_steps()->tuple[int,int]:
  candidates=[]
  for steps in range(200):
    count=len(_sink(steps,"baseline").toposort());candidates.append((abs(count-405),steps,count))
    if count>=405:break
  _,steps,count=min(candidates);return steps,count

def _histogram(sink:UOp)->dict[str,int]:return dict(sorted(Counter(node.op.name for node in sink.toposort()).items()))

def _admission(steps:int)->dict[str,Any]:
  hist={topology:_histogram(_sink(steps,topology)) for topology in TOPOLOGIES};base=hist["baseline"]
  expected={"baseline":{"STORE":0,"INDEX":0,"CMPNE":0,"AND":0},
            "one_and_255":{"STORE":255,"INDEX":255,"CMPNE":510,"AND":255},
            "two_and_255":{"STORE":255,"INDEX":255,"CMPNE":765,"AND":510}}
  actual={topology:{op:hist[topology].get(op,0)-base.get(op,0) for op in ("STORE","INDEX","CMPNE","AND")} for topology in TOPOLOGIES}
  admitted=actual==expected
  return {"status":"admitted" if admitted else "rejected","expected_deltas":expected,"actual_deltas":actual,
          "full_histograms":hist,"failure_audit":[] if admitted else [f"{t}.{op}: expected {expected[t][op]}, actual {actual[t][op]}"
            for t in TOPOLOGIES for op in expected[t] if expected[t][op]!=actual[t][op]]}

def _clock(fn:Callable[[],Any])->tuple[Any,int]:
  start=time.perf_counter_ns();value=fn();return value,time.perf_counter_ns()-start

def _summary(samples:list[int],overhead:int)->dict[str,Any]:
  median=float(statistics.median(samples));return {"samples_ns":samples,"median_ns":median,"min_ns":min(samples),"max_ns":max(samples),
    "overhead_corrected_median_ns":max(0.0,median-overhead)}

def _topology_contrasts(rows:list[dict[str,Any]],phase:str)->dict[str,Any]:
  med={r["topology"]:r["phases"][phase]["overhead_corrected_median_ns"] for r in rows};base=med["baseline"]
  return {"baseline_ns":base,"one_and_255_delta_ns":med["one_and_255"]-base,"two_and_255_delta_ns":med["two_and_255"]-base,
          "second_and_cmpne_increment_ns":med["two_and_255"]-med["one_and_255"],
          "per_site":{"one_and_ns":(med["one_and_255"]-base)/255,"two_and_ns":(med["two_and_255"]-base)/255,
                      "second_and_cmpne_increment_ns":(med["two_and_255"]-med["one_and_255"])/255}}

def run_invocation_v4(*,rounds:int=30,warmups:int=5,seed:int=20260711,system_snapshot_id:str|None=None)->dict[str,Any]:
  if rounds<30:raise ValueError("invocation-v4 requires rounds >= 30")
  if warmups<1:raise ValueError("invocation-v4 requires warmups >= 1")
  steps,base_uops=_resolve_steps();admission=_admission(steps)
  if admission["status"]!="admitted":raise RuntimeError("invocation-v4 topology admission failed: "+"; ".join(admission["failure_audit"]))
  q4=make_finite_q4k_bytes(16,256,seed);activation=make_q8_activation_inputs(16,256,seed+1,ACTIVATION_LAYOUT_MMQ_DS4);ds4=activation.ds4_activation
  if ds4 is None:raise RuntimeError("DS4 construction failed")
  resident={"words":Tensor(_as_u32_words(q4),dtype=dtypes.uint32,device="AMD").realize(),
    "values":Tensor(np.ascontiguousarray(ds4.values.reshape(-1)),dtype=dtypes.int8,device="AMD").realize(),
    "scales":Tensor(np.ascontiguousarray(ds4.scales.reshape(-1)),dtype=dtypes.float32,device="AMD").realize(),
    "sums":Tensor(np.ascontiguousarray(ds4.sums.reshape(-1)),dtype=dtypes.float32,device="AMD").realize()}
  state={}
  for topology in TOPOLOGIES:
    sink=_sink(steps,topology);program=to_program(sink,Device["AMD"].renderer);source=program.src[3].arg
    state[topology]={"sink_uops":len(sink.toposort()),"source_bytes":len(source.encode()),"source_lines":len(source.splitlines()),
      "rendered_statements":source.count(";"),"source_sha256":hashlib.sha256(source.encode()).hexdigest(),"program_key":program.key.hex()}
  overhead_samples=[]
  for _ in range(2000):start=time.perf_counter_ns();overhead_samples.append(time.perf_counter_ns()-start)
  overhead=int(statistics.median(overhead_samples));samples={topology:{phase:[] for phase in PHASES} for topology in TOPOLOGIES}
  for topology in TOPOLOGIES:
    for _ in range(warmups):
      lazy=Tensor.empty(16,16,dtype=dtypes.float32,device="AMD").custom_kernel(resident["words"],resident["values"],resident["scales"],resident["sums"],fxn=_kernel(steps,topology))[0]
      linear,_=lazy.linear_with_vars();compile_linear(linear)
  order=[topology for topology in TOPOLOGIES for _ in range(rounds)];random.Random(seed).shuffle(order)
  for topology in order:
    lazy,elapsed=_clock(lambda:Tensor.empty(16,16,dtype=dtypes.float32,device="AMD").custom_kernel(
      resident["words"],resident["values"],resident["scales"],resident["sums"],fxn=_kernel(steps,topology))[0])
    samples[topology]["uop_construction"].append(elapsed)
    scheduled,elapsed=_clock(lambda:lazy.linear_with_vars());samples[topology]["schedule_creation"].append(elapsed);linear,_=scheduled
    _,elapsed=_clock(lambda:compile_linear(linear));samples[topology]["warmed_compile_cache_lookup"].append(elapsed)
  rows=[{"generated_id":generated_id(topology),"candidate_id":None,"topology":topology,"base_achieved_uops":base_uops,
    "identity":state[topology],"op_counts":admission["full_histograms"][topology],
    "phases":{phase:_summary(samples[topology][phase],overhead) for phase in PHASES}} for topology in TOPOLOGIES]
  return {"schema":SCHEMA,"provenance_class":"generated_host_microbenchmark",**_identity(system_snapshot_id),"shape":{"M":16,"N":16,"K":256},
    "candidate_ids":[],"candidate_binaries":[],"candidate_timings":[],"base_target_uops":405,"base_achieved_uops":base_uops,
    "topology_admission":admission,"protocol":{"rounds":rounds,"warmups":warmups,"seed":seed,"randomized_interleaved_order":order,
      "device_time_policy":"host-only; no device launch or candidate timing"},"instrumentation_overhead":_summary(overhead_samples,0),
    "rows":rows,"topology_contrasts":{phase:_topology_contrasts(rows,phase) for phase in PHASES},"production_dispatch_changed":False}

def main()->None:
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path);parser.add_argument("--rounds",type=int,default=30)
  parser.add_argument("--warmups",type=int,default=5);parser.add_argument("--seed",type=int,default=20260711);parser.add_argument("--system-snapshot-id")
  args=parser.parse_args();result=run_invocation_v4(rounds=args.rounds,warmups=args.warmups,seed=args.seed,system_snapshot_id=args.system_snapshot_id)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":main()
