#!/usr/bin/env python3
"""Exact-1246 UOp host probe across generated MMQ backbone families."""
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
from extra.qk.q4k_q8_fixture import ACTIVATION_LAYOUT_MMQ_DS4,make_finite_q4k_bytes,make_q8_activation_inputs
from extra.qk.mmq_invocation_v1 import _identity
from extra.qk.mmq_invocation_v5_scaffolding import _raw_cmp
from extra.qk.mmq_q4k_q8_atom import _as_u32_words

SCHEMA="tinygrad.mmq_invocation_v6.exact1246_backbone.v1"
BACKBONES=("comparison_filler","candidate_arithmetic","full_arithmetic_bitwise_control")
PHASES=("uop_construction","schedule_creation","warmed_compile_cache_lookup")

def generated_id(backbone:str)->str:
  if backbone not in BACKBONES:raise ValueError("unsupported invocation-v6 backbone")
  return f"generated_noncandidate.mmq_invocation_v6.exact1246.{backbone}"

def _backbone(value:UOp,row:UOp,words:UOp,kind:str,events:int)->tuple[UOp,list[dict[str,Any]]]:
  trace=[]
  for event in range(events):
    if kind=="comparison_filler":
      pred=_raw_cmp(Ops.CMPNE,value,6000+event);value=pred.where(value,value+UOp.const(dtypes.float32,(event+1)*1e-7))
      trace.append({"event":event,"type":"cmpne_where_add","builder_ops":["CMPNE","WHERE","ADD","CONST"]})
    elif kind=="candidate_arithmetic":
      operand=words[(row+event+1)%288].cast(dtypes.float32)
      value=value*UOp.const(dtypes.float32,1.000001+(event%13)*1e-7)+operand
      trace.append({"event":event,"type":"indexed_cast_mul_add","builder_ops":["ADD","FLOORMOD","INDEX","CAST","CONST","MUL","ADD"]})
    elif kind=="full_arithmetic_bitwise_control":
      packed=words[(row+event+1)%288];bits=(packed>>(event%16))&UOp.const(dtypes.uint32,0xF+(event==events-1))
      pred=_raw_cmp(Ops.CMPNE,bits,7000+event);candidate=value+bits.cast(dtypes.float32)*UOp.const(dtypes.float32,0.01)
      value=pred.where(candidate,value)
      trace.append({"event":event,"type":"indexed_shift_mask_cast_mul_add_cmpne_where",
                    "builder_ops":["INDEX","SHR","AND","CAST","MUL","ADD","CMPNE","WHERE"]})
    else:raise ValueError("unknown backbone")
  return value,trace

def _kernel(kind:str,events:int)->Callable[...,UOp]:
  def kernel(out:UOp,words:UOp,values:UOp,scales:UOp,sums:UOp)->UOp:
    row,batch,lane=UOp.special(16,"gidx0"),UOp.special(16,"gidx1"),UOp.special(32,"lidx0")
    value=words[row].cast(dtypes.float32)+values[batch*256+lane].cast(dtypes.float32)
    value=value+scales[batch*8].cast(dtypes.float32)+sums[batch*8].cast(dtypes.float32)
    value,_=_backbone(value,row,words,kind,events)
    stores=[out[batch,row].store(value,gate=lane.eq(0))]
    grouped=[_raw_cmp(Ops.CMPNE,batch,1000+group) for group in range(64)]
    ownership=[_raw_cmp(Ops.CMPEQ,row,2000+pred) for pred in range(256)]
    for site in range(255):
      mi,ni=divmod(site,16);gate=grouped[site//4]&ownership[site]
      if site==0:gate=gate&ownership[255]
      stores.append(out[mi,ni].store(value,gate=gate))
    return UOp.group(*stores).sink(arg=KernelInfo(name=generated_id(kind).replace(".","_"),opts_to_apply=()))
  return kernel

def _sink(kind:str,events:int)->UOp:
  return _kernel(kind,events)(UOp.placeholder((16,16),dtypes.float32,0),UOp.placeholder((288,),dtypes.uint32,1),
    UOp.placeholder((4096,),dtypes.int8,2),UOp.placeholder((128,),dtypes.float32,3),UOp.placeholder((128,),dtypes.float32,4))

def _resolve_events(kind:str)->tuple[int,int]:
  candidates=[]
  for events in range(100):
    count=len(_sink(kind,events).toposort());candidates.append((abs(count-1246),events,count))
    if count>=1246:break
  _,events,count=min(candidates);return events,count

def _hist(sink:UOp)->dict[str,int]:return dict(sorted(Counter(node.op.name for node in sink.toposort()).items()))
def _depth(root:UOp)->int:
  depths={}
  for node in root.toposort():depths[node]=1+max((depths[src] for src in node.src),default=0)
  return depths[root]
def _event_trace(kind:str,events:int)->list[dict[str,Any]]:
  value=UOp.placeholder((1,),dtypes.float32,90)[0];row=UOp.special(16,"gidx0");words=UOp.placeholder((288,),dtypes.uint32,91)
  return _backbone(value,row,words,kind,events)[1]

EXPECTED:dict[str,dict[str,Any]]={
  "comparison_filler":{"events":28,"total_uops":1246,"dependency_depth":68,
    "opcode_histogram":{"ADD":32,"AND":256,"CAST":2,"CMPEQ":256,"CMPNE":94,"CONST":49,"GROUP":1,"INDEX":260,"MUL":2,"PARAM":5,"RESHAPE":1,"SINK":1,"SPECIAL":3,"STORE":256,"WHERE":28},
    "event_type_counts":{"cmpne_where_add":28},"builder_op_event_counts":{"ADD":28,"CMPNE":28,"CONST":28,"WHERE":28}},
  "candidate_arithmetic":{"events":14,"total_uops":1246,"dependency_depth":40,
    "opcode_histogram":{"ADD":46,"AND":256,"CAST":16,"CMPEQ":256,"CMPNE":66,"CONST":35,"FLOORMOD":14,"GROUP":1,"INDEX":274,"MUL":16,"PARAM":5,"RESHAPE":1,"SINK":1,"SPECIAL":3,"STORE":256},
    "event_type_counts":{"indexed_cast_mul_add":14},"builder_op_event_counts":{"ADD":28,"CAST":14,"CONST":14,"FLOORMOD":14,"INDEX":14,"MUL":14}},
  "full_arithmetic_bitwise_control":{"events":9,"total_uops":1246,"dependency_depth":31,
    "opcode_histogram":{"ADD":31,"AND":265,"CAST":11,"CMPEQ":256,"CMPNE":75,"CONST":34,"FLOORMOD":9,"GROUP":1,"INDEX":269,"MUL":11,"PARAM":5,"RESHAPE":1,"SHR":9,"SINK":1,"SPECIAL":3,"STORE":256,"WHERE":9},
    "event_type_counts":{"indexed_shift_mask_cast_mul_add_cmpne_where":9},"builder_op_event_counts":{"ADD":9,"AND":9,"CAST":9,"CMPNE":9,"INDEX":9,"MUL":9,"SHR":9,"WHERE":9}},
}

def backbone_admission()->dict[str,Any]:
  actual={}
  for kind in BACKBONES:
    events,total=_resolve_events(kind);sink=_sink(kind,events);trace=_event_trace(kind,events)
    actual[kind]={"events":events,"total_uops":total,"dependency_depth":_depth(sink),"opcode_histogram":_hist(sink),
      "event_type_counts":dict(sorted(Counter(row["type"] for row in trace).items())),
      "builder_op_event_counts":dict(sorted(Counter(op for row in trace for op in row["builder_ops"]).items()))}
  failures=[]
  for kind in BACKBONES:
    if actual[kind]!=EXPECTED.get(kind):failures.append(f"{kind}: exact event/count/depth/histogram contract mismatch")
  return {"status":"admitted" if not failures and all(v["total_uops"]==1246 for v in actual.values()) else "rejected",
          "expected":EXPECTED,"actual":actual,"failure_audit":failures}

def _clock(fn:Callable[[],Any])->tuple[Any,int]:start=time.perf_counter_ns();value=fn();return value,time.perf_counter_ns()-start
def _summary(samples:list[int],overhead:int)->dict[str,Any]:
  median=float(statistics.median(samples));return {"samples_ns":samples,"median_ns":median,"min_ns":min(samples),"max_ns":max(samples),"overhead_corrected_median_ns":max(0.0,median-overhead)}

def run_probe(*,rounds=30,warmups=5,seed=20260711,system_snapshot_id=None):
  if rounds<30:raise ValueError("invocation-v6 requires rounds >= 30")
  admission=backbone_admission()
  if admission["status"]!="admitted":raise RuntimeError("backbone admission failed: "+"; ".join(admission["failure_audit"]))
  q4=make_finite_q4k_bytes(16,256,seed);ds4=make_q8_activation_inputs(16,256,seed+1,ACTIVATION_LAYOUT_MMQ_DS4).ds4_activation;assert ds4
  resident=[Tensor(_as_u32_words(q4),dtype=dtypes.uint32,device="AMD").realize(),Tensor(np.ascontiguousarray(ds4.values.reshape(-1)),dtype=dtypes.int8,device="AMD").realize(),Tensor(np.ascontiguousarray(ds4.scales.reshape(-1)),dtype=dtypes.float32,device="AMD").realize(),Tensor(np.ascontiguousarray(ds4.sums.reshape(-1)),dtype=dtypes.float32,device="AMD").realize()]
  state={}
  for kind in BACKBONES:
    events=admission["actual"][kind]["events"];sink=_sink(kind,events);program=to_program(sink,Device["AMD"].renderer);source=program.src[3].arg
    state[kind]={"events":events,"sink_uops":len(sink.toposort()),"dependency_depth":_depth(sink),"source_bytes":len(source.encode()),
      "source_lines":len(source.splitlines()),"rendered_statements":source.count(";"),"source_sha256":hashlib.sha256(source.encode()).hexdigest(),
      "program_key":program.key.hex(),"generated_identity_sha256":hashlib.sha256(repr(sink).encode()).hexdigest()}
  overhead_samples=[]
  for _ in range(2000):start=time.perf_counter_ns();overhead_samples.append(time.perf_counter_ns()-start)
  overhead=int(statistics.median(overhead_samples));samples={kind:{phase:[] for phase in PHASES} for kind in BACKBONES}
  for kind in BACKBONES:
    for _ in range(warmups):lazy=Tensor.empty(16,16,dtype=dtypes.float32,device="AMD").custom_kernel(*resident,fxn=_kernel(kind,state[kind]["events"]))[0];linear,_=lazy.linear_with_vars();compile_linear(linear)
  order=[kind for kind in BACKBONES for _ in range(rounds)];random.Random(seed).shuffle(order)
  for kind in order:
    lazy,elapsed=_clock(lambda:Tensor.empty(16,16,dtype=dtypes.float32,device="AMD").custom_kernel(*resident,fxn=_kernel(kind,state[kind]["events"]))[0]);samples[kind]["uop_construction"].append(elapsed)
    scheduled,elapsed=_clock(lambda:lazy.linear_with_vars());samples[kind]["schedule_creation"].append(elapsed);linear,_=scheduled
    _,elapsed=_clock(lambda:compile_linear(linear));samples[kind]["warmed_compile_cache_lookup"].append(elapsed)
  rows=[{"generated_id":generated_id(kind),"candidate_id":None,"backbone":kind,"identity":state[kind],"construction_events":_event_trace(kind,state[kind]["events"]),
    "full_opcode_histogram":admission["actual"][kind]["opcode_histogram"],"phases":{phase:_summary(samples[kind][phase],overhead) for phase in PHASES}} for kind in BACKBONES]
  return {"schema":SCHEMA,"provenance_class":"generated_host_microbenchmark",**_identity(system_snapshot_id),"candidate_ids":[],"candidate_binaries":[],"candidate_timings":[],
    "noncandidate_separation":{"generated_namespace":"generated_noncandidate.mmq_invocation_v6","candidate_builder_imported":False,
      "all_generated_identity_hashes_unique":len({state[k]["generated_identity_sha256"] for k in BACKBONES})==3},"backbone_admission":admission,
    "protocol":{"rounds":rounds,"warmups":warmups,"randomized_interleaved_order":order,"device_time_policy":"host-only; no launch"},
    "instrumentation_overhead":_summary(overhead_samples,0),"rows":rows,"production_dispatch_changed":False}

def main():
  p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--rounds",type=int,default=30);p.add_argument("--warmups",type=int,default=5);p.add_argument("--seed",type=int,default=20260711);p.add_argument("--system-snapshot-id");a=p.parse_args();r=run_probe(rounds=a.rounds,warmups=a.warmups,seed=a.seed,system_snapshot_id=a.system_snapshot_id);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
