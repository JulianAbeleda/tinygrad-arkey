#!/usr/bin/env python3
"""Factorial generated probes for false writeback sites, LDS, and store interactions."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, statistics, os, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, UOp
from extra.qk.mmq_compile_evidence import analyze_final_isa, disassemble_amdgpu, parse_amdgpu_metadata

SCHEMA="tinygrad.mmq_residual_probe.v4"

@dataclass(frozen=True)
class ResidualCase:
  false_sites:int
  lds_stage:bool
  real_stores:int
  @property
  def case_id(self): return f"residual.false{self.false_sites}.lds{int(self.lds_stage)}.real{self.real_stores}"

def _kernel(case:ResidualCase):
  def kernel(out:UOp, inp:UOp):
    gid,lane=UOp.special(16,"gidx0"),UOp.special(64,"lidx0")
    value=inp[lane]
    ordered=None
    if case.lds_stage:
      lds=UOp.placeholder((64,),dtypes.float32,230,addrspace=AddrSpace.LOCAL)
      stage=lds[lane].store(value); ordered=UOp.barrier(UOp.group(stage)); value=lds.after(ordered)[lane]
    stores=[out[i].store(value) for i in range(case.real_stores)]
    for site in range(case.false_sites):
      # Runtime launches gidx0=0 only; these remain statically present but dynamically false.
      stores.append(out[case.real_stores+site].store(value,gate=gid.eq(1+site%15)))
    return UOp.group(*stores).sink(arg=KernelInfo(name=case.case_id.replace(".","_"),opts_to_apply=()))
  return kernel

def _sink(case): return _kernel(case)(UOp.placeholder((case.real_stores+case.false_sites,),dtypes.float32,0),UOp.placeholder((64,),dtypes.float32,1))

def run_case(case:ResidualCase,*,warmups=5,rounds=30,system_snapshot_id:str):
  if case.false_sites<0 or case.real_stores<1: raise ValueError("invalid residual case")
  inp=Tensor.empty(64,dtype=dtypes.float32,device="AMD").realize()
  out=Tensor.empty(case.real_stores+case.false_sites,dtype=dtypes.float32,device="AMD").custom_kernel(inp,fxn=_kernel(case))[0].realize()
  prg=to_program(_sink(case),Device["AMD"].renderer)
  from tinygrad.engine.realize import runtime_cache
  rt=runtime_cache[(prg.key,"AMD")]
  if rt.lib!=prg.src[4].arg: raise RuntimeError("loaded residual binary mismatch")
  args=(out.uop.buffer._buf,inp.uop.buffer._buf)
  gs=(1,1,1);ls=prg.arg.local_size
  from tinygrad.device import Compiled
  Compiled.profile_events.clear()
  for _ in range(warmups):rt(*args,global_size=gs,local_size=ls,wait=True)
  samples=[float(rt(*args,global_size=gs,local_size=ls,wait=True))*1e3 for _ in range(rounds)]
  binary=rt.lib;meta=parse_amdgpu_metadata(binary);disasm,tool=disassemble_amdgpu(binary);isa=analyze_final_isa(disasm,wavefront_size=meta["wavefront_size"])
  result={"schema":SCHEMA,"provenance_class":"generated_microbenchmark","case_id":case.case_id,"knobs":case.__dict__,
    "system_snapshot_id":system_snapshot_id,"binary_sha256":hashlib.sha256(binary).hexdigest(),"isa_sha256":hashlib.sha256(disasm.encode()).hexdigest(),
    "resources":meta,"isa_summary":{k:v for k,v in isa.items() if k!="instructions"},"samples_ms":samples,"median_ms":statistics.median(samples),
    "protocol":{"warmups":warmups,"rounds":rounds,"timed_global_size":[1,1,1],"compiled_global_size":list(prg.arg.global_size),"gpu_timestamp_only":True},
    "dynamic_contract":{"gidx0":0,"false_sites_execute":False,"real_stores_execute":case.real_stores,"transactions":"profile_pass_required"},
    "disassembly_tool":tool,"production_dispatch_changed":False}
  events=[e for e in Compiled.profile_events if type(e).__name__=="ProfilePMCEvent"]
  if events:
    from extra.qk.mmq_amd_pmc import _decode_event
    counters=_decode_event(events[-1]); result["sq_profile"]={name:{"value":value,"status":"live" if value>0 else "zero_suspect"} for name,value in counters.items()}
  return result

def collect_sq_profile(case:ResidualCase,*,system_snapshot_id:str,timeout=90):
  code=("import json;from extra.qk.mmq_residual_probe import ResidualCase,run_case;"
        f"print(json.dumps(run_case(ResidualCase({case.false_sites},{case.lds_stage!r},{case.real_stores}),warmups=1,rounds=3,system_snapshot_id={system_snapshot_id!r})))")
  env=dict(os.environ,PROFILE="1",PMC="1",PMC_COUNTERS="SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_WAVES,SQ_WAVE_CYCLES,SQ_WAIT_ANY",VIZ="0")
  proc=subprocess.run([sys.executable,"-c",code],env=env,text=True,capture_output=True,timeout=timeout,check=True)
  row=json.loads(proc.stdout.splitlines()[-1]); return {"case_id":case.case_id,"binary_sha256":row["binary_sha256"],"sq_profile":row.get("sq_profile",{}),"stderr":proc.stderr[-2000:]}

def run_matrix(output:Path,*,system_snapshot_id:str,warmups=5,rounds=30):
  output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
  for false in (0,32,128,256):
    for lds in (False,True):
      for real in (1,8):
        row=run_case(ResidualCase(false,lds,real),warmups=warmups,rounds=rounds,system_snapshot_id=system_snapshot_id)
        (output/(row["case_id"]+".json")).write_text(json.dumps(row,indent=2,sort_keys=True)+"\n");rows.append(row)
  X=[];y=[]
  for r in rows:
    k=r["knobs"]; sites=r["isa_summary"]["global_store_sites"]; branches=r["isa_summary"]["branch_sites"]
    X.append([1,sites,branches,int(k["lds_stage"]),sites*int(k["lds_stage"]),k["real_stores"]]);y.append(r["median_ms"])
  coef=np.linalg.lstsq(np.asarray(X,float),np.asarray(y),rcond=None)[0];pred=np.asarray(X)@coef
  r2=1-float(np.sum((np.asarray(y)-pred)**2))/float(np.sum((np.asarray(y)-np.mean(y))**2))
  manifest={"schema":SCHEMA,"system_snapshot_id":system_snapshot_id,"cases":len(rows),"fit":{"terms":["intercept","static_store_sites","branch_sites","lds","store_sites_x_lds","real_stores"],"coefficients_ms":coef.tolist(),"r2":r2},"rows":[{"case_id":r["case_id"],"median_ms":r["median_ms"],"binary_sha256":r["binary_sha256"]} for r in rows]}
  (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return manifest
