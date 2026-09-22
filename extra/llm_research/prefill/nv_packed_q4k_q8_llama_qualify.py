#!/usr/bin/env python3
"""Correctness and service discriminator for the pinned llama Q4_K extraction."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for
from extra.llm_research.prefill.nv_packed_q4k_q8_llama_candidate import *

def buf(x): return x.uop.buffer.get_buf("NV")
def digest(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def stats(x): return {"min_us":min(x),"median_us":statistics.median(x),"p95_us":float(np.percentile(x,95)),"mean_us":statistics.mean(x),"samples_us":x}
def timed(dev,n,fn):
  out=[]
  for _ in range(n): dev.synchronize();s=time.perf_counter_ns();fn();dev.synchronize();out.append((time.perf_counter_ns()-s)/1e3)
  return out

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--out",required=True);ap.add_argument("--rounds",type=int,default=40);a=ap.parse_args()
  result={"schema":"tinygrad.nv-packed-q4k-q8-llama-extracted.v1","status":"FAIL","stage":"setup"}
  try:
    from extra.llm_research.layout import read_metadata,packed_u32_slice
    path=pathlib.Path(a.model);md=read_metadata(path);info=next(i for i in md.infos if i.name.endswith("ffn_gate.weight"));dev=Device["NV"]
    words=packed_u32_slice(path,md,info,device="NV").contiguous().realize();candidate=compile_candidate(dev);control=binding_for("NV")
    rng=np.random.default_rng(20260830);host16=rng.standard_normal((M,K),dtype=np.float32).astype(np.float16);host32=host16.astype(np.float32)
    x16=Tensor(host16,device="NV").contiguous().realize();x32=Tensor(host32,device="NV").contiguous().realize()
    record=Tensor.empty(Q8_RECORD_BYTES,dtype=dtypes.uint8,device="NV").realize();out=Tensor.full(M*N,np.nan,dtype=dtypes.float32,device="NV").realize()
    scratch=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device="NV").realize()
    result.update({"model":str(path),"tensor":info.name,"shape":{"M":M,"N":N,"K":K},"metadata":candidate.metadata.__dict__,
      "pointers":{"weight":buf(words).va_addr,"input_f32":buf(x32).va_addr,"record":buf(record).va_addr,"output":buf(out).va_addr,"scratch":buf(scratch).va_addr}})
    result["stage"]="candidate-producer";candidate.launch_producer(buf(x32),buf(record),wait=True)
    result["stage"]="candidate-main";candidate.launch_main(buf(words),buf(record),buf(out),buf(scratch),wait=True)
    result["stage"]="candidate-fixup";candidate.launch_fixup(buf(out),buf(scratch),wait=True);got=out.numpy()
    result["stage"]="installed-reference";ref=control.project(x16,words,model_family="qwen3_8b",role="ffn_gate").contiguous().realize().numpy().reshape(-1)
    diff=np.abs(got-ref);close=np.isclose(got,ref,rtol=.02,atol=.5);result["correctness"]={"finite":bool(np.isfinite(got).all()),
      "reference_finite":bool(np.isfinite(ref).all()),"unwritten":int(np.isnan(got).sum()),"max_abs":float(diff.max()),
      "mean_abs":float(diff.mean()),"mismatch_count":int(np.count_nonzero(~close)),"output_checksum":digest(got),"reference_checksum":digest(ref)}
    repeats=[];result["stage"]="candidate-determinism"
    for _ in range(20):candidate.launch(buf(x32),buf(words),buf(record),buf(out),buf(scratch),wait=True);repeats.append(digest(out.numpy()))
    result["correctness"]["deterministic_20"]=len(set(repeats))==1
    @TinyJit
    def control_route(x):return control.project(x,words,model_family="qwen3_8b",role="ffn_gate").realize()
    for _ in range(3):control_route(x16)
    for _ in range(3):candidate.launch(buf(x32),buf(words),buf(record),buf(out),buf(scratch),wait=False);dev.synchronize()
    aa=[];cc=[];bb=[];r9=[]
    for rep in range(9):
      result["stage"]=f"r9-{rep}-control-a";ca=timed(dev,a.rounds,lambda:control_route(x16))
      result["stage"]=f"r9-{rep}-candidate";cn=timed(dev,a.rounds,lambda:candidate.launch(buf(x32),buf(words),buf(record),buf(out),buf(scratch),wait=False))
      result["stage"]=f"r9-{rep}-control-b";cb=timed(dev,a.rounds,lambda:control_route(x16));aa+=ca;cc+=cn;bb+=cb
      r9.append({"repetition":rep,"control_a":stats(ca),"candidate":stats(cn),"control_b":stats(cb)})
    result["timing"]={"control_a":stats(aa),"candidate":stats(cc),"control_b":stats(bb),"r9":r9}
    correct=all((result["correctness"]["finite"],result["correctness"]["reference_finite"],result["correctness"]["unwritten"]==0,
      result["correctness"]["mismatch_count"]==0,result["correctness"]["deterministic_20"]))
    cm=result["timing"]["candidate"]["median_us"];beats=cm<result["timing"]["control_a"]["median_us"] and cm<result["timing"]["control_b"]["median_us"]
    result["status"]="PHASE-1 PASS" if correct and beats and cm<=250 else "ITERATE" if correct and beats else "FAIL"
    result["stage"]="complete"
  except Exception as e: result["observed_failure"]=f"stage={result['stage']} {type(e).__name__}: {e}"
  p=pathlib.Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
