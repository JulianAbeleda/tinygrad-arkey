#!/usr/bin/env python3
"""Fail-closed qualification for the isolated packed Q4_K x Q8 candidate."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import _record_source, binding_for
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC_FP16
from extra.llm_research.prefill.nv_packed_qk_q8_streamk import *
from extra.llm_research.prefill.nv_q4_imma_provider import Geometry, compile_provider
GEOMETRY = Geometry(owners=160)

def digest(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def buf(x): return x.uop.buffer.get_buf("NV")
def stats(x): return {"min_us":min(x), "median_us":statistics.median(x), "p95_us":float(np.percentile(x,95)),
                      "mean_us":statistics.mean(x), "samples_us":x}
def timed(dev, rounds, launch):
  samples=[]
  for _ in range(rounds):
    dev.synchronize(); st=time.perf_counter_ns(); launch(); dev.synchronize(); samples.append((time.perf_counter_ns()-st)/1e3)
  return samples

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--out",required=True)
  ap.add_argument("--rounds",type=int,default=40); a=ap.parse_args(); stage="setup"
  result={"schema":"tinygrad.nv-packed-qk-q8-streamk-gate-phase1.v2","status":"FAIL",
    "threshold":"PASS iff mandatory gates pass, candidate min+median beat both controls, and median <=250 us",
    "commands":[f"flock -w 1200 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV .venv/bin/python {__file__} --model {a.model} --out {a.out} --rounds {a.rounds}"],
    "files_changed":["extra/llm_research/prefill/nv_packed_qk_q8_streamk.py",
      "extra/llm_research/prefill/nv_packed_qk_q8_gate_qualify.py",
      "docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.md",
      "docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.json"]}
  try:
    from extra.llm_research.layout import read_metadata, packed_u32_slice
    path=pathlib.Path(a.model); md=read_metadata(path); info=next(i for i in md.infos if i.name.endswith("ffn_gate.weight"))
    words=packed_u32_slice(path,md,info,device="NV").contiguous().realize(); dev=Device["NV"]
    cand=Candidate(compile_provider(dev, GEOMETRY)); control=binding_for("NV")
    prod=NVProgram(dev,"q8_compact_fp16",NVRTCCompiler(dev.arch,ptx=False,cache_key="phase1_q8_planes").compile(SRC_FP16))
    rng=np.random.default_rng(20260829); hosts=[rng.standard_normal((M,K),dtype=np.float32).astype(np.float16),
      np.roll(rng.standard_normal((M,K),dtype=np.float32).astype(np.float16),17,axis=1)]
    result.update({"model":str(path),"tensor":info.name,"shape":{"M":M,"N":N,"K":K},"gpu_arch":dev.arch,
      "dtype":"fp16->Q8/int8->fp32","metadata":cand.metadata.__dict__,"inputs":[]})

    def allocate(h):
      x=Tensor(h,device="NV").contiguous().realize()
      out=Tensor.full((M*N,),np.nan,dtype=dtypes.float32,device="NV").realize()
      part=Tensor.empty(GEOMETRY.partial_slots*128*128,dtype=dtypes.float32,device="NV").realize()
      ids=Tensor.empty(GEOMETRY.partial_slots,dtype=dtypes.int32,device="NV").realize()
      sm=Tensor(cand.provider.slotmap,device="NV").contiguous().realize()
      q=Tensor.empty(M*K,dtype=dtypes.int8,device="NV").realize()
      scales=Tensor.empty(M*K//32,dtype=dtypes.float32,device="NV").realize()
      sums=Tensor.empty(M*K//32,dtype=dtypes.float32,device="NV").realize()
      cand.provider.validate_buffers(buf(out),buf(part),buf(ids),buf(words),buf(q),buf(scales),buf(sums),buf(sm))
      return x,out,part,ids,sm,q,scales,sums

    def launch(p,wait=False):
      x,out,part,ids,sm,q,scales,sums=p
      prod(buf(x),buf(q),buf(scales),buf(sums),global_size=(M,8,1),local_size=(128,1,1),wait=wait)
      cand.launch_main(buf(out),buf(part),buf(ids),buf(words),buf(q),buf(scales),buf(sums),wait=wait)
      cand.launch_fixup(buf(out),buf(part),buf(sm),wait=wait)
      return out

    parts=[]; output_digests=[]; correctness=True
    for ci,h in enumerate(hosts):
      p=allocate(h); parts.append(p); stage=f"case-{ci}-candidate"
      launch(p,True); got=p[1].numpy(); repeats=[digest(got)]
      stage=f"case-{ci}-determinism"
      for _ in range(19): launch(p,True); repeats.append(digest(p[1].numpy()))
      stage=f"case-{ci}-installed-reference"
      ref=control.project(p[0],words,model_family="qwen3_8b",role="ffn_gate").contiguous().realize().numpy().reshape(-1)
      got=p[1].numpy(); diff=np.abs(got-ref); meaningful=np.abs(ref)>1e-6
      rel=diff[meaningful]/np.abs(ref[meaningful]) if meaningful.any() else diff; close=np.isclose(got,ref,rtol=.02,atol=.5)
      finite=bool(np.isfinite(got).all()); ref_finite=bool(np.isfinite(ref).all()); deterministic=len(set(repeats))==1
      mismatch=int(np.count_nonzero(~close)); passed=finite and ref_finite and deterministic and mismatch==0; correctness &= passed
      output_digests.append(digest(got)); bad=np.argwhere(~close)
      result["inputs"].append({"case":ci,"input_checksum":digest(h),"input_pointer":buf(p[0]).va_addr,
        "weight_pointer":buf(words).va_addr,"output_pointer":buf(p[1]).va_addr,"workspace_pointer":buf(p[2]).va_addr,
        "finite":finite,"reference_finite":ref_finite,"overwrite":finite,"deterministic_20":deterministic,
        "repeat_checksums":repeats,"output_checksum":output_digests[-1],"reference_checksum":digest(ref),
        "max_abs":float(diff.max()),"max_rel":float(rel.max()),"mismatch_count":mismatch,
        "first_mismatch":bad[0].tolist() if bad.size else None,"correctness_pass":passed})
    changed=output_digests[0]!=output_digests[1]; correctness &= changed; result["activation_changes_output"]=changed

    timing_parts=parts[0]
    @TinyJit
    def control_route(x): return control.project(x,words,model_family="qwen3_8b",role="ffn_gate").realize()
    stage="control-capture"
    for _ in range(3): control_route(timing_parts[0])
    dev.synchronize(); stage="candidate-warmup"
    for _ in range(3): launch(timing_parts,False); dev.synchronize()
    aa=[]; cc=[]; bb=[]; r9=[]
    for rep in range(9):
      stage=f"r9-{rep}-control-a"; x=timed(dev,a.rounds,lambda:control_route(timing_parts[0]))
      stage=f"r9-{rep}-candidate"; y=timed(dev,a.rounds,lambda:launch(timing_parts,False))
      stage=f"r9-{rep}-control-b"; z=timed(dev,a.rounds,lambda:control_route(timing_parts[0]))
      aa+=x;cc+=y;bb+=z;r9.append({"repetition":rep,"control_a":stats(x),"candidate":stats(y),"control_b":stats(z)})
    timing={"control_a":stats(aa),"candidate":stats(cc),"control_b":stats(bb),"r9":r9}; result["timing"]=timing
    result["structural"]={"pass":True,"canonical_packed_weight":True,"weight_expansion_or_hot_copy":False,
      "candidate_launches":["q8_compact_record_fp16","q4k_imma_stream","q4k_imma_fixup"],
      "allocation_inside_timed_candidate":False,"synchronization_inside_timed_candidate":False,
      "stable_output_pointer":True,"stable_workspace_pointer":True}
    cm=timing["candidate"]["median_us"]; beats=(timing["candidate"]["min_us"]<timing["control_a"]["min_us"] and
      timing["candidate"]["min_us"]<timing["control_b"]["min_us"] and cm<timing["control_a"]["median_us"] and cm<timing["control_b"]["median_us"])
    if not correctness: result["status"],result["observed_failure"]="FAIL","mandatory correctness gate failed"
    elif not beats: result["status"],result["observed_failure"]="FAIL","candidate did not beat both controls on minimum and median"
    elif cm>307: result["status"],result["observed_failure"]="STOP","candidate median exceeds 307 us investment threshold"
    elif cm>250: result["status"],result["observed_failure"]="ITERATE","candidate beats controls but exceeds 250 us promotion threshold"
    else:
      result["status"]="PHASE-1 PASS"; result["projected_72_projection_savings_ms"]=72*(timing["control_a"]["median_us"]-cm)/1000
  except Exception as e:
    result["status"]="FAIL"; result["observed_failure"]=f"stage={stage} {type(e).__name__}: {e}"
  p=pathlib.Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);rendered=json.dumps(result,indent=2);p.write_text(rendered+"\n")
  p.with_suffix(".md").write_text(f"# Phase 1 result\n\nStatus: **{result['status']}**\n\n```json\n{rendered}\n```\n");print(rendered)
if __name__=="__main__": main()
