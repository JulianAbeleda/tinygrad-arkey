#!/usr/bin/env python3
"""Primitive gate for streaming only one-touch Q6 payload while retaining cached metadata."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, time
os.environ.setdefault("NV_L2_STREAMING_WEIGHT_PROGRAMS",
  "q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd_splitstream@2")

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import Q6K_HALFWORDS_PER_BLOCK
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.llm.q6k_ffn_down_mmvq import K, ROWS, emit_q6k_four_warp_fp16_direct


def run(replays:int,reps:int)->dict:
  dev=Device.DEFAULT
  nwords=ROWS*(K//256)*Q6K_HALFWORDS_PER_BLOCK
  # Zero payload is enough for the exactness gate and preserves the full 39.4-MiB service footprint.
  w=Tensor.zeros(nwords,dtype=dtypes.uint16,device=dev).contiguous().realize()
  x=Tensor.ones(K,dtype=dtypes.float16,device=dev).contiguous().realize()
  h=Tensor.zeros(ROWS,dtype=dtypes.float32,device=dev).contiguous().realize()
  programs={
    "control":KernelProgram("research.q6_reuse_class","control",KernelProgramProvenance.RESEARCH_ONLY,
      emit_q6k_four_warp_fp16_direct(packed_lanemap=True,unroll_blocks=4),OutputSpec((ROWS,),dtypes.float32)),
    "payload_stream":KernelProgram("research.q6_reuse_class","payload_stream",KernelProgramProvenance.RESEARCH_ONLY,
      emit_q6k_four_warp_fp16_direct(packed_lanemap=True,unroll_blocks=4,split_weight_stream=True),OutputSpec((ROWS,),dtypes.float32))}
  @TinyJit
  def control(ww,xx,hh): return execute_research_program(None,ww,xx,hh,program=programs["control"])
  @TinyJit
  def candidate(ww,xx,hh): return execute_research_program(None,ww,ww,xx,hh,program=programs["payload_stream"])
  calls={"control":control,"payload_stream":candidate};outputs={}
  for name,call in calls.items(): call(w,x,h).realize();outputs[name]=call(w,x,h).realize().numpy()
  Device[dev].synchronize()
  exact=bool(np.array_equal(outputs["control"].view(np.uint32),outputs["payload_stream"].view(np.uint32)))
  for _ in range(100):
    for call in calls.values():call(w,x,h).realize()
  Device[dev].synchronize();samples={x:[] for x in calls}
  for name in ("control","payload_stream","payload_stream","control"):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize();begin=time.perf_counter_ns()
      for _ in range(replays):calls[name](w,x,h).realize()
      Device[dev].synchronize();vals.append((time.perf_counter_ns()-begin)/1e3/replays)
    samples[name].append(vals)
  med={k:statistics.median([statistics.median(x) for x in v]) for k,v in samples.items()}
  recovery=med["control"]-med["payload_stream"]
  return {"schema":"tinygrad.nv_q6_ffn_down_reuse_class_microgate.v1","device":str(dev),
    "shape":{"rows":ROWS,"k":K,"weight_halfwords":nwords},
    "policy":{"ordinary_pointer":"scale and d metadata","evict_first_alias":"ql/qh quantized payload only"},
    "exact":{"bitwise_equal":exact,"sha256":{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in outputs.items()}},
    "timing":{"unit":"us_per_graph_replay_host_synchronized","replays":replays,"reps":reps,"samples":samples,
      "medians":med,"candidate_recovery_us":recovery},
    "verdict":"PRIMITIVE_PASS" if exact and recovery>0 else "NO_GO_PRIMITIVE"}


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--replays",type=int,default=1000);ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args();result=run(a.replays,a.reps)
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
