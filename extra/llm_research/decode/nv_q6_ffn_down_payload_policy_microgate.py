#!/usr/bin/env python3
"""ABI-preserving primitive gate for streaming only Q6 ql/qh payload."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import Q6K_HALFWORDS_PER_BLOCK
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.llm.q6k_ffn_down_mmvq import K, ROWS, emit_q6k_four_warp_fp16_direct

PROGRAM="q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd_payloadpolicy"

def run(replays:int,reps:int)->dict:
  dev=Device.DEFAULT;nwords=ROWS*(K//256)*Q6K_HALFWORDS_PER_BLOCK
  w=Tensor.zeros(nwords,dtype=dtypes.uint16,device=dev).contiguous().realize()
  x=Tensor.ones(K,dtype=dtypes.float16,device=dev).contiguous().realize()
  h=Tensor.zeros(ROWS,dtype=dtypes.float32,device=dev).contiguous().realize()
  control_program=KernelProgram("research.q6_payload_policy","control",KernelProgramProvenance.RESEARCH_ONLY,
    emit_q6k_four_warp_fp16_direct(packed_lanemap=True,unroll_blocks=4),OutputSpec((ROWS,),dtypes.float32))
  candidate_program=KernelProgram("research.q6_payload_policy","candidate",KernelProgramProvenance.RESEARCH_ONLY,
    emit_q6k_four_warp_fp16_direct(packed_lanemap=True,unroll_blocks=4,research_name_suffix="_payloadpolicy"),OutputSpec((ROWS,),dtypes.float32))
  @TinyJit
  def control(ww,xx,hh):return execute_research_program(None,ww,xx,hh,program=control_program)
  os.environ.pop("NV_L2_STREAMING_Q6_PAYLOAD_PROGRAMS",None)
  control(w,x,h).realize();out0=control(w,x,h).realize().numpy()
  os.environ["NV_L2_STREAMING_Q6_PAYLOAD_PROGRAMS"]=PROGRAM
  @TinyJit
  def candidate(ww,xx,hh):return execute_research_program(None,ww,xx,hh,program=candidate_program)
  candidate(w,x,h).realize();out1=candidate(w,x,h).realize().numpy()
  calls={"control":control,"payload_stream":candidate}
  Device[dev].synchronize()
  for _ in range(100):
    for call in calls.values():call(w,x,h).realize()
  Device[dev].synchronize();samples={k:[] for k in calls}
  for name in ("control","payload_stream","payload_stream","control"):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize();begin=time.perf_counter_ns()
      for _ in range(replays):calls[name](w,x,h).realize()
      Device[dev].synchronize();vals.append((time.perf_counter_ns()-begin)/1e3/replays)
    samples[name].append(vals)
  med={k:statistics.median([statistics.median(x) for x in v]) for k,v in samples.items()};recovery=med["control"]-med["payload_stream"]
  exact=bool(np.array_equal(out0.view(np.uint32),out1.view(np.uint32)))
  return {"schema":"tinygrad.nv_q6_ffn_down_payload_policy_microgate.v1","device":str(dev),
    "shape":{"rows":ROWS,"k":K,"weight_halfwords":nwords},"policy":{"abi":"unchanged","evict_first":"ql/qh payload only","retained":"scales and d metadata"},
    "exact":{"bitwise_equal":exact,"sha256":{"control":hashlib.sha256(out0.tobytes()).hexdigest(),"candidate":hashlib.sha256(out1.tobytes()).hexdigest()}},
    "timing":{"unit":"us_per_graph_replay_host_synchronized","replays":replays,"reps":reps,"samples":samples,"medians":med,"candidate_recovery_us":recovery},
    "verdict":"PRIMITIVE_PASS" if exact and recovery>0 else "NO_GO_PRIMITIVE"}

def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--replays",type=int,default=1000);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  result=run(a.replays,a.reps);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
