#!/usr/bin/env python3
"""Causal primitive gate for actual cross-query-head K/V sharing in wide Flash."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel, flash_vec_llama_score_pv_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.uop.ops import UOp

HQ, HKV, HD, MAXC, TC, BOUND = 32, 8, 128, 1024, 641, 768
ARMS = {"qg1_control":(1, 6, False), "qg2_control":(2, 12, False), "qg2_shared":(2, 12, True),
        "qg4_control":(4, 24, False), "qg4_shared":(4, 24, True)}


def run(replays:int, reps:int) -> dict:
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  rng=np.random.default_rng(20260827)
  q=Tensor(rng.normal(0,0.3,(HQ,HD)).astype(np.float32),device=dev).contiguous().realize()
  cache=Tensor(rng.normal(0,0.3,(2,1,HKV,MAXC,HD)).astype(np.float16),device=dev).contiguous().realize()
  calls={}
  for name,(qg,splits,shared) in ARMS.items():
    tile=KernelProgram("research.flash_cross_head_kv_share",f"{name}.tile",KernelProgramProvenance.RESEARCH_ONLY,
      flash_vec_llama_score_pv_kernel(HD,HQ,HKV,MAXC,splits,UOp.const(dtypes.int,TC),wide_kv=True,wide_q=False,
        token_bound=BOUND,query_group_size=qg,share_kv_across_query_heads=shared),
      output_spec=OutputSpec((HQ*splits*(HD+2),),dtypes.float32))
    combine=KernelProgram("research.flash_cross_head_kv_share",f"{name}.combine",KernelProgramProvenance.RESEARCH_ONLY,
      flash_fused_gmax_combine_kernel(HD,HQ,splits,output_fp16=True,lane_width=128),
      output_spec=OutputSpec((HQ*HD,),dtypes.float16))
    def make_call(tile_program,combine_program):
      @TinyJit
      def call(q_arg,cache_arg):
        partial=execute_research_program(None,q_arg.reshape(HQ*HD),Tensor(cache_arg.uop.bitcast(dtypes.uint32)),program=tile_program)
        return execute_research_program(None,partial,program=combine_program)
      return call
    calls[name]=make_call(tile,combine)

  outputs={}
  for name,call in calls.items(): call(q,cache).realize();outputs[name]=call(q,cache).realize().numpy()
  Device[dev].synchronize();base=outputs["qg1_control"]
  exact={name:{"bitwise_equal_to_qg1":bool(np.array_equal(base.view(np.uint16),out.view(np.uint16))),
    "mismatched_fp16_words":int(np.count_nonzero(base.view(np.uint16)!=out.view(np.uint16))),
    "max_abs_delta":float(np.max(np.abs(base.astype(np.float32)-out.astype(np.float32)))),
    "sha256":hashlib.sha256(out.tobytes()).hexdigest()} for name,out in outputs.items()}
  if not all(np.isfinite(x).all() for x in outputs.values()): raise RuntimeError("finite-output gate failed")

  for _ in range(200):
    for call in calls.values(): call(q,cache).realize()
  Device[dev].synchronize()
  order=("qg2_control","qg2_shared","qg4_control","qg4_shared","qg4_shared","qg4_control","qg2_shared","qg2_control")
  samples={name:[] for name in ARMS}
  for name in order:
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize();begin=time.perf_counter_ns()
      for _ in range(replays): calls[name](q,cache).realize()
      Device[dev].synchronize();vals.append((time.perf_counter_ns()-begin)/1e3/replays)
    samples[name].append(vals)
  med={name:statistics.median([statistics.median(x) for x in rows]) for name,rows in samples.items() if rows}
  recovery={f"qg{qg}":med[f"qg{qg}_control"]-med[f"qg{qg}_shared"] for qg in (2,4)}
  passing=[qg for qg in (2,4) if exact[f"qg{qg}_shared"]["bitwise_equal_to_qg1"] and recovery[f"qg{qg}"] > 0]
  return {"schema":"tinygrad.nv_flash_cross_head_kv_share_microgate.v1","device":str(dev),
    "shape":{"hq":HQ,"hkv":HKV,"hd":HD,"max_context":MAXC,"tc":TC,"token_bound":BOUND},
    "geometry":{name:{"query_group_size":qg,"split_count":s,"ctas":HQ//qg*s,"warps":HQ//qg*s*4,
      "global_kv_request_fraction_vs_control":1/qg if shared else 1.0} for name,(qg,s,shared) in ARMS.items()},
    "exact":exact,"timing":{"unit":"us_per_tile_plus_combine_host_synchronized","replays":replays,"reps":reps,
      "samples":samples,"medians":med,"shared_recovery_us":recovery},
    "verdict":"PRIMITIVE_PASS" if passing else "NO_GO_PRIMITIVE","passing_query_groups":passing}


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--replays",type=int,default=1000);ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args();result=run(args.replays,args.reps)
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__": raise SystemExit(main())
