#!/usr/bin/env python3
"""Exactness and isolated service sweep for normalized wide-Flash QG ownership."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel, flash_vec_llama_score_pv_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.uop.ops import UOp

HQ, HKV, HD, MAXC, TC, BOUND = 32, 8, 128, 1024, 641, 768
ARMS = {1:6, 2:12, 4:24}


def run(replays:int, reps:int) -> dict:
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  rng=np.random.default_rng(20260827)
  q=Tensor(rng.normal(0,0.3,(HQ,HD)).astype(np.float32),device=dev).contiguous().realize()
  cache=Tensor(rng.normal(0,0.3,(2,1,HKV,MAXC,HD)).astype(np.float16),device=dev).contiguous().realize()

  calls={}
  for qg,splits in ARMS.items():
    tile=KernelProgram("research.flash_wide_qgroup",f"qg{qg}.tile",KernelProgramProvenance.RESEARCH_ONLY,
      flash_vec_llama_score_pv_kernel(HD,HQ,HKV,MAXC,splits,UOp.const(dtypes.int,TC),wide_kv=True,wide_q=False,
        token_bound=BOUND,query_group_size=qg),
      output_spec=OutputSpec((HQ*splits*(HD+2),),dtypes.float32))
    combine=KernelProgram("research.flash_wide_qgroup",f"qg{qg}.combine",KernelProgramProvenance.RESEARCH_ONLY,
      flash_fused_gmax_combine_kernel(HD,HQ,splits,output_fp16=True,lane_width=128),
      output_spec=OutputSpec((HQ*HD,),dtypes.float16))
    def make_call(tile_program,combine_program):
      @TinyJit
      def call(q_arg,cache_arg):
        cache_view=Tensor(cache_arg.uop.bitcast(dtypes.uint32))
        partial=execute_research_program(None,q_arg.reshape(HQ*HD),cache_view,program=tile_program)
        return execute_research_program(None,partial,program=combine_program)
      return call
    calls[qg]=make_call(tile,combine)

  outputs={}
  for qg,call in calls.items(): call(q,cache).realize(); outputs[qg]=call(q,cache).realize().numpy()
  Device[dev].synchronize()
  base=outputs[1]
  exact={str(qg):{"bitwise_equal":bool(np.array_equal(base.view(np.uint16),out.view(np.uint16))),
    "mismatched_fp16_words":int(np.count_nonzero(base.view(np.uint16)!=out.view(np.uint16))),
    "max_abs_delta":float(np.max(np.abs(base.astype(np.float32)-out.astype(np.float32)))),
    "sha256":hashlib.sha256(out.tobytes()).hexdigest()} for qg,out in outputs.items()}
  if not all(np.isfinite(out).all() for out in outputs.values()):
    raise RuntimeError(f"finite-output gate failed: " +
      str({qg:int(np.count_nonzero(~np.isfinite(out))) for qg,out in outputs.items()}))

  for _ in range(300):
    for call in calls.values(): call(q,cache).realize()
  Device[dev].synchronize()
  samples={}
  for qg in (1,2,4,1):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize(); begin=time.perf_counter_ns()
      for _ in range(replays): calls[qg](q,cache).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-begin)/1e3/replays)
    samples.setdefault(str(qg),[]).append(vals)
  qg1=(statistics.median(samples["1"][0])+statistics.median(samples["1"][1]))/2
  med={"1":qg1,"2":statistics.median(samples["2"][0]),"4":statistics.median(samples["4"][0])}
  best=min((1,2,4),key=lambda x:med[str(x)])
  return {"schema":"tinygrad.nv_flash_wide_qgroup_microgate.v1","device":str(dev),
    "shape":{"hq":HQ,"hkv":HKV,"hd":HD,"max_context":MAXC,"tc":TC,"token_bound":BOUND},
    "geometry":{str(qg):{"split_count":s,"ctas":HQ//qg*s,"warps":HQ//qg*s*4} for qg,s in ARMS.items()},
    "exact":exact,"timing":{"unit":"us_per_tile_plus_combine_host_synchronized","replays":replays,"reps":reps,
      "samples":samples,"medians":med,"best_qg":best,"best_recovery_us":qg1-med[str(best)]},
    "verdict":"PRIMITIVE_PASS" if best != 1 and exact[str(best)]["bitwise_equal"] else "NO_GO_PRIMITIVE"}


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--replays",type=int,default=1000);ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args();result=run(args.replays,args.reps)
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
