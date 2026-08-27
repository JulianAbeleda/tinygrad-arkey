#!/usr/bin/env python3
"""Bit-exact partial ABI and isolated service gate for V live-range schedules."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, time
import numpy as np

from tinygrad import Context, Device, Tensor, TinyJit, dtypes
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel, flash_vec_llama_score_pv_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.uop.ops import UOp

HQ, HKV, HD, MAXC, TC, BOUND, SPLITS = 32, 8, 128, 1024, 641, 768, 6
ARMS = {"control":(0,False), "vtail1":(1,False), "vtail2":(2,False), "vtail4":(4,False),
        "vdimmajor":(8,True)}


def run(replays:int,reps:int,cubin_dir:pathlib.Path)->dict:
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"):raise RuntimeError(f"native NV required, got {dev}")
  schedule_context=Context(NV_FLASH_LOAD_SCHEDULE=1);schedule_context.__enter__()
  import tinygrad.runtime.ops_nv as ops_nv
  cubins={};orig_init=ops_nv.NVProgram.__init__
  def patched_init(self,device,name,lib,**kwargs):
    orig_init(self,device,name,lib,**kwargs)
    if name.startswith("flash_vec_llama_score_pv_32_128_6_widekv16"):
      cubin_dir.mkdir(parents=True,exist_ok=True);path=cubin_dir/f"{name}.cubin";path.write_bytes(bytes(lib))
      cubins[name]={"path":str(path),"sha256":hashlib.sha256(bytes(lib)).hexdigest(),
        "regs_usage":getattr(self,"regs_usage",None),"shmem_usage":getattr(self,"shmem_usage",None)}
  ops_nv.NVProgram.__init__=patched_init
  try:
    rng=np.random.default_rng(20260827)
    q=Tensor(rng.normal(0,0.3,(HQ,HD)).astype(np.float32),device=dev).contiguous().realize()
    cache=Tensor(rng.normal(0,0.3,(2,1,HKV,MAXC,HD)).astype(np.float16),device=dev).contiguous().realize()
    tile_programs={};combine=KernelProgram("research.flash_v_schedule","combine",KernelProgramProvenance.RESEARCH_ONLY,
      flash_fused_gmax_combine_kernel(HD,HQ,SPLITS,output_fp16=True,lane_width=128),
      output_spec=OutputSpec((HQ*HD,),dtypes.float16))
    calls={};partials={};outputs={}
    for name,(tail,dimmajor) in ARMS.items():
      tile=KernelProgram("research.flash_v_schedule",f"{name}.tile",KernelProgramProvenance.RESEARCH_ONLY,
        flash_vec_llama_score_pv_kernel(HD,HQ,HKV,MAXC,SPLITS,UOp.const(dtypes.int,TC),wide_kv=True,wide_q=False,
          token_bound=BOUND,v_pipeline_tail=tail,v_dimension_major=dimmajor),
        output_spec=OutputSpec((HQ*SPLITS*(HD+2),),dtypes.float32))
      tile_programs[name]=tile
      def make_call(tile_program):
        @TinyJit
        def call(q_arg,cache_arg):
          partial=execute_research_program(None,q_arg.reshape(HQ*HD),Tensor(cache_arg.uop.bitcast(dtypes.uint32)),program=tile_program)
          return execute_research_program(None,partial,program=combine)
        return call
      calls[name]=make_call(tile)
    cache_view=Tensor(cache.uop.bitcast(dtypes.uint32))
    for name,tile in tile_programs.items():
      partials[name]=execute_research_program(None,q.reshape(HQ*HD),cache_view,program=tile).realize().numpy()
      calls[name](q,cache).realize();outputs[name]=calls[name](q,cache).realize().numpy()
    Device[dev].synchronize();pbase=partials["control"];obase=outputs["control"]
    exact={name:{"partial_fp32_bitwise_equal":bool(np.array_equal(pbase.view(np.uint32),p.view(np.uint32))),
      "partial_mismatched_words":int(np.count_nonzero(pbase.view(np.uint32)!=p.view(np.uint32))),
      "final_fp16_bitwise_equal":bool(np.array_equal(obase.view(np.uint16),outputs[name].view(np.uint16))),
      "max_final_abs_delta":float(np.max(np.abs(obase.astype(np.float32)-outputs[name].astype(np.float32))))}
      for name,p in partials.items()}
    for _ in range(200):
      for call in calls.values():call(q,cache).realize()
    Device[dev].synchronize();samples={x:[] for x in calls}
    order=("control","vtail1","vtail2","vtail4","vdimmajor","vdimmajor","vtail4","vtail2","vtail1","control")
    for name in order:
      vals=[]
      for _ in range(reps):
        Device[dev].synchronize();begin=time.perf_counter_ns()
        for _ in range(replays):calls[name](q,cache).realize()
        Device[dev].synchronize();vals.append((time.perf_counter_ns()-begin)/1e3/replays)
      samples[name].append(vals)
    med={k:statistics.median([statistics.median(x) for x in v]) for k,v in samples.items()}
    return {"schema":"tinygrad.nv_flash_v_schedule_microgate.v1","device":str(dev),
      "shape":{"hq":HQ,"hkv":HKV,"hd":HD,"tc":TC,"token_bound":BOUND,"splits":SPLITS},
      "exact":exact,"cubins":cubins,"timing":{"unit":"us_per_tile_plus_combine_host_synchronized",
        "replays":replays,"reps":reps,"samples":samples,"medians":med}}
  finally:
    ops_nv.NVProgram.__init__=orig_init
    schedule_context.__exit__(None,None,None)


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--replays",type=int,default=1000);ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  result=run(a.replays,a.reps,pathlib.Path(str(a.out).removesuffix(".json"))/"cubins")
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
