#!/usr/bin/env python3
"""Matched native-CUDA timing for llama and tinygrad six-part Flash combine."""
from __future__ import annotations
import argparse, contextlib, json, pathlib, re, statistics, subprocess, tempfile
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

ROOT=pathlib.Path(__file__).resolve().parents[3]; LLAMA=pathlib.Path("/home/ubuntu/env/llama.cpp")
NVCC="/usr/local/cuda-13.2/bin/nvcc"; H,Hd,S,W=32,128,6,130
COST_PREDICTION={"metric":"tiny_register_minus_llama_us_per_launch","range_us":[-0.20,0.20],
  "premise":"matched six-part native combines should be launch-floor peers; this probe books no token-wall delta"}

def validate_cost_prediction(result:dict)->dict:
  measured=result["tiny_minus_llama_us"]["tiny_register"]
  lo,hi=COST_PREDICTION["range_us"]
  return {"contract":COST_PREDICTION,"measured_us":measured,"status":"CONFIRMED" if lo<=measured<=hi else "CONTRADICTED"}

def render(register:bool)->tuple[str,str]:
  out=UOp.placeholder((H*Hd,),dtypes.float16,0); part=UOp.placeholder((H*S*W,),dtypes.float32,1)
  ast=flash_fused_gmax_combine_kernel(Hd,H,S,output_fp16=True,lane_width=128,register_weights=register)(out,part)
  prg=to_program(ast,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False))
  src=next(x.arg for x in prg.src if x.op is Ops.SOURCE)
  return prg.arg.name,src

def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--replays",type=int,default=5000);ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--artifacts-dir",type=pathlib.Path)
  ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  shared,ss=render(False);reg,rs=render(True)
  # Strip duplicate preamble from the second generated source.
  rs=rs[rs.index('extern "C"'):]
  harness=f'''
#include "ggml-cuda/fattn-common.cuh"
{ss}
{rs}
#include <cstdio>
int main() {{
  constexpr int H={H},D={Hd},S={S},W={W}; float *tp,*lp,*lm,*lo; half *to; cudaEvent_t a,b;
  cudaMalloc(&tp,H*S*W*4);cudaMalloc(&to,H*D*2);cudaMalloc(&lp,H*S*D*4);cudaMalloc(&lm,H*S*8);cudaMalloc(&lo,H*D*4);
  cudaMemset(tp,1,H*S*W*4);cudaMemset(lp,1,H*S*D*4);cudaMemset(lm,1,H*S*8);cudaEventCreate(&a);cudaEventCreate(&b);
  auto one=[&](int arm,int n){{cudaEventRecord(a);for(int i=0;i<n;i++){{
    if(arm==0) {shared}<<<H,128>>>(to,tp); else if(arm==1) {reg}<<<H,128>>>(to,tp);
    else flash_attn_combine_results<D><<<dim3(1,H,1),128,S*sizeof(float2)>>>(lp,(float2*)lm,lo,S);
  }}cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0f*ms/n;}};
  for(int arm=0;arm<3;arm++)for(int i=0;i<200;i++)one(arm,1);
  for(int r=0;r<{a.reps};r++)printf("rep=%d tiny_shared=%.6f tiny_register=%.6f llama=%.6f\\n",r,one(0,{a.replays}),one(1,{a.replays}),one(2,{a.replays}));
}}
'''
  a.out.parent.mkdir(parents=True,exist_ok=True)
  if a.artifacts_dir:a.artifacts_dir.mkdir(parents=True,exist_ok=True)
  work=contextlib.nullcontext(str(a.artifacts_dir)) if a.artifacts_dir else tempfile.TemporaryDirectory(prefix="nv_combine_ab_")
  with work as td:
    cu=pathlib.Path(td)/"ab.cu";binary=pathlib.Path(td)/"ab";cu.write_text(harness)
    cmd=[NVCC,"-O3","-std=c++17","-arch=sm_120a","--ptxas-options=-v","-I",str(LLAMA/"ggml/src"),"-I",str(LLAMA/"ggml/include"),"-I",str(LLAMA/"ggml/src/ggml-cuda"),str(cu),"-o",str(binary)]
    build=subprocess.run(cmd,text=True,capture_output=True)
    if build.returncode: raise RuntimeError(build.stderr[-12000:])
    run=subprocess.run([str(binary)],text=True,capture_output=True,check=True)
  rows=[]
  for line in run.stdout.splitlines():
    if m:=re.match(r"rep=(\d+) tiny_shared=([0-9.]+) tiny_register=([0-9.]+) llama=([0-9.]+)",line):
      rows.append({"tiny_shared":float(m[2]),"tiny_register":float(m[3]),"llama":float(m[4])})
  med={k:statistics.median(x[k] for x in rows) for k in ("tiny_shared","tiny_register","llama")}
  result={"schema":"tinygrad.nv_flash_combine_native_ab.v1","replays":a.replays,"reps":a.reps,"samples":rows,"medians_us":med,
    "tiny_minus_llama_us":{k:med[k]-med["llama"] for k in ("tiny_shared","tiny_register")},"build_log":build.stderr}
  result["cost_gate"]=validate_cost_prediction(result)
  a.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
