#!/usr/bin/env python3
"""Conservative end-to-end gate/up gate for U4Z8.

Control is the installed fused Q4_K gate/up kernel. Candidate deliberately uses
two already-qualified U4Z8 projection kernels plus a separate SiLU*up kernel.
Passing this conservative construction is sufficient to justify a fused U4Z8
emitter; failing it does not reject fusion. No production path is changed.
"""
from __future__ import annotations

import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp
import nv_u4z8_g64_p256_ffn_projection_gate as proj

ROWS, K, CW, UW, ROT = proj.ROWS, proj.K, proj.CW, proj.UW, 16
CUDA_BIN = "/usr/local/cuda-13.2/bin"


def sources() -> tuple[str, str]:
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  p = UOp.placeholder
  ctrl = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=True)(
    p((ROWS,), dtypes.float16, 0), p((CW,), dtypes.uint32, 1), p((CW,), dtypes.uint32, 2), p((K,), dtypes.float16, 3))
  cs = next(x.arg for x in to_program(ctrl, ren).src if x.op is Ops.SOURCE)
  cs = cs[cs.index('extern "C" __global__'):]
  _, us = proj.render()
  return cs, us


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }
__CTRL__
__U4__
extern "C" __global__ void u4_gate_up_finish(half *o,const float *g,const float *u) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<12288){float x=g[i];o[i]=__float2half_rn((x/(1.0f+exp2f(x*(-1.4426950408889634f))))*u[i]);} }
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static void control(half*o,uint32_t*g,uint32_t*u,half*x,cudaStream_t s=0){q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288,32,0,s>>>(o,g,u,x);}
static void candidate(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,cudaStream_t s=0){u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_12288_4096<<<12288,32,0,s>>>(gf,g,x,z);u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_12288_4096<<<12288,32,0,s>>>(uf,u,x,z);u4_gate_up_finish<<<48,256,0,s>>>(o,gf,uf);}
static double tc(half*o,uint32_t*g,uint32_t*u,half*x,int n,cudaEvent_t a,cudaEvent_t b){cudaEventRecord(a);for(int i=0;i<n;i++)control(o,g,u,x);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
static double tu(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,int n,cudaEvent_t a,cudaEvent_t b){cudaEventRecord(a);for(int i=0;i<n;i++)candidate(o,gf,uf,g,u,x,z);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):32,reps=ac>2?atoi(av[2]):9;half *oc,*ou,*x;float *gf,*uf,*z;uint32_t *gc,*uc,*gu,*uu;ck(cudaMalloc(&oc,24576),"oc");ck(cudaMalloc(&ou,24576),"ou");ck(cudaMalloc(&gf,49152),"gf");ck(cudaMalloc(&uf,49152),"uf");ck(cudaMalloc(&z,49152),"z");ck(cudaMalloc(&x,8192),"x");ck(cudaMalloc(&gc,(size_t)ROT*CW*4),"gc");ck(cudaMalloc(&uc,(size_t)ROT*CW*4),"uc");ck(cudaMalloc(&gu,(size_t)ROT*UW*4),"gu");ck(cudaMalloc(&uu,(size_t)ROT*UW*4),"uu");cudaMemset(gc,0,(size_t)ROT*CW*4);cudaMemset(uc,0,(size_t)ROT*CW*4);cudaMemset(gu,0,(size_t)ROT*UW*4);cudaMemset(uu,0,(size_t)ROT*UW*4);cudaMemset(x,0,8192);cudaMemset(z,0,49152);control(oc,gc,uc,x);candidate(ou,gf,uf,gu,uu,x,z);ck(cudaDeviceSynchronize(),"warm");unsigned char hc[24576],hu[24576];cudaMemcpy(hc,oc,24576,cudaMemcpyDeviceToHost);cudaMemcpy(hu,ou,24576,cudaMemcpyDeviceToHost);int exact=!memcmp(hc,hu,24576);printf("zero_fixture_bitwise=%d\n",exact);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int r=0;r<reps;r++){double c=0,u=0;for(int i=0;i<n;i++){int j=i%ROT;c+=tc(oc,gc+(size_t)j*CW,uc+(size_t)j*CW,x,1,a,b);u+=tu(ou,gf,uf,gu+(size_t)j*UW,uu+(size_t)j*UW,x,z,1,a,b);}printf("rep=%d control_us=%.6f candidate_us=%.6f\n",r,c/n,u/n);}return exact?0:5;}
'''


def main() -> int:
  ap=argparse.ArgumentParser();ap.add_argument("--passes",type=int,default=32);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
  ctrl,u4=sources(); text=HARNESS.replace("__CTRL__",ctrl).replace("__U4__",u4)
  text=re.sub(r"\bROT\b",str(ROT),text);text=re.sub(r"\bCW\b",str(CW),text);text=re.sub(r"\bUW\b",str(UW),text)
  with tempfile.TemporaryDirectory(prefix="nv_u4_gate_up_chain_") as td:
    src,binp=Path(td)/"gate.cu",Path(td)/"gate";src.write_text(text);env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    b=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17",str(src),"-o",str(binp)],capture_output=True,text=True,env=env)
    if b.returncode:print(b.stderr[-10000:],file=sys.stderr);return 3
    run=subprocess.run([str(binp),str(a.passes),str(a.reps)],capture_output=True,text=True);print(run.stdout)
    if run.returncode not in (0,5):print(run.stderr[-5000:],file=sys.stderr);return 4
    exact="zero_fixture_bitwise=1" in run.stdout;c=[];u=[]
    for m in re.finditer(r"control_us=([0-9.]+) candidate_us=([0-9.]+)",run.stdout):c.append(float(m[1]));u.append(float(m[2]))
    cm,um=statistics.median(c),statistics.median(u);result={"schema":"tinygrad.nv_u4z8_gate_up_chain.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"construction":"installed fused Q4_K versus conservative two U4Z8 projections plus separate finish","semantic_scope":"zero-data composition smoke; each nonzero U4Z8 projection is independently oracle-qualified by the FFN projection gate","zero_fixture_bitwise":exact,"timing":{"unit":"us_per_pair_per_call_sync","control":c,"candidate":u,"control_median":cm,"candidate_median":um,"recovery_us":cm-um},"verdict":"PASS_PERFORMANCE_SEMANTIC_SMOKE" if exact and um<cm else "NEEDS_FUSED_EMITTER" if exact else "STOP_CORRECTNESS"};a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0 if exact else 5


if __name__=="__main__":raise SystemExit(main())
