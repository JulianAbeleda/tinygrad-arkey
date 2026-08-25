#!/usr/bin/env python3
"""Bit-exact complete-span gate for the ordinary full-grid Q4/Q4/Q4 producer."""
from __future__ import annotations

import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.q4k_kv_pair import emit_q4k_kv_pair_vector, emit_q4k_qkv_full
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp

Q_ROWS,KV_ROWS,K=4096,1024,4096
Q_WORDS=Q_ROWS*(K//256)*36
KV_WORDS=KV_ROWS*(K//256)*36
CUDA_BIN="/usr/local/cuda-13.2/bin"


def _render() -> tuple[str,str,str]:
  p=UOp.placeholder
  q=q4k_g3_lanemap_gemv_kernel(Q_ROWS,K,load_style="vector")(
    p((Q_ROWS,),dtypes.float32,0),p((Q_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2))
  pair=emit_q4k_kv_pair_vector()(p((KV_ROWS,),dtypes.float32,0),p((KV_ROWS,),dtypes.float32,1),
    p((KV_WORDS,),dtypes.uint32,2),p((KV_WORDS,),dtypes.uint32,3),p((K,),dtypes.float16,4))
  full=emit_q4k_qkv_full()(p((Q_ROWS,),dtypes.float32,0),p((KV_ROWS,),dtypes.float32,1),
    p((KV_ROWS,),dtypes.float32,2),p((Q_WORDS,),dtypes.uint32,3),p((KV_WORDS*2,),dtypes.uint32,4),
    p((K,),dtypes.float16,5))
  ren=CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False)
  def src(u:UOp) -> str:
    text=next(x.arg for x in to_program(u,ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]
  return src(q),src(pair),src(full)


HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define Q_ROWS 4096
#define KV_ROWS 1024
#define K 4096
#define Q_WORDS 2359296
#define KV_WORDS 589824
#define GROUP_WORDS (Q_WORDS+2*KV_WORDS)
#define ROTATIONS 16
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }

__Q__
__PAIR__
__FULL__

static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));exit(2);} }
static void control(float* out,unsigned int* wq,unsigned int* wk,unsigned int* wv,half* x) {
  q4k_g3_lanemap_gemv_vec_4096_4096<<<Q_ROWS,32>>>(out,wq,x);
  q4k_g3_lanemap_gemv_pair_vec_1024_4096<<<KV_ROWS,32>>>(out+Q_ROWS,out+Q_ROWS+KV_ROWS,wk,wv,x);
}
static void candidate(float* out,unsigned int* wq,unsigned int* wkv,half* x) {
  q4k_g3_lanemap_gemv_qkv_full_4096_1024_4096<<<Q_ROWS,32>>>(out,out+Q_ROWS,out+Q_ROWS+KV_ROWS,wq,wkv,x);
}
static void launch(int arm,float* out,unsigned int* group,half* x) {
  if(arm==0) control(out,group,group+Q_WORDS,group+Q_WORDS+KV_WORDS,x);
  else candidate(out,group,group+Q_WORDS,x);
}
static double hot(int arm,float* out,unsigned int* group,half* x,int passes) {
  cudaEvent_t s,e; ck(cudaEventCreate(&s),"event"); ck(cudaEventCreate(&e),"event"); ck(cudaEventRecord(s),"record");
  for(int i=0;i<passes;i++) launch(arm,out,group,x);
  ck(cudaEventRecord(e),"record"); ck(cudaEventSynchronize(e),"sync"); float ms=0; ck(cudaEventElapsedTime(&ms,s,e),"elapsed");
  cudaEventDestroy(s); cudaEventDestroy(e); return ms*1000.0/passes;
}
static double rotated(int arm,float* out,unsigned int* groups,half* x,int passes) {
  cudaEvent_t s,e; ck(cudaEventCreate(&s),"event"); ck(cudaEventCreate(&e),"event"); double us=0;
  for(int i=0;i<passes;i++) { unsigned int* g=groups+(i%ROTATIONS)*GROUP_WORDS; ck(cudaEventRecord(s),"record");
    launch(arm,out,g,x); ck(cudaEventRecord(e),"record"); ck(cudaEventSynchronize(e),"sync"); float ms=0; ck(cudaEventElapsedTime(&ms,s,e),"elapsed"); us+=ms*1000.0; }
  cudaEventDestroy(s); cudaEventDestroy(e); return us/passes;
}
int main(int argc,char** argv) {
  int hot_passes=argc>1?atoi(argv[1]):100,cold_passes=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):7;
  float *ctrl,*cand; unsigned int* groups; half* x;
  ck(cudaMalloc(&ctrl,(Q_ROWS+2*KV_ROWS)*sizeof(float)),"ctrl"); ck(cudaMalloc(&cand,(Q_ROWS+2*KV_ROWS)*sizeof(float)),"cand");
  ck(cudaMalloc(&groups,(size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int)),"groups"); ck(cudaMalloc(&x,K*sizeof(half)),"x");
  unsigned int* hw=(unsigned int*)malloc((size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int)); half* hx=(half*)malloc(K*sizeof(half));
  for(size_t i=0;i<(size_t)ROTATIONS*GROUP_WORDS;i++) hw[i]=(unsigned int)((i*2654435761u)^0x9e3779b9u);
  for(int i=0;i<K;i++) hx[i]=__float2half(((i%257)-128)*0.03125f);
  ck(cudaMemcpy(groups,hw,(size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int),cudaMemcpyHostToDevice),"weights");
  ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"x"); free(hw); free(hx);
  control(ctrl,groups,groups+Q_WORDS,groups+Q_WORDS+KV_WORDS,x); candidate(cand,groups,groups+Q_WORDS,x); ck(cudaDeviceSynchronize(),"warmup");
  float *hc=(float*)malloc((Q_ROWS+2*KV_ROWS)*sizeof(float)),*hn=(float*)malloc((Q_ROWS+2*KV_ROWS)*sizeof(float));
  ck(cudaMemcpy(hc,ctrl,(Q_ROWS+2*KV_ROWS)*sizeof(float),cudaMemcpyDeviceToHost),"copy"); ck(cudaMemcpy(hn,cand,(Q_ROWS+2*KV_ROWS)*sizeof(float),cudaMemcpyDeviceToHost),"copy");
  printf("bitwise_identical=%d\n",memcmp(hc,hn,(Q_ROWS+2*KV_ROWS)*sizeof(float))==0); free(hc); free(hn);
  for(int r=0;r<reps;r++) { double ch=hot(0,ctrl,groups,x,hot_passes),nh=hot(1,cand,groups,x,hot_passes);
    double cc=rotated(0,ctrl,groups,x,cold_passes),nc=rotated(1,cand,groups,x,cold_passes);
    printf("rep=%d hot_control=%.6f hot_candidate=%.6f cold_control=%.6f cold_candidate=%.6f\n",r,ch,nh,cc,nc); }
  return 0;
}
'''


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--hot-passes",type=int,default=100)
  ap.add_argument("--cold-passes",type=int,default=32); ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
  q,pair,full=_render(); source=HARNESS.replace("__Q__",q).replace("__PAIR__",pair).replace("__FULL__",full)
  with tempfile.TemporaryDirectory(prefix="q4k_ordinary_qkv_") as td:
    cu,binp=Path(td)/"gate.cu",Path(td)/"gate"; cu.write_text(source)
    env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    cp=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(cu),"-o",str(binp)],capture_output=True,text=True,env=env)
    if cp.returncode: raise RuntimeError(cp.stderr[-8000:])
    run=subprocess.run([str(binp),str(args.hot_passes),str(args.cold_passes),str(args.reps)],capture_output=True,text=True)
    if run.returncode: raise RuntimeError(run.stderr[-4000:])
  print(run.stdout.strip())
  rows=[]
  for line in run.stdout.splitlines():
    m=re.match(r"rep=(\d+) hot_control=([0-9.]+) hot_candidate=([0-9.]+) cold_control=([0-9.]+) cold_candidate=([0-9.]+)",line)
    if m: rows.append({"rep":int(m.group(1)),"hot_control_us":float(m.group(2)),"hot_candidate_us":float(m.group(3)),
      "cold_control_us":float(m.group(4)),"cold_candidate_us":float(m.group(5))})
  med=lambda key:statistics.median(r[key] for r in rows)
  out={"schema":"tinygrad.q4k_ordinary_qkv_full_microgate.v1","bitwise_identical":"bitwise_identical=1" in run.stdout,
    "shape":{"q_rows":Q_ROWS,"kv_rows":KV_ROWS,"k":K},"reps":args.reps,"samples":rows,
    "median":{"hot_control_us":med("hot_control_us"),"hot_candidate_us":med("hot_candidate_us"),
      "hot_recovery_us":med("hot_control_us")-med("hot_candidate_us"),"cold_control_us":med("cold_control_us"),
      "cold_candidate_us":med("cold_candidate_us"),"cold_recovery_us":med("cold_control_us")-med("cold_candidate_us")},
    "ptxas":cp.stderr.strip()}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
  return 0 if out["bitwise_identical"] else 5


if __name__=="__main__": raise SystemExit(main())
