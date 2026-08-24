#!/usr/bin/env python3
"""Bit-exact CUDA-event gate for one shared-Q8 Q4-K/Q6-V pair."""
from __future__ import annotations

import argparse,json,os,re,statistics,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.shared_q8_attention import (_emit_q4_cooperative,_emit_q6_warp_direct,
  _emit_q4_q6_cooperative_pair)
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp

ROWS,K=1024,4096
Q4WORDS,Q6HALFS,PACKED=ROWS*(K//256)*36,ROWS*(K//256)*110,K//4+K//32
CUDA_BIN="/usr/local/cuda-13.2/bin"


def _render() -> tuple[str,str,str]:
  p=UOp.placeholder; extent=UOp.const(dtypes.weakint,4); ren=CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False)
  bodies=(
    _emit_q4_cooperative(ROWS,extent,direct_output=True)(p((ROWS,),dtypes.float32,0),p((Q4WORDS,),dtypes.uint32,1),p((PACKED,),dtypes.uint32,2)),
    _emit_q6_warp_direct(ROWS)(p((ROWS,),dtypes.float32,0),p((Q6HALFS,),dtypes.uint16,1),p((PACKED,),dtypes.uint32,2)),
    _emit_q4_q6_cooperative_pair(ROWS,extent)(p((ROWS,),dtypes.float32,0),p((ROWS,),dtypes.float32,1),
      p((Q4WORDS,),dtypes.uint32,2),p((Q6HALFS,),dtypes.uint16,3),p((PACKED,),dtypes.uint32,4)))
  def src(u:UOp)->str:
    text=next(x.arg for x in to_program(u,ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]
  return tuple(src(u) for u in bodies)


HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define ROWS 1024
#define K 4096
#define Q4WORDS 589824
#define Q6HALFS 1802240
#define PACKED 1152
template <class T,class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
__Q4__
__Q6__
__PAIR__
static void ck(cudaError_t e,const char* w) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);} }
static double control(float*out,unsigned int*qw,unsigned short*qh,unsigned int*xp,int passes){
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<passes;i++){q4k_warp_coop_q8_dp4a_direct_1024_4096<<<ROWS,128>>>(out,qw,xp);q6k_q8_warp_direct_1024_4096<<<ROWS,128>>>(out+ROWS,qh,xp);}
  cudaEventRecord(e);ck(cudaDeviceSynchronize(),"control sync");float ms=0;cudaEventElapsedTime(&ms,s,e);cudaEventDestroy(s);cudaEventDestroy(e);return ms*1000.0/passes;
}
static double candidate(float*out,unsigned int*qw,unsigned short*qh,unsigned int*xp,int passes){
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<passes;i++)q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096<<<ROWS,128>>>(out,out+ROWS,qw,qh,xp);
  cudaEventRecord(e);ck(cudaDeviceSynchronize(),"candidate sync");float ms=0;cudaEventElapsedTime(&ms,s,e);cudaEventDestroy(s);cudaEventDestroy(e);return ms*1000.0/passes;
}
int main(int argc,char**argv){
  int passes=argc>1?atoi(argv[1]):500,reps=argc>2?atoi(argv[2]):9;float*ctrl=nullptr,*cand=nullptr;
  unsigned int*qw=nullptr,*xp=nullptr;unsigned short*qh=nullptr;
  ck(cudaMalloc(&ctrl,2*ROWS*sizeof(float)),"ctrl");ck(cudaMalloc(&cand,2*ROWS*sizeof(float)),"cand");
  ck(cudaMalloc(&qw,Q4WORDS*sizeof(unsigned int)),"qw");ck(cudaMalloc(&qh,Q6HALFS*sizeof(unsigned short)),"qh");ck(cudaMalloc(&xp,PACKED*sizeof(unsigned int)),"xp");
  auto hqw=(unsigned int*)malloc(Q4WORDS*sizeof(unsigned int));auto hqh=(unsigned short*)malloc(Q6HALFS*sizeof(unsigned short));auto hxp=(unsigned int*)malloc(PACKED*sizeof(unsigned int));
  for(int i=0;i<Q4WORDS;i++)hqw[i]=(i*2654435761u)^0x9e3779b9u;
  for(int i=0;i<Q6HALFS;i++)hqh[i]=(unsigned short)((i*40503u)^0x5a5au);
  for(int row=0;row<ROWS;row++)for(int block=0;block<K/256;block++){hqw[(row*(K/256)+block)*36]=0x24002400u;hqh[(row*(K/256)+block)*110+104]=0x2400u;}
  for(int i=0;i<K/4;i++)hxp[i]=(i*3266489917u)^0xc2b2ae35u;for(int i=K/4;i<PACKED;i++)hxp[i]=0x00002400u;
  ck(cudaMemcpy(qw,hqw,Q4WORDS*sizeof(unsigned int),cudaMemcpyHostToDevice),"qw copy");ck(cudaMemcpy(qh,hqh,Q6HALFS*sizeof(unsigned short),cudaMemcpyHostToDevice),"qh copy");
  ck(cudaMemcpy(xp,hxp,PACKED*sizeof(unsigned int),cudaMemcpyHostToDevice),"xp copy");free(hqw);free(hqh);free(hxp);
  q4k_warp_coop_q8_dp4a_direct_1024_4096<<<ROWS,128>>>(ctrl,qw,xp);q6k_q8_warp_direct_1024_4096<<<ROWS,128>>>(ctrl+ROWS,qh,xp);
  q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096<<<ROWS,128>>>(cand,cand+ROWS,qw,qh,xp);ck(cudaDeviceSynchronize(),"warmup");
  auto hc=(float*)malloc(2*ROWS*sizeof(float));auto hp=(float*)malloc(2*ROWS*sizeof(float));ck(cudaMemcpy(hc,ctrl,2*ROWS*sizeof(float),cudaMemcpyDeviceToHost),"ctrl copy");
  ck(cudaMemcpy(hp,cand,2*ROWS*sizeof(float),cudaMemcpyDeviceToHost),"cand copy");int mismatch=0;for(int i=0;i<2*ROWS;i++)mismatch+=memcmp(&hc[i],&hp[i],sizeof(float))!=0;
  printf("mismatched_words=%d bitwise_identical=%d\n",mismatch,mismatch==0);free(hc);free(hp);
  for(int r=0;r<reps;r++)printf("rep=%d control_pair=%.6f candidate_pair=%.6f\n",r,control(ctrl,qw,qh,xp,passes),candidate(cand,qw,qh,xp,passes));
  cudaFree(ctrl);cudaFree(cand);cudaFree(qw);cudaFree(qh);cudaFree(xp);return mismatch?5:0;
}
'''


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--passes",type=int,default=500);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
  q4,q6,pair=_render();source=HARNESS.replace("__Q4__",q4).replace("__Q6__",q6).replace("__PAIR__",pair)
  with tempfile.TemporaryDirectory(prefix="q4q6_shared_pair_") as td:
    src=Path(td)/"gate.cu";binary=Path(td)/"gate";src.write_text(source);env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    build=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(src),"-o",str(binary)],capture_output=True,text=True,env=env)
    if build.returncode:print(build.stderr[-8000:],file=sys.stderr);return 3
    run=subprocess.run([str(binary),str(args.passes),str(args.reps)],capture_output=True,text=True);print(run.stdout.strip())
    if run.returncode not in (0,5):print(run.stderr[-4000:],file=sys.stderr);return 4
    mm=re.search(r"mismatched_words=(\d+) bitwise_identical=(\d+)",run.stdout);cv,pv=[],[]
    for line in run.stdout.splitlines():
      if(m:=re.search(r"control_pair=([0-9.]+) candidate_pair=([0-9.]+)",line)):cv.append(float(m.group(1)));pv.append(float(m.group(2)))
    cm,pm=statistics.median(cv),statistics.median(pv);saving=cm-pm
    result={"schema":"tinygrad.q4k_q6k_shared_q8_pair_microgate.v1","commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
      "shape":{"rows":ROWS,"k":K,"shared_q8_q4q6_pairs_per_token":8},"passes":args.passes,"reps":args.reps,"bitwise_identical":bool(mm and int(mm.group(2))),
      "mismatched_words":int(mm.group(1)) if mm else None,"timing":{"unit":"us_per_pair_cuda_event","control_samples":cv,"candidate_samples":pv,
        "control_median":cm,"candidate_median":pm,"recovery_us_per_pair":saving,"projected_8_pair_recovery_us":saving*8,"candidate_over_control":pm/cm},
      "ptxas":build.stderr.strip().splitlines(),"verdict":"ADVANCE" if mm and int(mm.group(2)) and pm<cm and saving*8>=8 else "NO_GO"}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0 if result["bitwise_identical"] else 5


if __name__=="__main__":raise SystemExit(main())
