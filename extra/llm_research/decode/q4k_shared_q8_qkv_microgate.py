#!/usr/bin/env python3
"""Bit-exact hot/cold complete-span gate for the shared-Q8 Q4/Q4/Q4 producer."""
from __future__ import annotations

import argparse, csv, io, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.shared_q8_attention import (_emit_q4_cooperative, _emit_q4_cooperative_pair, _emit_q4_cooperative_qkv,
  _emit_q4_cooperative_qkv_balanced)
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

Q_ROWS,KV_ROWS,K=4096,1024,4096
Q_WORDS=Q_ROWS*(K//256)*36
KV_WORDS=KV_ROWS*(K//256)*36
PACKED=K//4+K//32
CUDA_BIN="/usr/local/cuda-13.2/bin"
NCU="/usr/local/bin/ncu"


def _ncu_csv(binary:Path,symbol:str) -> dict:
  metrics=",".join(("dram__bytes.sum","dram__bytes_op_read.sum","dram__bytes_op_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed","gpu__time_duration.sum","lts__t_bytes.sum",
    "lts__t_sector_op_read_hit_rate.pct","sm__throughput.avg.pct_of_peak_sustained_elapsed"))
  cp=subprocess.run(["sudo","-n",NCU,"-k",symbol,"--launch-skip","1","--launch-count","1","--cache-control","all",
    "--metrics",metrics,"--csv",str(binary),"1","1","1"],capture_output=True,text=True)
  if cp.returncode: raise RuntimeError(f"ncu failed for {symbol} rc={cp.returncode}: {cp.stderr[-4000:]}")
  rows=[]; header=None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0]=="ID": header=cols; continue
    if header is not None and len(cols)==len(header):
      row=dict(zip(header,cols)); rows.append({"metric":row["Metric Name"],"unit":row["Metric Unit"],"value":row["Metric Value"]})
  return {"symbol":symbol,"rows":rows,"stderr":cp.stderr.strip()}


def _render() -> tuple[str,str,str,str]:
  p=UOp.placeholder; extent=UOp.const(dtypes.weakint,4)
  q=_emit_q4_cooperative(Q_ROWS,extent,direct_output=True)(
    p((Q_ROWS,),dtypes.float32,0),p((Q_WORDS,),dtypes.uint32,1),p((PACKED,),dtypes.uint32,2))
  pair=_emit_q4_cooperative_pair(KV_ROWS,extent)(p((KV_ROWS,),dtypes.float32,0),p((KV_ROWS,),dtypes.float32,1),
    p((KV_WORDS,),dtypes.uint32,2),p((KV_WORDS,),dtypes.uint32,3),p((PACKED,),dtypes.uint32,4))
  triple=_emit_q4_cooperative_qkv(extent)(p((Q_ROWS,),dtypes.float32,0),p((KV_ROWS,),dtypes.float32,1),
    p((KV_ROWS,),dtypes.float32,2),p((Q_WORDS,),dtypes.uint32,3),p((KV_WORDS,),dtypes.uint32,4),
    p((KV_WORDS,),dtypes.uint32,5),p((PACKED,),dtypes.uint32,6))
  balanced=_emit_q4_cooperative_qkv_balanced(extent)(p((Q_ROWS,),dtypes.float32,0),p((KV_ROWS*2,),dtypes.float32,1),
    p((Q_WORDS,),dtypes.uint32,2),p((KV_WORDS*2,),dtypes.uint32,3),p((PACKED,),dtypes.uint32,4))
  ren=CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False)
  def src(u:UOp) -> str:
    text=next(x.arg for x in to_program(u,ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]
  return src(q),src(pair),src(triple),src(balanced)


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
#define PACKED 1152
#define EVICT_BYTES (256*1024*1024)
#define GROUP_WORDS (Q_WORDS+2*KV_WORDS)
#define ROTATIONS 16
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

__Q__
__PAIR__
__TRIPLE__
__BALANCED__

static void ck(cudaError_t e,const char* what) { if (e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e)); exit(2); } }
static void control(float* out,unsigned int* wq,unsigned int* wk,unsigned int* wv,unsigned int* xp) {
  q4k_warp_coop_q8_dp4a_direct_4096_4096<<<Q_ROWS,128>>>(out,wq,xp);
  q4k_warp_coop_q8_dp4a_pair_direct_1024_4096<<<KV_ROWS,128>>>(out+Q_ROWS,out+Q_ROWS+KV_ROWS,wk,wv,xp);
}
static void candidate(float* out,unsigned int* wq,unsigned int* wk,unsigned int* wv,unsigned int* xp) {
  q4k_warp_coop_q8_dp4a_qkv_direct_4096_1024_4096<<<KV_ROWS,128>>>(out,out+Q_ROWS,out+Q_ROWS+KV_ROWS,wq,wk,wv,xp);
}
static void balanced(float* out,unsigned int* wq,unsigned int* wkv,unsigned int* xp) {
  q4k_warp_coop_q8_dp4a_qkv_balanced_direct_4096_1024_4096<<<2*KV_ROWS,128>>>(out,out+Q_ROWS,wq,wkv,xp);
}
static void launch(int arm,float* out,unsigned int* wq,unsigned int* wk,unsigned int* wv,unsigned int* wkv,unsigned int* xp) {
  if (arm==0) control(out,wq,wk,wv,xp); else if (arm==1) candidate(out,wq,wk,wv,xp); else balanced(out,wq,wkv,xp);
}
static double hot(int arm,float* out,unsigned int* wq,unsigned int* wk,unsigned int* wv,unsigned int* wkv,unsigned int* xp,int passes) {
  cudaEvent_t s,e; ck(cudaEventCreate(&s),"event"); ck(cudaEventCreate(&e),"event"); ck(cudaEventRecord(s),"record");
  for (int i=0;i<passes;i++) launch(arm,out,wq,wk,wv,wkv,xp);
  ck(cudaEventRecord(e),"record"); ck(cudaEventSynchronize(e),"sync"); float ms=0; ck(cudaEventElapsedTime(&ms,s,e),"elapsed");
  cudaEventDestroy(s); cudaEventDestroy(e); return ms*1000.0/passes;
}
static double cold(int arm,float* out,unsigned int* wq,unsigned int* wk,unsigned int* wv,unsigned int* wkv,unsigned int* xp,unsigned char* evict,int passes) {
  cudaEvent_t s,e; ck(cudaEventCreate(&s),"event"); ck(cudaEventCreate(&e),"event"); double us=0;
  for (int i=0;i<passes;i++) {
    ck(cudaMemset(evict,17+i,EVICT_BYTES),"evict"); ck(cudaEventRecord(s),"record");
    launch(arm,out,wq,wk,wv,wkv,xp);
    ck(cudaEventRecord(e),"record"); ck(cudaEventSynchronize(e),"sync"); float ms=0; ck(cudaEventElapsedTime(&ms,s,e),"elapsed"); us+=ms*1000.0;
  }
  cudaEventDestroy(s); cudaEventDestroy(e); return us/passes;
}
static double rotated(int arm,float* out,unsigned int* rotations,unsigned int* xp,int passes) {
  cudaEvent_t s,e; ck(cudaEventCreate(&s),"event"); ck(cudaEventCreate(&e),"event"); double us=0;
  for (int i=0;i<passes;i++) {
    unsigned int* wq=rotations+(arm*ROTATIONS+(i%ROTATIONS))*GROUP_WORDS;
    unsigned int* wk=wq+Q_WORDS; unsigned int* wv=wk+KV_WORDS;
    ck(cudaEventRecord(s),"record"); launch(arm,out,wq,wk,wv,wk,xp); ck(cudaEventRecord(e),"record");
    ck(cudaEventSynchronize(e),"sync"); float ms=0; ck(cudaEventElapsedTime(&ms,s,e),"elapsed"); us+=ms*1000.0;
  }
  cudaEventDestroy(s); cudaEventDestroy(e); return us/passes;
}
int main(int argc,char** argv) {
  int hot_passes=argc>1?atoi(argv[1]):300,cold_passes=argc>2?atoi(argv[2]):20,reps=argc>3?atoi(argv[3]):9;
  float *ctrl=nullptr,*cand=nullptr,*bal=nullptr; unsigned int *wq=nullptr,*wk=nullptr,*wv=nullptr,*wkv=nullptr,*xp=nullptr,*rotations=nullptr; unsigned char* evict=nullptr;
  ck(cudaMalloc(&ctrl,(Q_ROWS+2*KV_ROWS)*sizeof(float)),"ctrl"); ck(cudaMalloc(&cand,(Q_ROWS+2*KV_ROWS)*sizeof(float)),"cand");
  ck(cudaMalloc(&bal,(Q_ROWS+2*KV_ROWS)*sizeof(float)),"bal"); ck(cudaMalloc(&wq,Q_WORDS*sizeof(unsigned int)),"wq"); ck(cudaMalloc(&wk,KV_WORDS*sizeof(unsigned int)),"wk");
  ck(cudaMalloc(&wv,KV_WORDS*sizeof(unsigned int)),"wv"); ck(cudaMalloc(&wkv,2*KV_WORDS*sizeof(unsigned int)),"wkv"); ck(cudaMalloc(&xp,PACKED*sizeof(unsigned int)),"xp"); ck(cudaMalloc(&evict,EVICT_BYTES),"evict");
  ck(cudaMalloc(&rotations,3ULL*ROTATIONS*GROUP_WORDS*sizeof(unsigned int)),"rotations");
  unsigned int *hwq=(unsigned int*)malloc(Q_WORDS*sizeof(unsigned int));
  unsigned int *hwk=(unsigned int*)malloc(KV_WORDS*sizeof(unsigned int));
  unsigned int *hwv=(unsigned int*)malloc(KV_WORDS*sizeof(unsigned int));
  unsigned int *hxp=(unsigned int*)malloc(PACKED*sizeof(unsigned int));
  for (int i=0;i<Q_WORDS;i++) hwq[i]=(i*2654435761u)^0x9e3779b9u;
  for (int i=0;i<KV_WORDS;i++) { hwk[i]=(i*2246822519u)^0x85ebca6bu; hwv[i]=(i*3266489917u)^0xc2b2ae35u; }
  for (int row=0;row<Q_ROWS;row++) for (int block=0;block<K/256;block++) hwq[(row*(K/256)+block)*36]=0x24002400u;
  for (int row=0;row<KV_ROWS;row++) for (int block=0;block<K/256;block++) {
    hwk[(row*(K/256)+block)*36]=0x24002400u; hwv[(row*(K/256)+block)*36]=0x24002400u;
  }
  for (int i=0;i<K/4;i++) hxp[i]=(i*668265263u)^0x27d4eb2fu;
  for (int i=K/4;i<PACKED;i++) hxp[i]=0x00002400u;
  ck(cudaMemcpy(wq,hwq,Q_WORDS*4,cudaMemcpyHostToDevice),"wq copy"); ck(cudaMemcpy(wk,hwk,KV_WORDS*4,cudaMemcpyHostToDevice),"wk copy");
  ck(cudaMemcpy(wv,hwv,KV_WORDS*4,cudaMemcpyHostToDevice),"wv copy"); ck(cudaMemcpy(xp,hxp,PACKED*4,cudaMemcpyHostToDevice),"xp copy");
  ck(cudaMemcpy(wkv,hwk,KV_WORDS*4,cudaMemcpyHostToDevice),"wkv k copy"); ck(cudaMemcpy(wkv+KV_WORDS,hwv,KV_WORDS*4,cudaMemcpyHostToDevice),"wkv v copy");
  for (int i=0;i<3*ROTATIONS;i++) {
    unsigned int* dst=rotations+(size_t)i*GROUP_WORDS;
    ck(cudaMemcpy(dst,wq,Q_WORDS*4,cudaMemcpyDeviceToDevice),"rotation q"); ck(cudaMemcpy(dst+Q_WORDS,wk,KV_WORDS*4,cudaMemcpyDeviceToDevice),"rotation k");
    ck(cudaMemcpy(dst+Q_WORDS+KV_WORDS,wv,KV_WORDS*4,cudaMemcpyDeviceToDevice),"rotation v");
  }
  free(hwq); free(hwk); free(hwv); free(hxp);
  control(ctrl,wq,wk,wv,xp); candidate(cand,wq,wk,wv,xp); balanced(bal,wq,wkv,xp); ck(cudaDeviceSynchronize(),"warmup");
  float *hc=(float*)malloc((Q_ROWS+2*KV_ROWS)*4),*hp=(float*)malloc((Q_ROWS+2*KV_ROWS)*4);
  ck(cudaMemcpy(hc,ctrl,(Q_ROWS+2*KV_ROWS)*4,cudaMemcpyDeviceToHost),"ctrl copy"); ck(cudaMemcpy(hp,cand,(Q_ROWS+2*KV_ROWS)*4,cudaMemcpyDeviceToHost),"cand copy");
  int mismatch=0; for (int i=0;i<Q_ROWS+2*KV_ROWS;i++) mismatch+=memcmp(&hc[i],&hp[i],4)!=0;
  ck(cudaMemcpy(hp,bal,(Q_ROWS+2*KV_ROWS)*4,cudaMemcpyDeviceToHost),"balanced copy"); int balanced_mismatch=0;
  for (int i=0;i<Q_ROWS+2*KV_ROWS;i++) balanced_mismatch+=memcmp(&hc[i],&hp[i],4)!=0;
  printf("mismatched_words=%d balanced_mismatched_words=%d bitwise_identical=%d\n",mismatch,balanced_mismatch,mismatch==0&&balanced_mismatch==0); free(hc); free(hp);
  for (int r=0;r<reps;r++) {
    int order[3]; if (r&1) { order[0]=2; order[1]=1; order[2]=0; } else { order[0]=0; order[1]=1; order[2]=2; }
    double hh[3],cc[3],rr[3]; float* outs[3]={ctrl,cand,bal};
    for (int i=0;i<3;i++) hh[order[i]]=hot(order[i],outs[order[i]],wq,wk,wv,wkv,xp,hot_passes);
    for (int i=0;i<3;i++) cc[order[i]]=cold(order[i],outs[order[i]],wq,wk,wv,wkv,xp,evict,cold_passes);
    for (int i=0;i<3;i++) rr[order[i]]=rotated(order[i],outs[order[i]],rotations,xp,cold_passes);
    printf("rep=%d hot_control=%.6f hot_candidate=%.6f hot_balanced=%.6f cold_control=%.6f cold_candidate=%.6f cold_balanced=%.6f rotated_control=%.6f rotated_candidate=%.6f rotated_balanced=%.6f\n",r,hh[0],hh[1],hh[2],cc[0],cc[1],cc[2],rr[0],rr[1],rr[2]);
  }
  cudaFree(ctrl); cudaFree(cand); cudaFree(bal); cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(wkv); cudaFree(xp); cudaFree(evict); cudaFree(rotations); return mismatch||balanced_mismatch?5:0;
}
'''


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--hot-passes",type=int,default=300); ap.add_argument("--cold-passes",type=int,default=20)
  ap.add_argument("--reps",type=int,default=9); ap.add_argument("--counters",action="store_true")
  ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
  q,pair,triple,balanced=_render(); source=HARNESS.replace("__Q__",q).replace("__PAIR__",pair).replace("__TRIPLE__",triple).replace("__BALANCED__",balanced)
  with tempfile.TemporaryDirectory(prefix="q4k_shared_q8_qkv_") as td:
    src=Path(td)/"gate.cu"; binary=Path(td)/"gate"; src.write_text(source)
    env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    build=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(src),"-o",str(binary)],capture_output=True,text=True,env=env)
    if build.returncode: print(build.stderr[-12000:],file=sys.stderr); return 3
    run=subprocess.run([str(binary),str(args.hot_passes),str(args.cold_passes),str(args.reps)],capture_output=True,text=True)
    print(run.stdout.strip())
    if run.returncode not in (0,5): print(run.stderr[-4000:],file=sys.stderr); return 4
    mm=re.search(r"mismatched_words=(\d+) balanced_mismatched_words=(\d+) bitwise_identical=(\d+)",run.stdout)
    samples={k:[] for k in ("hot_control","hot_candidate","hot_balanced","cold_control","cold_candidate","cold_balanced",
      "rotated_control","rotated_candidate","rotated_balanced")}
    for line in run.stdout.splitlines():
      for key in samples:
        if (m:=re.search(rf"{key}=([0-9.]+)",line)): samples[key].append(float(m.group(1)))
    med={k:statistics.median(v) for k,v in samples.items()}
    counters=None
    if args.counters:
      symbols=("q4k_warp_coop_q8_dp4a_direct_4096_4096","q4k_warp_coop_q8_dp4a_pair_direct_1024_4096",
        "q4k_warp_coop_q8_dp4a_qkv_direct_4096_1024_4096","q4k_warp_coop_q8_dp4a_qkv_balanced_direct_4096_1024_4096")
      counters={symbol:_ncu_csv(binary,symbol) for symbol in symbols}
    result={"schema":"tinygrad.q4k_shared_q8_qkv_microgate.v1","commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
      "shape":{"q_rows":Q_ROWS,"kv_rows":KV_ROWS,"k":K,"shared_q8_q4q4q4_groups_per_token":9},"hot_passes":args.hot_passes,"cold_passes":args.cold_passes,"reps":args.reps,
      "bitwise_identical":bool(mm and int(mm.group(3))),"mismatched_words":{"collapsed":int(mm.group(1)),"balanced":int(mm.group(2))} if mm else None,"samples":samples,"medians_us":med,
      "recovery_us":{"hot_per_group":med["hot_control"]-med["hot_candidate"],"cold_per_group":med["cold_control"]-med["cold_candidate"],
        "cold_projected_9_groups":9*(med["cold_control"]-med["cold_candidate"]),
        "balanced_hot_per_group":med["hot_control"]-med["hot_balanced"],"balanced_cold_per_group":med["cold_control"]-med["cold_balanced"],
        "balanced_cold_projected_9_groups":9*(med["cold_control"]-med["cold_balanced"]),
        "rotated_per_group":med["rotated_control"]-med["rotated_candidate"],"rotated_projected_9_groups":9*(med["rotated_control"]-med["rotated_candidate"]),
        "balanced_rotated_per_group":med["rotated_control"]-med["rotated_balanced"],"balanced_rotated_projected_9_groups":9*(med["rotated_control"]-med["rotated_balanced"])},
      "ptxas":build.stderr.strip().splitlines(),"ncu":counters,
      "verdict":"ADVANCE_BALANCED" if mm and int(mm.group(3)) and med["rotated_balanced"]<med["rotated_control"] else "NO_GO"}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result["bitwise_identical"] else 5


if __name__=="__main__": raise SystemExit(main())
