#!/usr/bin/env python3
"""Admissibility/tax gate for ClusterFusion-style O ownership.

ClusterFusion partitions output-projection work over attention heads and uses
FP16 atomicAdd to join head contributions.  Tinygrad's installed dense route
requires deterministic FP32 association.  This gate measures the numerical
gap of the atomic topology and the minimum global-scratch tax of an exact
left-to-right replacement.  It is research-only and changes no model route.
"""
from __future__ import annotations

import argparse, json, re, shutil, statistics, subprocess, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
NVCC="/usr/local/cuda-13.2/bin/nvcc"

SOURCE=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
constexpr int HEADS=32,ROWS=4096,THREADS=256;
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}

__global__ void direct_exact(float*out,const float*part){
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<ROWS;row+=gridDim.x*blockDim.x){
    float acc=0.0f;
    #pragma unroll
    for(int h=0;h<HEADS;h++)acc=acc+part[h*ROWS+row];
    out[row]=acc;
  }
}
__global__ void write_scratch(float*scratch,const float*part){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<HEADS*ROWS;i+=gridDim.x*blockDim.x)scratch[i]=part[i];
}
__global__ void combine_exact(float*out,const float*scratch){
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<ROWS;row+=gridDim.x*blockDim.x){
    float acc=0.0f;
    #pragma unroll
    for(int h=0;h<HEADS;h++)acc=acc+scratch[h*ROWS+row];
    out[row]=acc;
  }
}
__global__ void atomic_half(half*out,const float*part){
  const int h=blockIdx.x;
  for(int row=threadIdx.x;row<ROWS;row+=blockDim.x)atomicAdd(out+row,__float2half(part[h*ROWS+row]));
}

template<class F>static double timed(F launch,int iters,cudaStream_t s){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ea");ck(cudaEventCreate(&b),"eb");for(int i=0;i<100;i++)launch();ck(cudaEventRecord(a,s),"ra");for(int i=0;i<iters;i++)launch();ck(cudaEventRecord(b,s),"rb");ck(cudaEventSynchronize(b),"sb");float ms;ck(cudaEventElapsedTime(&ms,a,b),"el");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0/iters;}

int main(int argc,char**argv){int reps=argc>1?atoi(argv[1]):9,iters=argc>2?atoi(argv[2]):1000;float *part,*scratch,*direct,*exact;half*atomic;
  ck(cudaMalloc(&part,(size_t)HEADS*ROWS*4),"part");ck(cudaMalloc(&scratch,(size_t)HEADS*ROWS*4),"scratch");ck(cudaMalloc(&direct,ROWS*4),"direct");ck(cudaMalloc(&exact,ROWS*4),"exact");ck(cudaMalloc(&atomic,ROWS*2),"atomic");
  std::vector<float>hp((size_t)HEADS*ROWS);for(int h=0;h<HEADS;h++)for(int r=0;r<ROWS;r++){int q=((h*131+r*17)%257)-128;hp[(size_t)h*ROWS+r]=(float)q*(h%3==0?0.0009765625f:0.00390625f);}
  ck(cudaMemcpy(part,hp.data(),hp.size()*4,cudaMemcpyHostToDevice),"part copy");cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");
  direct_exact<<<16,THREADS,0,s>>>(direct,part);write_scratch<<<170,THREADS,0,s>>>(scratch,part);combine_exact<<<16,THREADS,0,s>>>(exact,scratch);ck(cudaMemsetAsync(atomic,0,ROWS*2,s),"zero");atomic_half<<<HEADS,THREADS,0,s>>>(atomic,part);ck(cudaStreamSynchronize(s),"validate sync");
  std::vector<float>hd(ROWS),he(ROWS);std::vector<half>ha(ROWS);ck(cudaMemcpy(hd.data(),direct,ROWS*4,cudaMemcpyDeviceToHost),"direct copy");ck(cudaMemcpy(he.data(),exact,ROWS*4,cudaMemcpyDeviceToHost),"exact copy");ck(cudaMemcpy(ha.data(),atomic,ROWS*2,cudaMemcpyDeviceToHost),"atomic copy");
  int bitwise=memcmp(hd.data(),he.data(),ROWS*4)==0;double ss=0,se=0;float ma=0;for(int i=0;i<ROWS;i++){float av=__half2float(ha[i]),e=av-hd[i];ma=fmaxf(ma,fabsf(e));ss+=(double)e*e;se+=(double)hd[i]*hd[i];}printf("validate exact_bitwise=%d atomic_max_abs=%.9g atomic_rel_l2=%.9g\n",bitwise,ma,sqrt(ss/se));
  for(int r=0;r<reps;r++){double d=timed([&](){direct_exact<<<16,THREADS,0,s>>>(direct,part);},iters,s);double x=timed([&](){write_scratch<<<170,THREADS,0,s>>>(scratch,part);combine_exact<<<16,THREADS,0,s>>>(exact,scratch);},iters,s);double a=timed([&](){cudaMemsetAsync(atomic,0,ROWS*2,s);atomic_half<<<HEADS,THREADS,0,s>>>(atomic,part);},iters,s);printf("sample rep=%d direct_us=%.6f scratch_exact_us=%.6f atomic_half_us=%.6f\n",r,d,x,a);}
  return bitwise?0:5;
}
'''

VAL=re.compile(r"validate exact_bitwise=(\d+) atomic_max_abs=([0-9.eE+-]+) atomic_rel_l2=([0-9.eE+-]+)")
SAMPLE=re.compile(r"sample rep=(\d+) direct_us=([0-9.]+) scratch_exact_us=([0-9.]+) atomic_half_us=([0-9.]+)")

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--reps",type=int,default=9);ap.add_argument("--iters",type=int,default=1000);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--artifact-dir",type=Path);args=ap.parse_args()
  with tempfile.TemporaryDirectory(prefix="nv_clusterfusion_o_exact_") as td:
    cu,exe=Path(td)/"gate.cu",Path(td)/"gate";cu.write_text(SOURCE)
    cp=subprocess.run([NVCC,"-arch=sm_120a","-O3","-std=c++20","--ptxas-options=-v",str(cu),"-o",str(exe)],capture_output=True,text=True)
    if cp.returncode:raise RuntimeError(cp.stdout+"\n"+cp.stderr)
    run=subprocess.run([str(exe),str(args.reps),str(args.iters)],capture_output=True,text=True,timeout=600)
    if run.returncode:raise RuntimeError(run.stdout+"\n"+run.stderr)
    if args.artifact_dir:args.artifact_dir.mkdir(parents=True,exist_ok=True);shutil.copy2(cu,args.artifact_dir/"gate.cu");shutil.copy2(exe,args.artifact_dir/"gate")
  vm=next((VAL.fullmatch(x) for x in run.stdout.splitlines() if VAL.fullmatch(x)),None);rows=[]
  for line in run.stdout.splitlines():
    if m:=SAMPLE.fullmatch(line):rows.append({"rep":int(m[1]),"direct_us":float(m[2]),"scratch_exact_us":float(m[3]),"atomic_half_us":float(m[4])})
  if vm is None or not rows:raise RuntimeError("unparsed output\n"+run.stdout)
  med={k:statistics.median(r[k] for r in rows) for k in ("direct_us","scratch_exact_us","atomic_half_us")};med["scratch_exact_tax_us"]=med["scratch_exact_us"]-med["direct_us"]
  result={"schema":"tinygrad.nv_clusterfusion_o_exactness_gate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"shape":{"heads":32,"rows":4096,"partial_bytes":32*4096*4},"validation":{"exact_scratch_bitwise":bool(int(vm[1])),"atomic_half_max_abs":float(vm[2]),"atomic_half_rel_l2":float(vm[3])},"median":med,"samples":rows,"stdout":run.stdout,"ptxas":cp.stderr}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":main()
