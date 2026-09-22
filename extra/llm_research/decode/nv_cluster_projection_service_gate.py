#!/usr/bin/env python3
"""Measure Blackwell thread-block-cluster handoff for dense decode projections.

This is a research-only falsification gate.  It does not implement a model
route.  Each cluster block produces one disjoint segment of a 4096-element
activation, then every block consumes the complete activation.  The gate
compares distributed shared memory with a global-memory handoff and a direct
global read, while separately charging the cluster barrier.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, statistics, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NVCC = "/usr/local/cuda-13.2/bin/nvcc"

SOURCE = r'''
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace cg = cooperative_groups;

constexpr int WIDTH=4096;
constexpr int THREADS=128;

static void ck(cudaError_t e,const char*w) {
  if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e)); std::exit(2); }
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void cluster_barrier(float*out) {
  cg::cluster_group cluster=cg::this_cluster();
  float v=(float)(blockIdx.x+threadIdx.x);
  cluster.sync();
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=v;
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void dsm_handoff(float*out,const float*in) {
  constexpr int SEG=WIDTH/CS;
  __shared__ float tile[SEG];
  cg::cluster_group cluster=cg::this_cluster();
  const int rank=cluster.block_rank();
  for(int i=threadIdx.x;i<SEG;i+=blockDim.x) tile[i]=in[rank*SEG+i]+(float)rank*0.25f;
  cluster.sync();
  float acc=0.0f;
  for(int i=threadIdx.x;i<WIDTH;i+=blockDim.x) {
    const int remote_rank=i/SEG, offset=i-remote_rank*SEG;
    float*remote=cluster.map_shared_rank(tile,remote_rank);
    acc+=remote[offset];
  }
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=acc;
  cluster.sync();
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void global_handoff(float*out,float*stage,const float*in) {
  constexpr int SEG=WIDTH/CS;
  cg::cluster_group cluster=cg::this_cluster();
  const int rank=cluster.block_rank(), cid=blockIdx.x/CS;
  float*base=stage+(size_t)cid*WIDTH;
  for(int i=threadIdx.x;i<SEG;i+=blockDim.x) base[rank*SEG+i]=in[rank*SEG+i]+(float)rank*0.25f;
  cluster.sync();
  float acc=0.0f;
  for(int i=threadIdx.x;i<WIDTH;i+=blockDim.x) acc+=base[i];
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=acc;
  cluster.sync();
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void direct_global(float*out,const float*in) {
  cg::cluster_group cluster=cg::this_cluster();
  float acc=0.0f;
  for(int i=threadIdx.x;i<WIDTH;i+=blockDim.x) acc+=in[i/ (WIDTH/CS) * (WIDTH/CS) + i%(WIDTH/CS)] + (float)(i/(WIDTH/CS))*0.25f;
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=acc;
  cluster.sync();
}

template<class F> static double timed(F launch,int iters,cudaStream_t stream) {
  cudaEvent_t a,b; ck(cudaEventCreate(&a),"event-a"); ck(cudaEventCreate(&b),"event-b");
  for(int i=0;i<100;i++) launch();
  ck(cudaEventRecord(a,stream),"record-a");
  for(int i=0;i<iters;i++) launch();
  ck(cudaEventRecord(b,stream),"record-b"); ck(cudaEventSynchronize(b),"sync-b");
  float ms=0; ck(cudaEventElapsedTime(&ms,a,b),"elapsed");
  cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0/iters;
}

template<int CS> static void run_size(int clusters,int reps,int iters,float*out,float*stage,float*in,cudaStream_t stream) {
  const int blocks=clusters*CS;
  dsm_handoff<CS><<<blocks,THREADS,0,stream>>>(out,in); ck(cudaGetLastError(),"dsm warm");
  global_handoff<CS><<<blocks,THREADS,0,stream>>>(out,stage,in); ck(cudaGetLastError(),"global warm");
  direct_global<CS><<<blocks,THREADS,0,stream>>>(out,in); ck(cudaGetLastError(),"direct warm");
  ck(cudaStreamSynchronize(stream),"warm sync");
  for(int r=0;r<reps;r++) {
    const double barrier=timed([&](){cluster_barrier<CS><<<blocks,THREADS,0,stream>>>(out);},iters,stream);
    const double dsm=timed([&](){dsm_handoff<CS><<<blocks,THREADS,0,stream>>>(out,in);},iters,stream);
    const double global=timed([&](){global_handoff<CS><<<blocks,THREADS,0,stream>>>(out,stage,in);},iters,stream);
    const double direct=timed([&](){direct_global<CS><<<blocks,THREADS,0,stream>>>(out,in);},iters,stream);
    std::printf("sample cs=%d clusters=%d rep=%d barrier_us=%.6f dsm_us=%.6f global_us=%.6f direct_us=%.6f\n",
      CS,clusters,r,barrier,dsm,global,direct);
  }
}

template<int CS> static bool validate_size(int clusters,float*out,float*stage,float*in,cudaStream_t stream) {
  const int blocks=clusters*CS;const size_t count=(size_t)blocks*THREADS;
  std::vector<float>dsm(count),global(count),direct(count);
  dsm_handoff<CS><<<blocks,THREADS,0,stream>>>(out,in);ck(cudaGetLastError(),"validate dsm");
  ck(cudaMemcpyAsync(dsm.data(),out,count*sizeof(float),cudaMemcpyDeviceToHost,stream),"copy dsm");
  global_handoff<CS><<<blocks,THREADS,0,stream>>>(out,stage,in);ck(cudaGetLastError(),"validate global");
  ck(cudaMemcpyAsync(global.data(),out,count*sizeof(float),cudaMemcpyDeviceToHost,stream),"copy global");
  direct_global<CS><<<blocks,THREADS,0,stream>>>(out,in);ck(cudaGetLastError(),"validate direct");
  ck(cudaMemcpyAsync(direct.data(),out,count*sizeof(float),cudaMemcpyDeviceToHost,stream),"copy direct");
  ck(cudaStreamSynchronize(stream),"validate sync");
  bool exact=std::memcmp(dsm.data(),global.data(),count*sizeof(float))==0 && std::memcmp(dsm.data(),direct.data(),count*sizeof(float))==0;
  bool finite=true;for(size_t i=0;i<count;i++)finite&=std::isfinite(dsm[i])&&std::isfinite(global[i])&&std::isfinite(direct[i]);
  std::printf("validate cs=%d clusters=%d exact=%d finite=%d\n",CS,clusters,(int)exact,(int)finite);
  return exact&&finite;
}

int main(int argc,char**argv) {
  const int reps=argc>1?std::atoi(argv[1]):9, iters=argc>2?std::atoi(argv[2]):1000;
  int cluster_launch=0,sm_count=0;cudaDeviceProp p{};
  ck(cudaGetDeviceProperties(&p,0),"properties");
  ck(cudaDeviceGetAttribute(&cluster_launch,cudaDevAttrClusterLaunch,0),"cluster attr");
  ck(cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0),"sm attr");
  std::printf("device name=%s cc=%d.%d sm=%d cluster_launch=%d\n",p.name,p.major,p.minor,sm_count,cluster_launch);
  if(!cluster_launch) return 4;
  const int max_blocks=sm_count*4;
  float *out,*stage,*in;cudaStream_t stream;
  ck(cudaMalloc(&out,(size_t)max_blocks*THREADS*sizeof(float)),"out");
  ck(cudaMalloc(&stage,(size_t)max_blocks*WIDTH*sizeof(float)),"stage");
  ck(cudaMalloc(&in,WIDTH*sizeof(float)),"in");
  float h[WIDTH];for(int i=0;i<WIDTH;i++)h[i]=(float)((i%127)-63)*0.03125f;
  ck(cudaMemcpy(in,h,sizeof(h),cudaMemcpyHostToDevice),"input");
  ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
  bool valid=true;
  valid&=validate_size<2>((sm_count+1)/2,out,stage,in,stream);run_size<2>((sm_count+1)/2,reps,iters,out,stage,in,stream);
  valid&=validate_size<4>((sm_count+3)/4,out,stage,in,stream);run_size<4>((sm_count+3)/4,reps,iters,out,stage,in,stream);
  valid&=validate_size<8>((sm_count+7)/8,out,stage,in,stream);run_size<8>((sm_count+7)/8,reps,iters,out,stage,in,stream);
  ck(cudaStreamSynchronize(stream),"final sync");
  return valid?0:5;
}
'''

SAMPLE = re.compile(
  r"sample cs=(\d+) clusters=(\d+) rep=(\d+) barrier_us=([0-9.]+) dsm_us=([0-9.]+) global_us=([0-9.]+) direct_us=([0-9.]+)")
DEVICE = re.compile(r"device name=(.+) cc=(\d+)\.(\d+) sm=(\d+) cluster_launch=(\d+)")
VALIDATE = re.compile(r"validate cs=(\d+) clusters=(\d+) exact=(\d+) finite=(\d+)")


def main() -> None:
  ap=argparse.ArgumentParser()
  ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--iters",type=int,default=1000)
  ap.add_argument("--out",type=Path,required=True)
  ap.add_argument("--artifact-dir",type=Path)
  args=ap.parse_args()
  with tempfile.TemporaryDirectory(prefix="nv_cluster_projection_") as td:
    cu,exe=Path(td)/"gate.cu",Path(td)/"gate"
    cu.write_text(SOURCE)
    cp=subprocess.run([NVCC,"-arch=sm_120a","-O3","-std=c++20","--ptxas-options=-v",str(cu),"-o",str(exe)],capture_output=True,text=True)
    if cp.returncode: raise RuntimeError(cp.stdout+"\n"+cp.stderr)
    run=subprocess.run([str(exe),str(args.reps),str(args.iters)],capture_output=True,text=True,timeout=600)
    if run.returncode: raise RuntimeError(run.stdout+"\n"+run.stderr)
    if args.artifact_dir:
      args.artifact_dir.mkdir(parents=True,exist_ok=True)
      shutil.copy2(cu,args.artifact_dir/"gate.cu");shutil.copy2(exe,args.artifact_dir/"gate")
  rows=[]
  for line in run.stdout.splitlines():
    if m:=SAMPLE.fullmatch(line):
      rows.append({"cluster_size":int(m[1]),"clusters":int(m[2]),"rep":int(m[3]),
        "barrier_us":float(m[4]),"dsm_us":float(m[5]),"global_handoff_us":float(m[6]),"direct_global_us":float(m[7])})
  dm=next((DEVICE.fullmatch(x) for x in run.stdout.splitlines() if DEVICE.fullmatch(x)),None)
  validations=[]
  for line in run.stdout.splitlines():
    if m:=VALIDATE.fullmatch(line): validations.append({"cluster_size":int(m[1]),"clusters":int(m[2]),"exact":bool(int(m[3])),"finite":bool(int(m[4]))})
  if not rows or dm is None or len(validations)!=3: raise RuntimeError("unparsed output\n"+run.stdout)
  medians={}
  for cs in sorted({r["cluster_size"] for r in rows}):
    subset=[r for r in rows if r["cluster_size"]==cs]
    medians[str(cs)]={k:statistics.median(r[k] for r in subset) for k in
      ("barrier_us","dsm_us","global_handoff_us","direct_global_us")}
    medians[str(cs)]["dsm_minus_global_us"]=medians[str(cs)]["dsm_us"]-medians[str(cs)]["global_handoff_us"]
    medians[str(cs)]["dsm_minus_direct_us"]=medians[str(cs)]["dsm_us"]-medians[str(cs)]["direct_global_us"]
  result={"schema":"tinygrad.nv_cluster_projection_service_gate.v1",
    "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
    "device":{"name":dm[1],"compute_capability":f"{dm[2]}.{dm[3]}","sm_count":int(dm[4]),"cluster_launch":bool(int(dm[5]))},
    "shape":{"activation_width":4096,"threads_per_block":128},"validations":validations,"medians":medians,"samples":rows,
    "stdout":run.stdout,"ptxas":cp.stderr}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
