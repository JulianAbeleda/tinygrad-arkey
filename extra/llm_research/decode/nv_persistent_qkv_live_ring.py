#!/usr/bin/env python3
"""Native live-publication gate for a persistent Q4_K Q/K/V row service.

Research only.  A resident set of CTAs is launched before the host publishes an
epoch through mapped memory.  Warps then claim Q/K/V rows from a system-scope
atomic counter.  The arithmetic body is mechanically derived from the same
tinygrad-generated one-row kernel used by the standalone control.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.q4k_striped_projection_group_microgate import _render

CUDA_BIN="/usr/local/cuda-13.2/bin"
Q_ROWS,KV_ROWS,K,TOTAL_ROWS=4096,1024,4096,6144
Q_WORDS,KV_WORDS=2359296,589824
GROUP_WORDS=3538944

def row_body(src:str) -> str:
  # _render()[3] is the phased one-row-per-CTA body.  Keep its arithmetic
  # verbatim; only turn blockIdx/threadIdx into explicit row/lane arguments.
  src=src.replace('extern "C" __global__ void __launch_bounds__(32) q4k_projection_group_one_task_phased_6144_4096(float* data0_6144, unsigned int* data1_3538944, half* data2_4096) {',
                  '__device__ __forceinline__ void q4k_row(float* data0_6144, unsigned int* data1_3538944, half* data2_4096, int gidx0, int lidx0) {')
  src=src.replace('  int gidx0 = blockIdx.x; /* 6144 */\n','').replace('  int lidx0 = threadIdx.x; /* 32 */\n','')
  if '__device__ __forceinline__ void q4k_row' not in src or 'blockIdx.x' in src or 'threadIdx.x' in src:
    raise RuntimeError('failed to derive row body')
  return src

HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#define TOTAL_ROWS 6144
#define GROUP_WORDS 3538944
#define ROTATIONS 16
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }
__ROW_BODY__

struct alignas(64) Control { unsigned epoch, next, done, abort, entered, exited; unsigned long long deadline; };

extern "C" __global__ __launch_bounds__(256) void persistent_qkv_live(float* out, unsigned int* words, half* x, Control* c, unsigned* gpu_complete) {
  const int lane=threadIdx.x&31;
  const unsigned consumer_blocks=gridDim.x-1;
  if(blockIdx.x==0) { if(threadIdx.x==0) { unsigned long long s=clock64(); while(cuda::atomic_ref<unsigned,cuda::thread_scope_device>(c->entered).load(cuda::memory_order_acquire)<consumer_blocks) { if(clock64()-s>c->deadline){c->abort=3;return;} __nanosleep(64); } cuda::atomic_ref<unsigned,cuda::thread_scope_device>(c->epoch).store(1u,cuda::memory_order_release); } return; }
  if(threadIdx.x==0) atomicAdd(&c->entered,1u);
  unsigned long long wait_start=clock64();
  unsigned epoch;
  do {
    epoch=cuda::atomic_ref<unsigned,cuda::thread_scope_device>(c->epoch).load(cuda::memory_order_acquire);
    if(cuda::atomic_ref<unsigned,cuda::thread_scope_device>(c->abort).load(cuda::memory_order_relaxed)) return;
    if(clock64()-wait_start>c->deadline) { if(threadIdx.x==0) atomicExch(&c->abort,2u); return; }
    __nanosleep(64);
  } while(epoch==0);
  // Epoch publication is system-scoped, row assignment is deterministic and
  // GPU-local.  A system atomic per row is a PCIe-coherence benchmark, not a
  // persistent GEMV service design.
  const unsigned warp=(blockIdx.x-1)*(blockDim.x/32)+(threadIdx.x/32);
  const unsigned warps=consumer_blocks*(blockDim.x/32);
  for(unsigned row=warp;row<TOTAL_ROWS;row+=warps) {
    q4k_row(out,words,x,(int)row,lane);
  }
  __syncthreads();
  if(threadIdx.x==0) {
    unsigned old=atomicAdd(gpu_complete,1u);
    if(old+1u==consumer_blocks) { c->exited=consumer_blocks; cuda::atomic_ref<unsigned,cuda::thread_scope_device>(c->done).store(epoch,cuda::memory_order_release); }
  }
}

extern "C" __global__ void __launch_bounds__(32) standalone_qkv(float* out,unsigned int* words,half* x) {
  q4k_row(out,words,x,(int)blockIdx.x,(int)threadIdx.x);
}

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static void legal_words(unsigned* w,size_t n) {
  for(size_t i=0;i<n;i++) w[i]=(unsigned)((i*2654435761u)^0x9e3779b9u);
  for(size_t base=0;base<n;base+=36) { w[base]=0x30003000u; w[base+1]=0x10101010u; }
}
static double one_standalone(float* out,unsigned* words,half* x,cudaStream_t s) {
  cudaEvent_t a,b; ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");
  ck(cudaEventRecord(a,s),"record"); standalone_qkv<<<TOTAL_ROWS,32,0,s>>>(out,words,x);ck(cudaEventRecord(b,s),"record");ck(cudaEventSynchronize(b),"sync");
  float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0;
}
static double one_persistent(float* out,unsigned* words,half* x,Control* c,unsigned* gpu_complete,int blocks,cudaStream_t consumer,cudaStream_t producer) {
  int khz=0;ck(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,0),"clock");
  Control h={};h.deadline=(unsigned long long)khz*2000ull;
  ck(cudaMemcpyAsync(c,&h,sizeof(h),cudaMemcpyHostToDevice,producer),"control reset");ck(cudaMemsetAsync(gpu_complete,0,sizeof(unsigned),producer),"completion reset");ck(cudaStreamSynchronize(producer),"reset sync");
  cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");ck(cudaEventRecord(a,consumer),"record");persistent_qkv_live<<<blocks+1,256,0,consumer>>>(out,words,x,c,gpu_complete);ck(cudaEventRecord(b,consumer),"record");ck(cudaEventSynchronize(b),"sync");
  float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");ck(cudaMemcpy(&h,c,sizeof(h),cudaMemcpyDeviceToHost),"control read");if(h.done!=1||h.abort){fprintf(stderr,"watchdog done=%u abort=%u exited=%u\n",h.done,h.abort,h.exited);exit(7);}cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0;
}
int main(int argc,char**argv){
  int reps=argc>1?atoi(argv[1]):9;cudaDeviceProp p;ck(cudaGetDeviceProperties(&p,0),"props");int blocks=2*p.multiProcessorCount-1;
  printf("sm=%d resident_blocks=%d warps=%d\n",p.multiProcessorCount,blocks,blocks*8);
  float *a,*b;unsigned* groups,*gpu_complete;half*x;Control*c;ck(cudaMalloc(&a,TOTAL_ROWS*4),"out");ck(cudaMalloc(&b,TOTAL_ROWS*4),"out");
  ck(cudaMalloc(&groups,(size_t)ROTATIONS*GROUP_WORDS*4),"groups");ck(cudaMalloc(&x,4096*2),"x");ck(cudaMalloc(&gpu_complete,sizeof(unsigned)),"gpu completion");ck(cudaMalloc(&c,sizeof(Control)),"control");
  unsigned* hw=(unsigned*)malloc((size_t)ROTATIONS*GROUP_WORDS*4);half*hx=(half*)malloc(4096*2);legal_words(hw,(size_t)ROTATIONS*GROUP_WORDS);for(int i=0;i<4096;i++)hx[i]=__float2half(((i%257)-128)*.03125f);
  ck(cudaMemcpy(groups,hw,(size_t)ROTATIONS*GROUP_WORDS*4,cudaMemcpyHostToDevice),"weights");ck(cudaMemcpy(x,hx,4096*2,cudaMemcpyHostToDevice),"x");free(hw);free(hx);cudaStream_t s,producer;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");int least,greatest;ck(cudaDeviceGetStreamPriorityRange(&least,&greatest),"priorities");ck(cudaStreamCreateWithPriority(&producer,cudaStreamNonBlocking,greatest),"producer");
  standalone_qkv<<<TOTAL_ROWS,32,0,s>>>(a,groups,x);one_persistent(b,groups,x,c,gpu_complete,blocks,s,producer);ck(cudaStreamSynchronize(s),"warm");
  float *ha=(float*)malloc(TOTAL_ROWS*4),*hb=(float*)malloc(TOTAL_ROWS*4);ck(cudaMemcpy(ha,a,TOTAL_ROWS*4,cudaMemcpyDeviceToHost),"ref");ck(cudaMemcpy(hb,b,TOTAL_ROWS*4,cudaMemcpyDeviceToHost),"got");
  int exact=memcmp(ha,hb,TOTAL_ROWS*4)==0,finite=1;for(int i=0;i<TOTAL_ROWS;i++)if(!isfinite(ha[i])||!isfinite(hb[i]))finite=0;printf("bitwise=%d finite=%d\n",exact,finite);free(ha);free(hb);
  for(int r=0;r<reps;r++){double sh=0,ph=0,sc=0,pc=0;for(int i=0;i<32;i++){sh+=one_standalone(a,groups,x,s);ph+=one_persistent(b,groups,x,c,gpu_complete,blocks,s,producer);}for(int i=0;i<16;i++){unsigned* g=groups+(size_t)i*GROUP_WORDS;sc+=one_standalone(a,g,x,s);pc+=one_persistent(b,g,x,c,gpu_complete,blocks,s,producer);}printf("rep=%d standalone_hot=%.6f persistent_hot=%.6f standalone_cold=%.6f persistent_cold=%.6f\n",r,sh/32,ph/32,sc/16,pc/16);}
  return exact&&finite?0:5;
}
'''

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=9);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--artifact-dir',type=Path);args=ap.parse_args()
  src=HARNESS.replace('__ROW_BODY__',row_body(_render()[3]))
  with tempfile.TemporaryDirectory(prefix='persistent_qkv_') as td:
    cu,exe=Path(td)/'gate.cu',Path(td)/'gate';cu.write_text(src);env={**os.environ,'PATH':CUDA_BIN+':'+os.environ.get('PATH','')}
    cp=subprocess.run(['nvcc','-arch=sm_120a','-O3','-std=c++20','--ptxas-options=-v',str(cu),'-o',str(exe)],capture_output=True,text=True,env=env)
    if cp.returncode: raise RuntimeError(cp.stderr[-16000:])
    if args.artifact_dir: args.artifact_dir.mkdir(parents=True,exist_ok=True);shutil.copy2(cu,args.artifact_dir/'gate.cu');shutil.copy2(exe,args.artifact_dir/'gate')
    run=subprocess.run([str(exe),str(args.reps)],capture_output=True,text=True,timeout=600)
    if run.returncode: raise RuntimeError(run.stdout+'\n'+run.stderr)
  rows=[]
  for line in run.stdout.splitlines():
    m=re.match(r'rep=(\d+) standalone_hot=([0-9.]+) persistent_hot=([0-9.]+) standalone_cold=([0-9.]+) persistent_cold=([0-9.]+)',line)
    if m: rows.append(dict(rep=int(m[1]),standalone_hot_us=float(m[2]),persistent_hot_us=float(m[3]),standalone_cold_us=float(m[4]),persistent_cold_us=float(m[5])))
  med={k:statistics.median(r[k] for r in rows) for k in rows[0] if k!='rep'}
  out={'schema':'tinygrad.nv_persistent_qkv_live_ring.v1','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'exact':('bitwise=1 finite=1' in run.stdout),'median':med,'samples':rows,'stdout':run.stdout,'ptxas':cp.stderr}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))

if __name__=='__main__':main()
