#!/usr/bin/env python3
"""Exact full-grid Flash score with last-split-CTA-owned per-head combine."""
from __future__ import annotations
import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel, flash_vec_llama_score_pv_kernel
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.nv_segmented_flash_o_complete_span import _render, _preambles

H,HKV,HD,S,MAXC,TC,W=32,8,128,6,1024,513,130
NVCC="/usr/local/cuda-13.2/bin/nvcc"

def source():
  p=UOp.placeholder
  sn,ss=_render(flash_vec_llama_score_pv_kernel(HD,H,HKV,MAXC,S,UOp.const(dtypes.int,TC),wide_kv=True,wide_q=False,
    token_bound=768,v_pipeline_tail=1)(p((H*S*W,),dtypes.float32,0),p((H*HD,),dtypes.float32,1),p((2*HKV*MAXC*HD//2,),dtypes.uint32,2)))
  cn,cs=_render(flash_fused_gmax_combine_kernel(HD,H,S,output_fp16=True,lane_width=128)(
    p((H*HD,),dtypes.float16,0),p((H*S*W,),dtypes.float32,1)))
  body=ss[ss.index('extern "C"'):]
  fused=sn+"_lastcta"
  body=body.replace(sn,fused,1).replace("unsigned int* data2_1048576) {",
    "unsigned int* data2_1048576, half* final_out, unsigned int* counters) {",1)
  hook=r'''
  __shared__ int tg_last;
  __syncthreads();
  if (lidx0 == 0 && lidx1 == 0) {
    __threadfence();
    tg_last = (atomicAdd(counters + gidx1, 1u) == 5u);
  }
  __syncthreads();
  if (tg_last) {
    int tg_lane = lidx1 * 32 + lidx0;
    float tg_max = -1e30f;
    #pragma unroll
    for (int tg_s=0; tg_s<6; tg_s++) tg_max = fmaxf(tg_max, data0_24960[(gidx1*6+tg_s)*130+129]);
    float tg_acc=0.0f, tg_den=0.0f;
    #pragma unroll
    for (int tg_s=0; tg_s<6; tg_s++) {
      float tg_w=exp2f((data0_24960[(gidx1*6+tg_s)*130+129]-tg_max)*1.4426950408889634f);
      tg_acc += tg_w * data0_24960[(gidx1*6+tg_s)*130+tg_lane];
      tg_den += tg_w * data0_24960[(gidx1*6+tg_s)*130+128];
    }
    final_out[gidx1*128+tg_lane] = (half)(tg_acc/tg_den);
    __syncthreads();
    if (tg_lane == 0) counters[gidx1]=0u;
  }
'''
  body=body.rsplit('}',1)[0]+hook+'}\n'
  return _preambles(ss,cs)+ss[ss.index('extern "C"'):]+cs[cs.index('extern "C"'):]+body, sn,cn,fused

HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define ROT 16
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
__SRC__
static void control(half*out,float*part,float*q,unsigned int*cache){__S__<<<dim3(6,32,1),dim3(32,4,1)>>>(part,q,cache);__C__<<<32,128>>>(out,part);}
static void candidate(half*out,float*part,float*q,unsigned int*cache,unsigned int*ctr){__F__<<<dim3(6,32,1),dim3(32,4,1)>>>(part,q,cache,out,ctr);}
static float timed(int arm,half*out,float*part,float*q,unsigned int*cache,unsigned int*ctr,int n){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ea");ck(cudaEventCreate(&b),"eb");ck(cudaEventRecord(a),"ra");for(int i=0;i<n;i++)arm?candidate(out,part,q,cache,ctr):control(out,part,q,cache);ck(cudaEventRecord(b),"rb");ck(cudaEventSynchronize(b),"sb");float ms;ck(cudaEventElapsedTime(&ms,a,b),"el");return ms*1000/n;}
int main(int ac,char**av){int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);float *q,*pa,*pb;half *oa,*ob;unsigned int *cache,*ctr;ck(cudaMalloc(&q,4096*4),"q");ck(cudaMalloc(&pa,24960*4),"pa");ck(cudaMalloc(&pb,24960*4),"pb");ck(cudaMalloc(&oa,4096*2),"oa");ck(cudaMalloc(&ob,4096*2),"ob");ck(cudaMalloc(&cache,(size_t)2*ROT*1048576*4),"cache");unsigned int* cacheb=cache+(size_t)ROT*1048576;ck(cudaMalloc(&ctr,32*4),"ctr");std::vector<float> hq(4096);for(int i=0;i<4096;i++)hq[i]=float((i*17%127)-63)/256;ck(cudaMemcpy(q,hq.data(),4096*4,cudaMemcpyHostToDevice),"qcopy");ck(cudaMemset(cache,0x30,(size_t)2*ROT*1048576*4),"cacheinit");ck(cudaMemset(ctr,0,128),"ctrinit");control(oa,pa,q,cache);candidate(ob,pb,q,cacheb,ctr);ck(cudaDeviceSynchronize(),"sync");std::vector<unsigned char>a(8192),b(8192);ck(cudaMemcpy(a.data(),oa,8192,cudaMemcpyDeviceToHost),"a");ck(cudaMemcpy(b.data(),ob,8192,cudaMemcpyDeviceToHost),"b");printf("bitwise=%d\n",memcmp(a.data(),b.data(),8192)==0);for(int r=0;r<reps;r++){float ah,bh;if(r&1){bh=timed(1,ob,pb,q,cacheb,ctr,hp);ah=timed(0,oa,pa,q,cache,ctr,hp);}else{ah=timed(0,oa,pa,q,cache,ctr,hp);bh=timed(1,ob,pb,q,cacheb,ctr,hp);}double acold=0,bcold=0;for(int i=0;i<cp;i++){int rot=(r*cp+i)%ROT;unsigned int*ca=cache+(size_t)rot*1048576;unsigned int*cb=cacheb+(size_t)rot*1048576;if((r+i)&1){bcold+=timed(1,ob,pb,q,cb,ctr,1);acold+=timed(0,oa,pa,q,ca,ctr,1);}else{acold+=timed(0,oa,pa,q,ca,ctr,1);bcold+=timed(1,ob,pb,q,cb,ctr,1);}}printf("rep=%d control_hot=%.6f candidate_hot=%.6f control_cold=%.6f candidate_cold=%.6f\n",r,ah,bh,acold/cp,bcold/cp);}return 0;}
'''

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reps',type=int,default=9);ap.add_argument('--hot',type=int,default=200);ap.add_argument('--cold',type=int,default=16);a=ap.parse_args()
  src,s,c,f=source();cu=HARNESS.replace('__SRC__',src).replace('__S__',s).replace('__C__',c).replace('__F__',f)
  with tempfile.TemporaryDirectory(prefix='lastcta_') as td:
    p=Path(td);(p/'g.cu').write_text(cu);b=subprocess.run([NVCC,'-O3','-arch=sm_120a','--ptxas-options=-v',str(p/'g.cu'),'-o',str(p/'g')],capture_output=True,text=True)
    if b.returncode: raise RuntimeError(b.stderr[-12000:])
    r=subprocess.run([str(p/'g'),str(a.hot),str(a.cold),str(a.reps)],capture_output=True,text=True)
  print(r.stdout);rows=[]
  for m in re.finditer(r'rep=(\d+) control_hot=([0-9.]+) candidate_hot=([0-9.]+) control_cold=([0-9.]+) candidate_cold=([0-9.]+)',r.stdout): rows.append([float(x) for x in m.groups()[1:]])
  med=[statistics.median(x[i] for x in rows) for i in range(4)]
  out={'schema':'tinygrad.nv_flash_last_cta_combine.v1','bitwise':'bitwise=1' in r.stdout,'medians':dict(zip(('control_hot','candidate_hot','control_cold','candidate_cold'),med)),'cold_recovery_us':med[2]-med[3],'samples':rows,'ptxas':b.stderr}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['bitwise'] else 5
if __name__=='__main__':raise SystemExit(main())
