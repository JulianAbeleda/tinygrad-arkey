#!/usr/bin/env python3
"""Standalone generic M/N/K single-main-launch FP16 stream-K microgate."""
from __future__ import annotations
import argparse, json, pathlib, re, shutil, subprocess, tempfile

SRC=r'''
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define M __M__
#define N __N__
#define K __K__
#define TM 128
#define TN 128
#define TILES ((M/TM)*(N/TN))
#define STREAM_BLOCKS __SB__
#define ELEMS (TM*TN)
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
extern "C" __global__ __launch_bounds__(256) void control32(const half* a,const half* b,float* out){
 int tile=blockIdx.x, mt=tile/(N/TN),nt=tile%(N/TN);
 for(int z=threadIdx.x;z<ELEMS;z+=blockDim.x){int mi=z/TN,ni=z%TN;float v=0;
  for(int k=0;k<K;k++)v=fmaf(__half2float(a[(mt*TM+mi)*K+k]),__half2float(b[(nt*TN+ni)*K+k]),v);
  out[(mt*TM+mi)*N+nt*TN+ni]=v;}}
extern "C" __global__ __launch_bounds__(256) void stream_main(const half* a,const half* b,float* partial){
 int tile=blockIdx.x%TILES, split=blockIdx.x/TILES, parts=(STREAM_BLOCKS-1-tile)/TILES+1;
 int k0=(K*split)/parts,k1=(K*(split+1))/parts,mt=tile/(N/TN),nt=tile%(N/TN);
 for(int z=threadIdx.x;z<ELEMS;z+=blockDim.x){int mi=z/TN,ni=z%TN;float v=0;
  for(int k=k0;k<k1;k++)v=fmaf(__half2float(a[(mt*TM+mi)*K+k]),__half2float(b[(nt*TN+ni)*K+k]),v);
  partial[blockIdx.x*ELEMS+z]=v;}}
extern "C" __global__ __launch_bounds__(256) void stream_fixup(const float* partial,float*out){
 int tile=blockIdx.x,mt=tile/(N/TN),nt=tile%(N/TN),parts=(STREAM_BLOCKS-1-tile)/TILES+1;
 for(int z=threadIdx.x;z<ELEMS;z+=blockDim.x){float v=0;for(int s=0;s<parts;s++)v+=partial[(tile+s*TILES)*ELEMS+z];
  int mi=z/TN,ni=z%TN;out[(mt*TM+mi)*N+nt*TN+ni]=v;}}
static float once(bool stream,const half*a,const half*b,float*p,float*out){cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
 if(stream){stream_main<<<STREAM_BLOCKS,256>>>(a,b,p);stream_fixup<<<TILES,256>>>(p,out);}else control32<<<TILES,256>>>(a,b,out);
 cudaEventRecord(e);ck(cudaEventSynchronize(e),"event");float ms;cudaEventElapsedTime(&ms,s,e);cudaEventDestroy(s);cudaEventDestroy(e);return ms*1000;}
int main(int ac,char**av){int reps=ac>1?atoi(av[1]):9;half *a,*b;float *p,*c,*o;ck(cudaMalloc(&a,M*K*2),"a");ck(cudaMalloc(&b,N*K*2),"b");
 ck(cudaMalloc(&p,(size_t)STREAM_BLOCKS*ELEMS*4),"p");ck(cudaMalloc(&c,M*N*4),"c");ck(cudaMalloc(&o,M*N*4),"o");
 unsigned short*ha=(unsigned short*)malloc(M*K*2),*hb=(unsigned short*)malloc(N*K*2);for(int i=0;i<M*K;i++)ha[i]=0x3400+(i%17);for(int i=0;i<N*K;i++)hb[i]=0x3000+(i%13);
 ck(cudaMemcpy(a,ha,M*K*2,cudaMemcpyHostToDevice),"ca");ck(cudaMemcpy(b,hb,N*K*2,cudaMemcpyHostToDevice),"cb");once(0,a,b,p,c);once(1,a,b,p,o);
 float *hc=(float*)malloc(M*N*4),*ho=(float*)malloc(M*N*4);ck(cudaMemcpy(hc,c,M*N*4,cudaMemcpyDeviceToHost),"oc");ck(cudaMemcpy(ho,o,M*N*4,cudaMemcpyDeviceToHost),"oo");double ma=0,me=0;int finite=1;for(int i=0;i<M*N;i++){double d=fabs((double)hc[i]-ho[i]);if(d>ma)ma=d;me+=d;if(!isfinite(ho[i]))finite=0;}me/=M*N;
 printf("correct finite=%d max_abs=%.9g mean_abs=%.9g tiles=%d stream_blocks=%d\n",finite,ma,me,TILES,STREAM_BLOCKS);
 for(int r=0;r<reps;r++){float x=once(0,a,b,p,c),y=once(1,a,b,p,o),z=once(0,a,b,p,c);printf("rep=%d control_a_us=%.3f stream_us=%.3f control_c_us=%.3f\n",r,x,y,z);}return 0;}
'''

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--m',type=int,default=512);ap.add_argument('--n',type=int,default=1024);ap.add_argument('--k',type=int,default=4096);ap.add_argument('--stream-blocks',type=int,default=170);ap.add_argument('--reps',type=int,default=9);ap.add_argument('--out',required=True);ap.add_argument('--artifacts',default='');a=ap.parse_args()
 if a.m%128 or a.n%128: raise SystemExit('M/N must be multiples of 128')
 src=SRC.replace('__M__',str(a.m)).replace('__N__',str(a.n)).replace('__K__',str(a.k)).replace('__SB__',str(a.stream_blocks))
 with tempfile.TemporaryDirectory(prefix='nv_kv_streamk_') as td:
  cu=pathlib.Path(td)/'gate.cu';binp=pathlib.Path(td)/'gate';cu.write_text(src)
  cp=subprocess.run(['/usr/local/cuda-13.2/bin/nvcc','-arch=sm_120a','-O3','--use_fast_math','--ptxas-options=-v',str(cu),'-o',str(binp)],capture_output=True,text=True)
  if cp.returncode: raise SystemExit(cp.stderr)
  sass=''
  if a.artifacts:
   ad=pathlib.Path(a.artifacts);ad.mkdir(parents=True,exist_ok=True);shutil.copy2(cu,ad/'gate.cu');shutil.copy2(binp,ad/'gate')
   cub=ad/'gate.cubin';subprocess.run(['/usr/local/cuda-13.2/bin/nvcc','-arch=sm_120a','-O3','--use_fast_math','-cubin',str(cu),'-o',str(cub)],check=True)
   sd=subprocess.run(['/usr/local/cuda-13.2/bin/cuobjdump','--dump-sass',str(cub)],capture_output=True,text=True);(ad/'gate.sass').write_text(sd.stdout+sd.stderr);sass=str(ad/'gate.sass')
  run=subprocess.run([str(binp),str(a.reps)],capture_output=True,text=True,check=True)
 rows=[]
 for line in run.stdout.splitlines():
  m=re.match(r'rep=(\d+) control_a_us=([\d.]+) stream_us=([\d.]+) control_c_us=([\d.]+)',line)
  if m: rows.append({'rep':int(m[1]),'control_a_us':float(m[2]),'stream_us':float(m[3]),'control_c_us':float(m[4])})
 q=re.search(r'correct finite=(\d+) max_abs=([\deE+.-]+) mean_abs=([\deE+.-]+)',run.stdout)
 rec={'schema':'tinygrad.nv_prefill_kv_native_streamk.v1','shape':{'m':a.m,'n':a.n,'k':a.k},'stream_blocks':a.stream_blocks,'fixup_blocks':a.m//128*a.n//128,'rows':rows,'correctness':{'finite':bool(int(q[1])),'max_abs':float(q[2]),'mean_abs':float(q[3])},'ptxas':cp.stderr.strip(),'sass_path':sass,'stdout':run.stdout}
 pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
