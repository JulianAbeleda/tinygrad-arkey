#!/usr/bin/env python3
"""Exact cluster-geometry gate for the installed 4096x4096 Q4_K O body.

The control is the installed one-row/one-warp vector-load kernel.  The
candidate keeps the same 4096 logical warps and material bytes, but packs them
as four 128-thread CTAs per hardware cluster.  Each CTA owns one installed
eight-lane block group for sixteen rows; rank zero reconstructs the installed
32-lane XOR reduction order from DSM.  This isolates service geometry without
changing arithmetic, representation, Flash, or production routing.
"""
from __future__ import annotations

import argparse, json, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue,q4k_g3_lanemap_gemv_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp

NVCC="/usr/local/cuda-13.2/bin/nvcc"
ROWS=K=4096
WORDS=ROWS*(K//256)*36
ROTATIONS=16

def installed_source()->tuple[str,str]:
  p=UOp.placeholder
  ast=q4k_g3_lanemap_gemv_kernel(ROWS,K,epilogue=Q4KGEMVEpilogue("residual_add"),load_style="vector")(
    p((ROWS,),dtypes.float32,0),p((WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2),p((ROWS,),dtypes.float32,3))
  prg=to_program(ast,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False))
  return prg.arg.name,next(x.arg for x in prg.src if x.op is Ops.SOURCE)

def cluster_source(name:str,src:str,cluster_size:int)->tuple[str,str]:
  if cluster_size not in (2,4):raise ValueError("cluster size must be 2 or 4")
  lanes_per_rank=32//cluster_size
  rows_per_cluster=128//lanes_per_rank
  batches=rows_per_cluster//4
  cname=name+f"_cluster{cluster_size}_r{rows_per_cluster}"
  pre=src[:src.index('extern "C"')]
  body=src[src.index('{',src.index('extern "C"'))+1:src.rfind('}')]
  body=body.replace('  int gidx0 = blockIdx.x; /* 4096 */\n  int lidx0 = threadIdx.x; /* 32 */\n','',1)
  body=body[:body.index('  float val13 = ')]
  if '(*(buf0+0))' not in body or 'gidx0' not in body or 'lidx0' not in body:raise RuntimeError("failed to derive installed lane body")
  kernel=f'''extern "C" __global__ __launch_bounds__(128) __cluster_dims__({cluster_size},1,1) void {cname}(float* data0_4096, unsigned int* data1_2359296, half* data2_4096, float* data3_4096) {{
  cg::cluster_group cluster=cg::this_cluster();
  int cluster_id=blockIdx.x/{cluster_size};
  int row_local=threadIdx.x/{lanes_per_rank};
  int gidx0=cluster_id*{rows_per_cluster}+row_local;
  int lidx0=cluster.block_rank()*{lanes_per_rank}+(threadIdx.x%{lanes_per_rank});
{body}
  __shared__ float partial[128];
  partial[threadIdx.x]=(*(buf0+0));
  cluster.sync();
  if(cluster.block_rank()==0) {{
    int lane=threadIdx.x&31, row_slot=threadIdx.x>>5;
    #pragma unroll
    for(int batch=0;batch<{batches};batch++) {{
      int rlocal=batch*4+row_slot;
      float*remote=cluster.map_shared_rank(partial,lane/{lanes_per_rank});
      float v=remote[rlocal*{lanes_per_rank}+(lane%{lanes_per_rank})];
      v=v+__shfl_xor_sync(0xffffffffu,v,16);
      v=v+__shfl_xor_sync(0xffffffffu,v,8);
      v=v+__shfl_xor_sync(0xffffffffu,v,4);
      v=v+__shfl_xor_sync(0xffffffffu,v,2);
      v=v+__shfl_xor_sync(0xffffffffu,v,1);
      if(lane==0) {{int row=cluster_id*{rows_per_cluster}+rlocal;data0_4096[row]=v+data3_4096[row];}}
    }}
  }}
  cluster.sync();
}}
'''
  return cname,pre+'#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n'+kernel

HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define ROWS 4096
#define WORDS 2359296
#define ROTATIONS 16
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static void legal_words(unsigned*w,size_t n,unsigned seed){for(size_t i=0;i<n;i++)w[i]=(unsigned)((i*2654435761u)^(seed*2246822519u)^0x9e3779b9u);for(size_t b=0;b<n;b+=36){w[b]=0x30003000u+(seed&3u);w[b+1]=0x10101010u;}}
template<class F>static double timed(F launch,int iters,cudaStream_t s){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ea");ck(cudaEventCreate(&b),"eb");for(int i=0;i<100;i++)launch();ck(cudaEventRecord(a,s),"ra");for(int i=0;i<iters;i++)launch();ck(cudaEventRecord(b,s),"rb");ck(cudaEventSynchronize(b),"sb");float ms;ck(cudaEventElapsedTime(&ms,a,b),"el");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0/iters;}
static double one_control(float*out,unsigned*w,half*x,float*res,cudaStream_t s){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ea");ck(cudaEventCreate(&b),"eb");ck(cudaEventRecord(a,s),"ra");__CONTROL__<<<ROWS,32,0,s>>>(out,w,x,res);ck(cudaEventRecord(b,s),"rb");ck(cudaEventSynchronize(b),"sb");float ms;ck(cudaEventElapsedTime(&ms,a,b),"el");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0;}
static double one_candidate2(float*out,unsigned*w,half*x,float*res,cudaStream_t s){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ea");ck(cudaEventCreate(&b),"eb");ck(cudaEventRecord(a,s),"ra");__CANDIDATE2__<<<1024,128,0,s>>>(out,w,x,res);ck(cudaEventRecord(b,s),"rb");ck(cudaEventSynchronize(b),"sb");float ms;ck(cudaEventElapsedTime(&ms,a,b),"el");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0;}
static double one_candidate4(float*out,unsigned*w,half*x,float*res,cudaStream_t s){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ea");ck(cudaEventCreate(&b),"eb");ck(cudaEventRecord(a,s),"ra");__CANDIDATE4__<<<1024,128,0,s>>>(out,w,x,res);ck(cudaEventRecord(b,s),"rb");ck(cudaEventSynchronize(b),"sb");float ms;ck(cudaEventElapsedTime(&ms,a,b),"el");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0;}
int main(int argc,char**argv){int reps=argc>1?atoi(argv[1]):9,iters=argc>2?atoi(argv[2]):1000;float*a,*b,*res;unsigned*w;half*x;cudaStream_t s;ck(cudaMalloc(&a,ROWS*4),"a");ck(cudaMalloc(&b,ROWS*4),"b");ck(cudaMalloc(&w,(size_t)ROTATIONS*WORDS*4),"w");ck(cudaMalloc(&x,ROWS*2),"x");ck(cudaMalloc(&res,ROWS*4),"res");ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");
  std::vector<unsigned>hw((size_t)ROTATIONS*WORDS);std::vector<half>hx(ROWS);std::vector<float>hr(ROWS);for(int q=0;q<ROTATIONS;q++)legal_words(hw.data()+(size_t)q*WORDS,WORDS,17+q);for(int i=0;i<ROWS;i++){hx[i]=__float2half(((i%257)-128)*.03125f);hr[i]=((i%113)-56)*.0078125f;}ck(cudaMemcpy(w,hw.data(),hw.size()*4,cudaMemcpyHostToDevice),"weights");ck(cudaMemcpy(x,hx.data(),ROWS*2,cudaMemcpyHostToDevice),"x");ck(cudaMemcpy(res,hr.data(),ROWS*4,cudaMemcpyHostToDevice),"res");
  bool exact2=true,exact4=true,finite2=true,finite4=true;std::vector<float>ha(ROWS),hb(ROWS);for(int q: {0,5,11}){unsigned*wp=w+(size_t)q*WORDS;__CONTROL__<<<ROWS,32,0,s>>>(a,wp,x,res);ck(cudaStreamSynchronize(s),"control validate");ck(cudaMemcpy(ha.data(),a,ROWS*4,cudaMemcpyDeviceToHost),"ca");__CANDIDATE2__<<<1024,128,0,s>>>(b,wp,x,res);ck(cudaStreamSynchronize(s),"candidate2 validate");ck(cudaMemcpy(hb.data(),b,ROWS*4,cudaMemcpyDeviceToHost),"cb2");exact2&=memcmp(ha.data(),hb.data(),ROWS*4)==0;for(int i=0;i<ROWS;i++)finite2&=isfinite(ha[i])&&isfinite(hb[i]);__CANDIDATE4__<<<1024,128,0,s>>>(b,wp,x,res);ck(cudaStreamSynchronize(s),"candidate4 validate");ck(cudaMemcpy(hb.data(),b,ROWS*4,cudaMemcpyDeviceToHost),"cb4");exact4&=memcmp(ha.data(),hb.data(),ROWS*4)==0;for(int i=0;i<ROWS;i++)finite4&=isfinite(ha[i])&&isfinite(hb[i]);}printf("validate exact2=%d finite2=%d exact4=%d finite4=%d\n",(int)exact2,(int)finite2,(int)exact4,(int)finite4);
  for(int r=0;r<reps;r++){double ch=timed([&](){__CONTROL__<<<ROWS,32,0,s>>>(a,w,x,res);},iters,s);double k2h=timed([&](){__CANDIDATE2__<<<1024,128,0,s>>>(b,w,x,res);},iters,s);double k4h=timed([&](){__CANDIDATE4__<<<1024,128,0,s>>>(b,w,x,res);},iters,s);double cc=0,k2c=0,k4c=0;for(int q=0;q<ROTATIONS;q++){cc+=one_control(a,w+(size_t)q*WORDS,x,res,s);k2c+=one_candidate2(b,w+(size_t)q*WORDS,x,res,s);k4c+=one_candidate4(b,w+(size_t)q*WORDS,x,res,s);}printf("sample rep=%d control_hot_us=%.6f candidate2_hot_us=%.6f candidate4_hot_us=%.6f control_cold_us=%.6f candidate2_cold_us=%.6f candidate4_cold_us=%.6f\n",r,ch,k2h,k4h,cc/ROTATIONS,k2c/ROTATIONS,k4c/ROTATIONS);}
  return exact2&&finite2&&exact4&&finite4?0:5;
}
'''

SAMPLE=re.compile(r"sample rep=(\d+) control_hot_us=([0-9.]+) candidate2_hot_us=([0-9.]+) candidate4_hot_us=([0-9.]+) control_cold_us=([0-9.]+) candidate2_cold_us=([0-9.]+) candidate4_cold_us=([0-9.]+)")

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--reps",type=int,default=9);ap.add_argument("--iters",type=int,default=1000);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--artifact-dir",type=Path);args=ap.parse_args()
  control,src=installed_source();candidate2,csrc2=cluster_source(control,src,2);candidate4,csrc4=cluster_source(control,src,4)
  full=src+'\n'+csrc2[csrc2.index('#include <cooperative_groups.h>'):]+'\n'+csrc4[csrc4.index('extern "C"'):]+'\n'+HARNESS.replace('__CONTROL__',control).replace('__CANDIDATE2__',candidate2).replace('__CANDIDATE4__',candidate4)
  with tempfile.TemporaryDirectory(prefix="nv_q4k_o_cluster_") as td:
    cu,exe=Path(td)/"gate.cu",Path(td)/"gate";cu.write_text(full)
    cp=subprocess.run([NVCC,"-arch=sm_120a","-O3","-std=c++20","--ptxas-options=-v",str(cu),"-o",str(exe)],capture_output=True,text=True)
    if cp.returncode:raise RuntimeError(cp.stdout+'\n'+cp.stderr)
    run=subprocess.run([str(exe),str(args.reps),str(args.iters)],capture_output=True,text=True,timeout=900)
    if run.returncode:raise RuntimeError(run.stdout+'\n'+run.stderr)
    if args.artifact_dir:args.artifact_dir.mkdir(parents=True,exist_ok=True);shutil.copy2(cu,args.artifact_dir/'gate.cu');shutil.copy2(exe,args.artifact_dir/'gate')
  rows=[]
  for line in run.stdout.splitlines():
    if m:=SAMPLE.fullmatch(line):rows.append({"rep":int(m[1]),"control_hot_us":float(m[2]),"candidate2_hot_us":float(m[3]),"candidate4_hot_us":float(m[4]),"control_cold_us":float(m[5]),"candidate2_cold_us":float(m[6]),"candidate4_cold_us":float(m[7])})
  if not rows or 'validate exact2=1 finite2=1 exact4=1 finite4=1' not in run.stdout:raise RuntimeError('unparsed or invalid output\n'+run.stdout)
  med={k:statistics.median(r[k] for r in rows) for k in ("control_hot_us","candidate2_hot_us","candidate4_hot_us","control_cold_us","candidate2_cold_us","candidate4_cold_us")}
  for cs in (2,4):med[f"candidate{cs}_hot_recovery_us"]=med["control_hot_us"]-med[f"candidate{cs}_hot_us"];med[f"candidate{cs}_cold_recovery_us"]=med["control_cold_us"]-med[f"candidate{cs}_cold_us"]
  result={"schema":"tinygrad.nv_q4k_o_cluster_geometry_gate.v2","commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"exact":True,"shape":{"rows":ROWS,"k":K,"control":{"blocks":4096,"threads":32},"candidate2":{"blocks":1024,"threads":128,"cluster_size":2,"rows_per_cluster":8},"candidate4":{"blocks":1024,"threads":128,"cluster_size":4,"rows_per_cluster":16}},"median":med,"samples":rows,"stdout":run.stdout,"ptxas":cp.stderr}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
