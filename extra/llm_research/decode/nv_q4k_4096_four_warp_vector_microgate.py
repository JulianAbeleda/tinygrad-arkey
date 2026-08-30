#!/usr/bin/env python3
"""Standalone current-vector versus four-warp-vector Q4_K 4096x4096 gate."""
from __future__ import annotations

import argparse, collections, csv, io, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import LanePartition, Q4KGateUpLaneMap, Q4K_WORDS_PER_BLOCK, _q4k_block_dot_packed_load_vec, q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS
from tinygrad.llm.shared_q8_attention import _emit_q4_cooperative, _emit_q8_provider
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp

ROWS=K=4096; WARPS=4; WARP=32; K_BLOCKS=K//Q4_K_BLOCK_ELEMS; WORDS=K_BLOCKS*36*ROWS
NCU="/usr/local/bin/ncu"


def ncu(binary:Path, symbol:str) -> list[dict[str,str]]:
  metrics=",".join(("dram__bytes.sum","dram__bytes_op_read.sum","dram__bytes_op_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed","gpu__time_duration.sum","lts__t_bytes.sum",
    "lts__t_sector_op_read_hit_rate.pct","sm__inst_executed.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread","smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"))
  cp=subprocess.run(["sudo","-n",NCU,"-k",symbol,"--launch-skip","1","--launch-count","1",
    "--cache-control","all","--metrics",metrics,"--csv",str(binary),"1","1"],capture_output=True,text=True)
  if cp.returncode: raise RuntimeError(f"ncu {symbol} failed: {cp.stderr[-4000:]}")
  rows=[]; header=None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0]=="ID": header=cols; continue
    if header is not None and len(cols)==len(header):
      row=dict(zip(header,cols)); rows.append({"metric":row["Metric Name"],"unit":row["Metric Unit"],"value":row["Metric Value"]})
  return rows


def sass_classes(binary:Path) -> dict[str,dict[str,int]]:
  cp=subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump","--dump-sass",str(binary)],capture_output=True,text=True)
  if cp.returncode: raise RuntimeError(f"cuobjdump failed: {cp.stderr[-4000:]}")
  current=None; out:dict[str,collections.Counter]={}
  for line in cp.stdout.splitlines():
    if m:=re.search(r"Function : (\S+)",line): current=m.group(1); out[current]=collections.Counter(); continue
    if current and (m:=re.search(r"/\*[^*]+\*/\s+([A-Z][A-Z0-9_.]*)",line)): out[current][m.group(1).split('.')[0]]+=1
  return {name:dict(counts.most_common()) for name,counts in out.items()}


def candidate(out:UOp, words:UOp, x:UOp) -> UOp:
  lm=Q4KGateUpLaneMap(k=K,n=ROWS); row=UOp.special(ROWS,"gidx0"); lid=UOp.special(WARP*WARPS,"lidx0")
  warp,lane=lid//WARP,lid%WARP; part=LanePartition(lane,lane_extent=lm.lane_extent,words_per_group=lm.words_per_group)
  block=warp*(K_BLOCKS//WARPS)+part.block_group; base=(row*K_BLOCKS+block)*Q4K_WORDS_PER_BLOCK
  contrib=_q4k_block_dot_packed_load_vec(words,x,base,block,part.word_col)
  subtotal=_warp_reduce_sum_staged(contrib,lane,WARP,90)
  smem=UOp.placeholder((WARPS,),dtypes.float32,40,addrspace=AddrSpace.LOCAL)
  ready=UOp.barrier(UOp.group(smem[warp].store(subtotal,lane.eq(0))))
  total=UOp.const(dtypes.float32,0.0)
  for wi in range(WARPS): total=total+smem.after(ready)[wi]
  return out[row].store(total,lid.eq(0)).sink(arg=KernelInfo(name="q4k_four_warp_vec_4096_4096",opts_to_apply=()))


def render() -> tuple[str,str,str,str]:
  ren=CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False)
  out=UOp.placeholder((ROWS,),dtypes.float32,0); words=UOp.placeholder((WORDS,),dtypes.uint32,1); x=UOp.placeholder((K,),dtypes.float16,2)
  control=q4k_g3_lanemap_gemv_kernel(ROWS,K,load_style="vector")(out,words,x)
  def src(u):
    s=next(v.arg for v in to_program(u,ren).src if v.op is Ops.SOURCE); return s[s.index('extern "C" __global__'):]
  packed=UOp.placeholder((K//4+K//32,),dtypes.uint32,3)
  provider=_emit_q8_provider()(packed,x)
  q8=_emit_q4_cooperative(ROWS,UOp.const(dtypes.weakint,4),direct_output=True)(out,words,packed)
  return src(control),src(candidate(out,words,x)),src(provider),src(q8)


HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }
__CONTROL__
__CANDIDATE__
__PROVIDER__
__Q8__
static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static double tc(float* o,unsigned int* w,half* x,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)q4k_g3_lanemap_gemv_vec_4096_4096<<<4096,32>>>(o,w,x);cudaEventRecord(b);ck(cudaDeviceSynchronize(),"control");float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
static double tv(float* o,unsigned int* w,half* x,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)q4k_four_warp_vec_4096_4096<<<4096,128>>>(o,w,x);cudaEventRecord(b);ck(cudaDeviceSynchronize(),"candidate");float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
static double tq(float* o,unsigned int* w,half* x,unsigned int* xp,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++){q8_1_llama_provider_4096<<<128,1>>>(x,xp);q4k_warp_coop_q8_dp4a_direct_4096_4096<<<4096,128>>>(o,w,xp);}cudaEventRecord(b);ck(cudaDeviceSynchronize(),"q8 included");float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
int main(int ac,char** av){int n=ac>1?atoi(av[1]):200,r=ac>2?atoi(av[2]):9;float *a,*b;unsigned int*w,*xp;half*x;ck(cudaMalloc(&a,4096*4),"a");ck(cudaMalloc(&b,4096*4),"b");ck(cudaMalloc(&w,WORDS_ARG*4),"w");ck(cudaMalloc(&x,4096*2),"x");ck(cudaMalloc(&xp,1152*4),"xp");unsigned int*hw=(unsigned int*)malloc(WORDS_ARG*4);half*hx=(half*)malloc(4096*2);for(size_t i=0;i<WORDS_ARG;i++)hw[i]=0x11111111u;for(int i=0;i<4096;i++)hx[i]=__float2half(((i%17)-8)*0.01f);cudaMemcpy(w,hw,WORDS_ARG*4,cudaMemcpyHostToDevice);cudaMemcpy(x,hx,4096*2,cudaMemcpyHostToDevice);free(hw);free(hx);q4k_g3_lanemap_gemv_vec_4096_4096<<<4096,32>>>(a,w,x);q4k_four_warp_vec_4096_4096<<<4096,128>>>(b,w,x);q8_1_llama_provider_4096<<<128,1>>>(x,xp);q4k_warp_coop_q8_dp4a_direct_4096_4096<<<4096,128>>>(b,w,xp);ck(cudaDeviceSynchronize(),"warmup");float *ha=(float*)malloc(4096*4),*hb=(float*)malloc(4096*4);q4k_g3_lanemap_gemv_vec_4096_4096<<<4096,32>>>(a,w,x);q4k_four_warp_vec_4096_4096<<<4096,128>>>(b,w,x);cudaMemcpy(ha,a,4096*4,cudaMemcpyDeviceToHost);cudaMemcpy(hb,b,4096*4,cudaMemcpyDeviceToHost);float md=0;for(int i=0;i<4096;i++)md=fmaxf(md,fabsf(ha[i]-hb[i]));printf("max_abs=%g\n",md);for(int i=0;i<r;i++)printf("rep=%d control=%.6f candidate=%.6f q8_included=%.6f\n",i,tc(a,w,x,n),tv(b,w,x,n),tq(b,w,x,xp,n));}
'''


def main() -> None:
  ap=argparse.ArgumentParser();ap.add_argument("--passes",type=int,default=200);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--ncu",action="store_true");ap.add_argument("--sass",action="store_true");ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
  control,cand,provider,q8=render(); cu=HARNESS.replace("__CONTROL__",control).replace("__CANDIDATE__",cand).replace("__PROVIDER__",provider).replace("__Q8__",q8).replace("WORDS_ARG",str(WORDS))
  with tempfile.TemporaryDirectory(prefix="q4k4096fw_") as td:
    src=Path(td)/"gate.cu"; binary=Path(td)/"gate"; src.write_text(cu)
    cp=subprocess.run(["/usr/local/cuda-13.2/bin/nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(src),"-o",str(binary)],capture_output=True,text=True)
    if cp.returncode: raise RuntimeError(cp.stderr[-8000:])
    run=subprocess.run([str(binary),str(a.passes),str(a.reps)],capture_output=True,text=True,check=True)
    counters={s:ncu(binary,s) for s in ("q4k_g3_lanemap_gemv_vec_4096_4096","q4k_four_warp_vec_4096_4096",
      "q8_1_llama_provider_4096","q4k_warp_coop_q8_dp4a_direct_4096_4096")} if a.ncu else None
    sass=sass_classes(binary) if a.sass else None
  cv,vv,qv=[],[],[]
  for line in run.stdout.splitlines():
    if m:=re.match(r"rep=\d+ control=([0-9.]+) candidate=([0-9.]+) q8_included=([0-9.]+)",line):cv.append(float(m.group(1)));vv.append(float(m.group(2)));qv.append(float(m.group(3)))
  md=float(re.search(r"max_abs=([0-9.eE+-]+)",run.stdout).group(1)); result={"schema":"tinygrad.nv_q4k_4096_four_warp_vector_microgate.v2","shape":[ROWS,K],"max_abs":md,"control_us":cv,"candidate_us":vv,"q8_included_us":qv,"control_median_us":statistics.median(cv),"candidate_median_us":statistics.median(vv),"q8_included_median_us":statistics.median(qv),"ratio":statistics.median(vv)/statistics.median(cv),"ptxas":cp.stderr.splitlines()}
  if counters is not None: result["ncu"]={"method":"cold cache, one post-warmup launch","kernels":counters}
  if sass is not None: result["sass_opcode_counts_static"]=sass
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
