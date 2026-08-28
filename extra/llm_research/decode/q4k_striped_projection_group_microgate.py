#!/usr/bin/env python3
"""Research-only Q4/Q4/Q4 one-task projection stripe discriminator.

Compares the installed Q + paired-K/V path and the historical Q-first full
grid with two 6144-row, one-dot-per-CTA grids.  The phased and interleaved
grids have identical arithmetic and differ only in CTA-to-virtual-row order.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import LanePartition, Q4KGateUpLaneMap, Q4K_WORDS_PER_BLOCK, _q4k_block_dot_packed_load_vec, q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.q4k_kv_pair import emit_q4k_kv_pair_vector, emit_q4k_qkv_full
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

Q_ROWS,KV_ROWS,K,TOTAL_ROWS=4096,1024,4096,6144
Q_WORDS=Q_ROWS*(K//256)*Q4K_WORDS_PER_BLOCK
KV_WORDS=KV_ROWS*(K//256)*Q4K_WORDS_PER_BLOCK
GROUP_WORDS=Q_WORDS+2*KV_WORDS
CUDA_BIN="/usr/local/cuda-13.2/bin"


def emit_one_task_grid(interleaved:bool):
  lm=Q4KGateUpLaneMap(k=K,n=Q_ROWS); lm.validate()
  suffix="interleaved" if interleaved else "phased"
  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    gid,lane=UOp.special(TOTAL_ROWS,"gidx0"),UOp.special(32,"lidx0")
    if interleaved:
      group,slot=gid//6,gid%6
      virtual_row=(slot<4).where(group*4+slot,slot.eq(4).where(Q_ROWS+group,Q_ROWS+KV_ROWS+group))
    else: virtual_row=gid
    part=LanePartition(lane,lane_extent=lm.lane_extent,words_per_group=lm.words_per_group)
    lblk=UOp.range(lm.blocks_per_group,0,axis_type=AxisType.REDUCE)
    block=part.block_group*lm.blocks_per_group+lblk
    base=(virtual_row*lm.k_blocks+block)*Q4K_WORDS_PER_BLOCK
    contrib=_q4k_block_dot_packed_load_vec(words,x,base,block,part.word_col)
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.0))
    acc=acc.after(acc[0].store(acc.after(lblk)[0]+contrib).end(lblk))
    total=_warp_reduce_sum_staged(acc[0],part.lane,part.lane_extent,90)
    return out[virtual_row].store(total).sink(arg=KernelInfo(name=f"q4k_projection_group_one_task_{suffix}_{TOTAL_ROWS}_{K}",opts_to_apply=()))
  return kernel


def _render() -> tuple[str,...]:
  p=UOp.placeholder
  bodies=(
    q4k_g3_lanemap_gemv_kernel(Q_ROWS,K,load_style="vector")(
      p((Q_ROWS,),dtypes.float32,0),p((Q_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2)),
    emit_q4k_kv_pair_vector()(p((KV_ROWS,),dtypes.float32,0),p((KV_ROWS,),dtypes.float32,1),
      p((KV_WORDS,),dtypes.uint32,2),p((KV_WORDS,),dtypes.uint32,3),p((K,),dtypes.float16,4)),
    emit_q4k_qkv_full()(p((Q_ROWS,),dtypes.float32,0),p((KV_ROWS,),dtypes.float32,1),p((KV_ROWS,),dtypes.float32,2),
      p((Q_WORDS,),dtypes.uint32,3),p((2*KV_WORDS,),dtypes.uint32,4),p((K,),dtypes.float16,5)),
    emit_one_task_grid(False)(p((TOTAL_ROWS,),dtypes.float32,0),p((GROUP_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2)),
    emit_one_task_grid(True)(p((TOTAL_ROWS,),dtypes.float32,0),p((GROUP_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2)),
  )
  ren=CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False)
  ret=[]
  for body in bodies:
    text=next(x.arg for x in to_program(body,ren).src if x.op is Ops.SOURCE)
    ret.append(text[text.index('extern "C" __global__'):])
  return tuple(ret)


HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define Q_ROWS 4096
#define KV_ROWS 1024
#define TOTAL_ROWS 6144
#define K 4096
#define Q_WORDS 2359296
#define KV_WORDS 589824
#define GROUP_WORDS 3538944
#define ROTATIONS 16
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }
__BODIES__
static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));exit(2);} }
static void launch(int arm,float* out,unsigned int* group,half* x) {
  if(arm==0) {
    q4k_g3_lanemap_gemv_vec_4096_4096<<<Q_ROWS,32>>>(out,group,x);
    q4k_g3_lanemap_gemv_pair_vec_1024_4096<<<KV_ROWS,32>>>(out+Q_ROWS,out+Q_ROWS+KV_ROWS,group+Q_WORDS,group+Q_WORDS+KV_WORDS,x);
  } else if(arm==1) {
    q4k_g3_lanemap_gemv_qkv_full_4096_1024_4096<<<Q_ROWS,32>>>(out,out+Q_ROWS,out+Q_ROWS+KV_ROWS,group,group+Q_WORDS,x);
  } else if(arm==2) {
    q4k_projection_group_one_task_phased_6144_4096<<<TOTAL_ROWS,32>>>(out,group,x);
  } else {
    q4k_projection_group_one_task_interleaved_6144_4096<<<TOTAL_ROWS,32>>>(out,group,x);
  }
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
  int hot_passes=argc>1?atoi(argv[1]):500,cold_passes=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9;
  float *outs[4]; unsigned int* groups; half* x;
  for(int a=0;a<4;a++) ck(cudaMalloc(&outs[a],TOTAL_ROWS*sizeof(float)),"out");
  ck(cudaMalloc(&groups,(size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int)),"groups"); ck(cudaMalloc(&x,K*sizeof(half)),"x");
  unsigned int* hw=(unsigned int*)malloc((size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int)); half* hx=(half*)malloc(K*sizeof(half));
  for(size_t i=0;i<(size_t)ROTATIONS*GROUP_WORDS;i++) hw[i]=(unsigned int)((i*2654435761u)^0x9e3779b9u);
  for(int i=0;i<K;i++) hx[i]=__float2half(((i%257)-128)*0.03125f);
  ck(cudaMemcpy(groups,hw,(size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int),cudaMemcpyHostToDevice),"weights");
  ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"x"); free(hw); free(hx);
  for(int a=0;a<4;a++) launch(a,outs[a],groups,x); ck(cudaDeviceSynchronize(),"warmup");
  float *ref=(float*)malloc(TOTAL_ROWS*sizeof(float)),*got=(float*)malloc(TOTAL_ROWS*sizeof(float));
  ck(cudaMemcpy(ref,outs[0],TOTAL_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"ref");
  for(int a=1;a<4;a++) { ck(cudaMemcpy(got,outs[a],TOTAL_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"got");
    printf("bitwise_arm%d=%d\n",a,memcmp(ref,got,TOTAL_ROWS*sizeof(float))==0); } free(ref); free(got);
  for(int r=0;r<reps;r++) for(int a=0;a<4;a++) {
    double h=hot(a,outs[a],groups,x,hot_passes),c=rotated(a,outs[a],groups,x,cold_passes);
    printf("rep=%d arm=%d hot=%.6f cold=%.6f\n",r,a,h,c);
  }
  return 0;
}
'''


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--hot-passes",type=int,default=500)
  ap.add_argument("--cold-passes",type=int,default=32); ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--out",type=Path,required=True); ap.add_argument("--artifact-dir",type=Path); args=ap.parse_args()
  source=HARNESS.replace("__BODIES__","\n".join(_render()))
  with tempfile.TemporaryDirectory(prefix="q4k_striped_projection_group_") as td:
    cu,binp=Path(td)/"gate.cu",Path(td)/"gate"; cu.write_text(source)
    env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    cp=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(cu),"-o",str(binp)],capture_output=True,text=True,env=env)
    if cp.returncode: raise RuntimeError(cp.stderr[-12000:])
    if args.artifact_dir:
      args.artifact_dir.mkdir(parents=True,exist_ok=True)
      shutil.copy2(cu,args.artifact_dir/"gate.cu"); shutil.copy2(binp,args.artifact_dir/"gate")
    run=subprocess.run([str(binp),str(args.hot_passes),str(args.cold_passes),str(args.reps)],capture_output=True,text=True)
    if run.returncode: raise RuntimeError(run.stderr[-6000:])
  print(run.stdout.strip())
  rows=[]
  for line in run.stdout.splitlines():
    m=re.match(r"rep=(\d+) arm=(\d+) hot=([0-9.]+) cold=([0-9.]+)",line)
    if m: rows.append({"rep":int(m.group(1)),"arm":int(m.group(2)),"hot_us":float(m.group(3)),"cold_us":float(m.group(4))})
  med={}
  for arm,name in enumerate(("installed","q_first_full","phased_one_task","interleaved_one_task")):
    vals=[r for r in rows if r["arm"]==arm]
    med[name]={"hot_us":statistics.median(r["hot_us"] for r in vals),"cold_us":statistics.median(r["cold_us"] for r in vals)}
  exact=all(f"bitwise_arm{a}=1" in run.stdout for a in (1,2,3))
  out={"schema":"tinygrad.q4k_striped_projection_group_microgate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
    "shape":{"q_rows":Q_ROWS,"k_rows":KV_ROWS,"v_rows":KV_ROWS,"input_width":K,"weight_bytes":GROUP_WORDS*4},
    "arms":["installed","q_first_full","phased_one_task","interleaved_one_task"],"bitwise_identical":exact,
    "samples":rows,"median":med,"ptxas":cp.stderr.strip()}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
  return 0 if exact else 5


if __name__=="__main__": raise SystemExit(main())
