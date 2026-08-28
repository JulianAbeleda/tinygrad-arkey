#!/usr/bin/env python3
"""Research-only Flash-combine -> O successor-prefetch chain gate."""
from __future__ import annotations

import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp

H,HD,S,W,ROWS,K=32,128,6,130,4096,4096
O_WORDS=ROWS*(K//256)*36
ROTATIONS=16
NVCC="/usr/local/cuda-13.2/bin/nvcc"


def render(ast:UOp)->str:
  prg=to_program(ast,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False))
  return next(x.arg for x in prg.src if x.op is Ops.SOURCE)


def prefetched_combine(base:str,groups:int,early:bool=False)->tuple[str,str]:
  old="flash_fused_gmax_combine_f16_32_128_s6_lw128_regw"
  name=f"{old}_opf{groups}{'early' if early else ''}"
  src=base[base.index('extern "C"'):].replace(old,name,1)
  src=src.replace("float* data1_24960) {","float* data1_24960, const unsigned int* successor_words) {",1)
  needle="  int lidx0 = threadIdx.x; /* 128 */" if early else "  *(data0_4096+(lidx0+(gidx0<<7))) ="
  if needle not in src: raise RuntimeError("combine insertion spelling changed")
  lines=["  int successor_row = (gidx0<<7)+lidx0;"]
  for block in (0,4,8,12)[:groups]:
    lines.append(f'  asm volatile("prefetch.global.L2 [%0];" :: "l"(successor_words+((successor_row*16+{block})*36)));')
  src=src.replace(needle,needle+"\n"+"\n".join(lines) if early else "\n".join(lines)+"\n"+needle,1)
  return name,src


def main()->int:
  ap=argparse.ArgumentParser(); ap.add_argument("--hot-passes",type=int,default=500); ap.add_argument("--cold-passes",type=int,default=32)
  ap.add_argument("--reps",type=int,default=9); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
  p=UOp.placeholder
  cbody=flash_fused_gmax_combine_kernel(HD,H,S,output_fp16=True,lane_width=128,register_weights=True)(
    p((H*HD,),dtypes.float16,0),p((H*S*W,),dtypes.float32,1))
  obody=q4k_g3_lanemap_gemv_kernel(ROWS,K,load_style="vector")(
    p((ROWS,),dtypes.float32,0),p((O_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2))
  csrc=render(cbody); osrc=render(obody); cname="flash_fused_gmax_combine_f16_32_128_s6_lw128_regw"
  variants=[prefetched_combine(csrc,n) for n in (1,2,4)]
  early2=prefetched_combine(csrc,2,early=True)
  half4="struct __align__(8) half4 { half x,y,z,w; };\n__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }\n"
  bodies=csrc+"\n"+half4+"\n".join(x[1] for x in variants+[early2])+"\n"+osrc[osrc.index('extern "C"'):]
  harness=f'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#define H {H}
#define HD {HD}
#define S {S}
#define W {W}
#define ROWS {ROWS}
#define O_WORDS {O_WORDS}
#define ROTATIONS {ROTATIONS}
{bodies}
static void ck(cudaError_t e,const char* w){{if(e!=cudaSuccess){{fprintf(stderr,"%s: %s\\n",w,cudaGetErrorString(e));exit(2);}}}}
static void chain(int arm,float* out,half* attn,float* partial,unsigned int* weights,float* residual){{
  if(arm==0) {cname}<<<H,128>>>(attn,partial);
  else if(arm==1) {variants[0][0]}<<<H,128>>>(attn,partial,weights);
  else if(arm==2) {variants[1][0]}<<<H,128>>>(attn,partial,weights);
  else if(arm==3) {variants[2][0]}<<<H,128>>>(attn,partial,weights);
  else {early2[0]}<<<H,128>>>(attn,partial,weights);
  q4k_g3_lanemap_gemv_vec_4096_4096<<<ROWS,32>>>(out,weights,attn);
}}
static double hot(int arm,float* out,half* attn,float* partial,unsigned int* w,float* residual,int passes){{
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<passes;i++)chain(arm,out,attn,partial,w,residual);
  cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0/passes;
}}
static double cold(int arm,float* out,half* attn,float* partial,unsigned int* ws,float* residual,unsigned char* evict,int passes){{
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);double us=0;for(int i=0;i<passes;i++){{unsigned int* w=ws+(i%ROTATIONS)*O_WORDS;
    cudaMemset(evict,i,128ull<<20);cudaDeviceSynchronize();
    cudaEventRecord(a);chain(arm,out,attn,partial,w,residual);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);us+=ms*1000.0;}}
  cudaEventDestroy(a);cudaEventDestroy(b);return us/passes;
}}
int main(int argc,char**argv){{int hp=atoi(argv[1]),cp=atoi(argv[2]),reps=atoi(argv[3]);float *outs[5],*partial,*residual;half*attn;unsigned int*ws;unsigned char*evict;
  for(int a=0;a<5;a++)ck(cudaMalloc(&outs[a],ROWS*4),"out");ck(cudaMalloc(&partial,H*S*W*4),"partial");ck(cudaMalloc(&attn,H*HD*2),"attn");
  ck(cudaMalloc(&residual,ROWS*4),"residual");ck(cudaMalloc(&ws,(size_t)ROTATIONS*O_WORDS*4),"weights");ck(cudaMalloc(&evict,128ull<<20),"evict");cudaMemset(partial,1,H*S*W*4);cudaMemset(residual,0,ROWS*4);cudaMemset(ws,7,(size_t)ROTATIONS*O_WORDS*4);
  for(int a=0;a<5;a++)chain(a,outs[a],attn,partial,ws,residual);cudaDeviceSynchronize();float *ref=(float*)malloc(ROWS*4),*got=(float*)malloc(ROWS*4);cudaMemcpy(ref,outs[0],ROWS*4,cudaMemcpyDeviceToHost);
  for(int a=1;a<5;a++){{cudaMemcpy(got,outs[a],ROWS*4,cudaMemcpyDeviceToHost);printf("bitwise_arm%d=%d\\n",a,memcmp(ref,got,ROWS*4)==0);}}
  for(int r=0;r<reps;r++)for(int i=0;i<5;i++){{int a=(r&1)?4-i:i;printf("rep=%d arm=%d hot=%.6f cold=%.6f\\n",r,a,hot(a,outs[a],attn,partial,ws,residual,hp),cold(a,outs[a],attn,partial,ws,residual,evict,cp));}}
}}
'''
  with tempfile.TemporaryDirectory(prefix="nv_flash_o_prefetch_") as td:
    cu,binp=Path(td)/"gate.cu",Path(td)/"gate";cu.write_text(harness)
    build=subprocess.run([NVCC,"-O3","-std=c++17","-arch=sm_120a","--ptxas-options=-v",str(cu),"-o",str(binp)],capture_output=True,text=True)
    if build.returncode: raise RuntimeError(build.stderr[-12000:])
    run=subprocess.run([str(binp),str(args.hot_passes),str(args.cold_passes),str(args.reps)],capture_output=True,text=True,check=True)
  print(run.stdout.strip());rows=[]
  for line in run.stdout.splitlines():
    if m:=re.match(r"rep=(\d+) arm=(\d+) hot=([0-9.]+) cold=([0-9.]+)",line): rows.append({"rep":int(m[1]),"arm":int(m[2]),"hot_us":float(m[3]),"cold_us":float(m[4])})
  names=("control","prefetch1_late","prefetch2_late","prefetch4_late","prefetch2_early");med={}
  for arm,name in enumerate(names):
    rr=[x for x in rows if x["arm"]==arm];med[name]={"hot_us":statistics.median(x["hot_us"] for x in rr),"cold_us":statistics.median(x["cold_us"] for x in rr)}
  out={"schema":"tinygrad.nv_flash_combine_o_successor_prefetch.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
    "arms":names,"bitwise_identical":all(f"bitwise_arm{a}=1" in run.stdout for a in (1,2,3,4)),"samples":rows,"median":med,"ptxas":build.stderr.strip()}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");return 0 if out["bitwise_identical"] else 5

if __name__=="__main__":raise SystemExit(main())
