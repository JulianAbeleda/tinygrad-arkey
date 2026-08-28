#!/usr/bin/env python3
"""Research-only exact in-loop L2-lookahead gate for residual-fused FP16 O.

The candidate is source-identical to the installed vector-load O body except
that the lane-0 member of each eight-lane block group hints the next Q4_K
block's first and second 128-byte cache lines into L2.  The hints retain no
future payload in registers and do not change weight layout or arithmetic.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue,q4k_g3_lanemap_gemv_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.decode.nv_q4k_qdata_transpose_o_microgate import _ncu,_sass_census

ROWS=K=4096
WORDS=2359296
WEIGHT_BYTES=WORDS*4
ROTATIONS=16
CUDA_BIN="/usr/local/cuda-13.2/bin"
CONTROL="q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096"
CANDIDATE="q4k_g3_lanemap_gemv_vec_epi_resadd_lookahead_4096_4096"


def _render()->tuple[str,str]:
  p=UOp.placeholder
  body=q4k_g3_lanemap_gemv_kernel(ROWS,K,epilogue=Q4KGEMVEpilogue("residual_add"),load_style="vector")(
    p((ROWS,),dtypes.float32,0),p((WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2),p((ROWS,),dtypes.float32,3))
  text=next(x.arg for x in to_program(body,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False)).src if x.op is Ops.SOURCE)
  control=text[text.index('extern "C" __global__'):]
  candidate=control.replace(CONTROL,CANDIDATE,1)
  anchor="    int alu4 = (alu3+alu2);\n"
  if candidate.count(anchor)!=1:raise RuntimeError("installed O loop spelling changed")
  # `alu1` selects one of the four block groups, while alu2 is its word-column.
  # Four lanes (8/warp) issue two non-binding hints each.  The hinted block is
  # Ridx0+1 in that group's sequence; +32 words reaches the block's second line.
  hints='''    if ((alu2 == 0) && (Ridx0 < 3)) {
      asm volatile("prefetch.global.L2 [%0];" :: "l"(data1_2359296+((alu1*144)+((Ridx0+1)*36)+(gidx0*576))));
      asm volatile("prefetch.global.L2 [%0];" :: "l"(data1_2359296+((alu1*144)+((Ridx0+1)*36)+(gidx0*576)+32)));
    }
'''
  candidate=candidate.replace(anchor,anchor+hints,1)
  return control,candidate


HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#define ROWS 4096
#define K 4096
#define K_BLOCKS 16
#define WORDS 2359296
#define ROTATIONS 16
template <class T,class F> __device__ __forceinline__ T tg_bitcast(F v){union U{F f;T t;};U u;u.f=v;return u.t;}
struct __align__(8) half4{half x,y,z,w;};
__device__ half4 make_half4(half x,half y,half z,half w){half4 r={x,y,z,w};return r;}
__CONTROL_SOURCE__
__CANDIDATE_SOURCE__
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static uint32_t step(uint32_t&s){s=1664525u*s+1013904223u;return s;}
static void fill_fixture(uint32_t*w,half*x,float*r,int fixture){
  uint32_t s=0x1234567u^(uint32_t(fixture)*0x9e3779b9u);const uint16_t d[4]={0x2c00u,0x3000u,0x3400u,0x3800u},m[4]={0x2800u,0x2c00u,0x3000u,0x3400u};
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++){size_t base=b*36;int p=int((b+fixture)&3);w[base]=uint32_t(d[p])|(uint32_t(m[(p+1)&3])<<16);
    if(fixture==0){w[base+1]=step(s);w[base+2]=step(s);w[base+3]=step(s);for(int i=4;i<36;i++)w[base+i]=step(s);}
    else if(fixture==1){w[base+1]=0x01020304u^uint32_t(b);w[base+2]=0x10203040u+uint32_t(b);w[base+3]=0x3f2f1f0fu;for(int i=4;i<36;i++)w[base+i]=0x01234567u*uint32_t(i)+uint32_t(b)*0x11111111u;}
    else{w[base+1]=0x3f3f3f3fu;w[base+2]=0x15151515u;w[base+3]=0x2a2a2a2au;for(int i=4;i<36;i++)w[base+i]=(i&1)?0xfedcba98u:0x76543210u;}}
  for(int i=0;i<K;i++){float v=fixture==0?float((int(step(s)>>16)%511)-255)/512.0f:fixture==1?float((i%257)-128)/256.0f:float((i%17)-8)/64.0f;x[i]=__float2half(v);}
  for(int i=0;i<ROWS;i++)r[i]=fixture==0?float((int(step(s)>>16)%255)-127)/1024.0f:fixture==1?float((i%31)-15)/128.0f:float((i%7)-3)/32.0f;
}
static void launch(int arm,float*out,uint32_t*w,half*x,float*r){
  if(arm==0)q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32>>>(out,w,x,r);
  else q4k_g3_lanemap_gemv_vec_epi_resadd_lookahead_4096_4096<<<ROWS,32>>>(out,w,x,r);
}
static double hot(int arm,float*out,uint32_t*w,half*x,float*r,int passes,cudaEvent_t a,cudaEvent_t b){cudaEventRecord(a);for(int i=0;i<passes;i++)launch(arm,out,w,x,r);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000.0/passes;}
static double cold(int arm,float*out,uint32_t*ws,half*x,float*r,int passes,cudaEvent_t a,cudaEvent_t b){double us=0;for(int i=0;i<passes;i++){uint32_t*w=ws+(size_t)(i%ROTATIONS)*WORDS;cudaEventRecord(a);launch(arm,out,w,x,r);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);us+=ms*1000.0;}return us/passes;}
int main(int argc,char**argv){int hp=argc>1?atoi(argv[1]):300,cp=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9;bool profile=argc>1&&!strcmp(argv[1],"profile");
  uint32_t*ws;half*x;float*r,*co,*ca;cudaMalloc(&ws,(size_t)ROTATIONS*WORDS*4);cudaMalloc(&x,K*2);cudaMalloc(&r,ROWS*4);cudaMalloc(&co,ROWS*4);cudaMalloc(&ca,ROWS*4);
  uint32_t*hw=(uint32_t*)malloc((size_t)WORDS*4);half*hx=(half*)malloc(K*2);float*hr=(float*)malloc(ROWS*4);int exact=1;
  for(int f=0;f<3;f++){fill_fixture(hw,hx,hr,f);cudaMemcpy(ws,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice);cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice);cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice);launch(0,co,ws,x,r);launch(1,ca,ws,x,r);cudaDeviceSynchronize();float*h0=(float*)malloc(ROWS*4),*h1=(float*)malloc(ROWS*4);cudaMemcpy(h0,co,ROWS*4,cudaMemcpyDeviceToHost);cudaMemcpy(h1,ca,ROWS*4,cudaMemcpyDeviceToHost);int mm=0,finite=1;double ma=0;for(int i=0;i<ROWS;i++){uint32_t a,b;memcpy(&a,h0+i,4);memcpy(&b,h1+i,4);mm+=a!=b;finite&=isfinite(h0[i])&&isfinite(h1[i]);ma=fmax(ma,fabs(double(h0[i])-double(h1[i])));}printf("fixture=%d finite=%d mismatched_words=%d max_abs=%.9g\n",f,finite,mm,ma);exact&=finite&&mm==0;free(h0);free(h1);}
  fill_fixture(hw,hx,hr,0);for(int i=0;i<ROTATIONS;i++)cudaMemcpy(ws+(size_t)i*WORDS,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice);cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice);cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice);free(hw);free(hx);free(hr);
  for(int i=0;i<20;i++){launch(0,co,ws,x,r);launch(1,ca,ws,x,r);}cudaDeviceSynchronize();if(profile){launch(0,co,ws,x,r);launch(1,ca,ws,x,r);cudaDeviceSynchronize();return exact?0:5;}
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int j=0;j<reps;j++){double ch,nh,cc,nc;if(!(j&1)){ch=hot(0,co,ws,x,r,hp,a,b);nh=hot(1,ca,ws,x,r,hp,a,b);cc=cold(0,co,ws,x,r,cp,a,b);nc=cold(1,ca,ws,x,r,cp,a,b);}else{nh=hot(1,ca,ws,x,r,hp,a,b);ch=hot(0,co,ws,x,r,hp,a,b);nc=cold(1,ca,ws,x,r,cp,a,b);cc=cold(0,co,ws,x,r,cp,a,b);}printf("rep=%d hot_control_us=%.6f hot_candidate_us=%.6f cold_control_us=%.6f cold_candidate_us=%.6f\n",j,ch,nh,cc,nc);}return exact?0:5;}
'''


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--hot-passes",type=int,default=500);ap.add_argument("--cold-passes",type=int,default=32);ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--threshold-us",type=float,default=.15);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--artifact-dir",type=Path);ap.add_argument("--ncu",action="store_true");a=ap.parse_args()
  control,candidate=_render();source=HARNESS.replace("__CONTROL_SOURCE__",control).replace("__CANDIDATE_SOURCE__",candidate)
  with tempfile.TemporaryDirectory(prefix="nv_q4k_o_lookahead_") as td:
    cu,bi=Path(td)/"gate.cu",Path(td)/"gate";cu.write_text(source);env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    build=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(cu),"-o",str(bi)],capture_output=True,text=True,env=env)
    if build.returncode:print(build.stderr[-12000:],file=sys.stderr);return 3
    art=a.artifact_dir
    if art:art.mkdir(parents=True,exist_ok=True);shutil.copy2(cu,art/"gate.cu");shutil.copy2(bi,art/"gate");(art/"ptxas.txt").write_text(build.stderr)
    run=subprocess.run([str(bi),str(a.hot_passes),str(a.cold_passes),str(a.reps)],capture_output=True,text=True);print(run.stdout.strip())
    if run.returncode not in (0,5):print(run.stderr[-8000:],file=sys.stderr);return 4
    fixtures=[{"fixture":int(m[1]),"finite":bool(int(m[2])),"mismatched_words":int(m[3]),"max_abs":float(m[4])} for m in re.finditer(r"fixture=(\d+) finite=(\d+) mismatched_words=(\d+) max_abs=([0-9.eE+-]+)",run.stdout)]
    samples={k:[] for k in ("hot_control","hot_candidate","cold_control","cold_candidate")};pat=re.compile(r"hot_control_us=([0-9.]+) hot_candidate_us=([0-9.]+) cold_control_us=([0-9.]+) cold_candidate_us=([0-9.]+)")
    for m in pat.finditer(run.stdout):
      for k,v in zip(samples,m.groups()):samples[k].append(float(v))
    med={k:statistics.median(v) for k,v in samples.items()};hot=med["hot_control"]-med["hot_candidate"];cold=med["cold_control"]-med["cold_candidate"];exact=len(fixtures)==3 and all(x["finite"] and x["mismatched_words"]==0 for x in fixtures)
    result={"schema":"tinygrad.nv_q4k_o_l2_lookahead_microgate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"method":f"installed residual-fused FP16 O plus register-free next-block two-line L2 hints; cudaEvent hot and 16-copy rotated-cold r{a.reps}",
      "shape":{"rows":ROWS,"k":K,"weight_bytes":WEIGHT_BYTES,"rotated_weight_bytes":ROTATIONS*WEIGHT_BYTES},"correctness":{"fixtures":fixtures,"bitwise_all":exact},
      "timing":{"unit":"us_per_call","hot_passes":a.hot_passes,"cold_passes":a.cold_passes,"reps":a.reps,"samples":samples,"medians":med,"hot_recovery_us":hot,"cold_recovery_us":cold,"control_cold_rate_tb_s":WEIGHT_BYTES/med["cold_control"]/1e6,"candidate_cold_rate_tb_s":WEIGHT_BYTES/med["cold_candidate"]/1e6},
      "lookahead":{"distance_blocks":1,"cache_lines_per_next_block":2,"issuer":"word_col zero in each 8-lane block group","payload_retained":False},"sass":{"control":_sass_census(bi,CONTROL),"candidate":_sass_census(bi,CANDIDATE)},"ptxas":build.stderr.strip().splitlines(),"threshold":{"cold_recovery_us_per_call":a.threshold_us,"scoreboard_must_reduce":True}}
    if a.ncu:
      if art is None:art=a.out.parent/"artifacts";art.mkdir(parents=True,exist_ok=True)
      result["ncu"]={"control":_ncu(bi,CONTROL,art/"ncu-control.csv.txt"),"candidate":_ncu(bi,CANDIDATE,art/"ncu-candidate.csv.txt")}
      def metric(arm,name):
        row=next((x for x in result["ncu"][arm]["rows"] if x["metric"]==name),None);return None if row is None else float(row["value"].replace(",",""))
      cs=metric("control","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct");ns=metric("candidate","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct")
      result["scoreboard"]={"control_pct":cs,"candidate_pct":ns,"reduced":cs is not None and ns is not None and ns<cs}
    scoreboard_ok=result.get("scoreboard",{}).get("reduced",True)
    result["verdict"]="ADVANCE_TO_PRODUCTION_WALL" if exact and cold>=a.threshold_us and scoreboard_ok else "STOP"
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0 if exact else 5


if __name__=="__main__":raise SystemExit(main())
