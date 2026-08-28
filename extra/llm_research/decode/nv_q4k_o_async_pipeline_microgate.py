#!/usr/bin/env python3
"""Exact one/two-stage cp.async Q4_K O projection discriminator.

Research only.  Four eight-lane subgroups cooperatively stage the four Q4_K
blocks consumed by each loop iteration.  Each subgroup copies its 144-byte
block as eight aligned 16-byte copies plus one 16-byte tail copy.  The
one-stage arm waits before each iteration; the two-stage arm double-buffers
and issues iteration N+1 before computing N.  Results are admissible only if
the cubin contains real global-to-shared async instructions (LDGSTS family).
"""
from __future__ import annotations

import argparse,json,os,re,shutil,statistics,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue,q4k_g3_lanemap_gemv_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.decode.nv_q4k_qdata_transpose_o_microgate import _ncu,_sass_census

ROWS=K=4096;WORDS=2359296;WEIGHT_BYTES=WORDS*4;ROTATIONS=16;CUDA_BIN="/usr/local/cuda-13.2/bin"
CONTROL="q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096"
ONE="q4k_g3_lanemap_gemv_vec_epi_resadd_async1_4096_4096"
TWO="q4k_g3_lanemap_gemv_vec_epi_resadd_async2_4096_4096"


def _control_source()->str:
  p=UOp.placeholder;u=q4k_g3_lanemap_gemv_kernel(ROWS,K,epilogue=Q4KGEMVEpilogue("residual_add"),load_style="vector")(
    p((ROWS,),dtypes.float32,0),p((WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2),p((ROWS,),dtypes.float32,3))
  text=next(x.arg for x in to_program(u,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False)).src if x.op is Ops.SOURCE)
  return text[text.index('extern "C" __global__'):]


def _copy_lines(global_expr:str,shared_expr:str)->str:
  return f'''    {{
      unsigned int sd = (unsigned int)__cvta_generic_to_shared(qtile+({shared_expr})+alu2*4);
      const unsigned int* sg = data1_2359296+({global_expr})+alu2*4;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sd), "l"(sg));
      if (alu2 == 0) {{
        unsigned int td = (unsigned int)__cvta_generic_to_shared(qtile+({shared_expr})+32);
        const unsigned int* tg = data1_2359296+({global_expr})+32;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(td), "l"(tg));
      }}
      asm volatile("cp.async.commit_group;");
    }}
'''


def _candidate(base:str,two_stage:bool)->str:
  name=TWO if two_stage else ONE
  src=base.replace(CONTROL,name,1)
  decl="  int lidx0 = threadIdx.x; /* 32 */\n"
  src=src.replace(decl,decl+("  __shared__ __align__(16) unsigned int qtile[288];\n" if two_stage else "  __shared__ __align__(16) unsigned int qtile[144];\n"),1)
  loop="  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {\n    int alu3 = ((alu1*144)+(Ridx0*36)+(gidx0*576));\n"
  if src.count(loop)!=1:raise RuntimeError("installed loop spelling changed")
  if not two_stage:
    stage=_copy_lines("(alu1*144)+(Ridx0*36)+(gidx0*576)","alu1*36")
    repl="  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {\n"+stage+'''    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int alu3 = alu1*36;
'''
    src=src.replace(loop,repl,1)
    tail="  }\n  float val13 ="
    src=src.replace(tail,"    __syncthreads();\n  }\n  float val13 =",1)
  else:
    initial=_copy_lines("(alu1*144)+(gidx0*576)","alu1*36")+'''  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
'''
    next_copy=_copy_lines("(alu1*144)+((Ridx0+1)*36)+(gidx0*576)","((Ridx0+1)&1)*144+alu1*36")
    repl=initial+"  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {\n    if (Ridx0 < 3) {\n"+next_copy+"    }\n    int alu3 = (Ridx0&1)*144+alu1*36;\n"
    src=src.replace(loop,repl,1)
    tail="  }\n  float val13 ="
    src=src.replace(tail,'    if (Ridx0 < 3) { asm volatile("cp.async.wait_group 0;"); __syncthreads(); }\n  }\n  float val13 =',1)
  # All packed-weight reads now target the staged tile; activation/residual reads stay global.
  src=src.replace("data1_2359296+(alu4+4)","qtile+(alu4+4)").replace("data1_2359296+(alu4+12)","qtile+(alu4+12)")
  src=src.replace("data1_2359296+(alu4+20)","qtile+(alu4+20)").replace("data1_2359296+(alu4+28)","qtile+(alu4+28)")
  src=src.replace("data1_2359296+alu3","qtile+alu3")
  return src


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
template<class T,class F>__device__ __forceinline__ T tg_bitcast(F v){union U{F f;T t;};U u;u.f=v;return u.t;}
struct __align__(8) half4{half x,y,z,w;};__device__ half4 make_half4(half x,half y,half z,half w){half4 r={x,y,z,w};return r;}
__BODIES__
static uint32_t step(uint32_t&s){s=1664525u*s+1013904223u;return s;}
static void fill(uint32_t*w,half*x,float*r,int f){uint32_t s=0x1234567u^(uint32_t(f)*0x9e3779b9u);const uint16_t d[4]={0x2c00,0x3000,0x3400,0x3800},m[4]={0x2800,0x2c00,0x3000,0x3400};for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++){size_t z=b*36;int p=(b+f)&3;w[z]=uint32_t(d[p])|(uint32_t(m[(p+1)&3])<<16);if(f==0){w[z+1]=step(s);w[z+2]=step(s);w[z+3]=step(s);for(int i=4;i<36;i++)w[z+i]=step(s);}else if(f==1){w[z+1]=0x01020304u^b;w[z+2]=0x10203040u+b;w[z+3]=0x3f2f1f0f;for(int i=4;i<36;i++)w[z+i]=0x01234567u*i+b*0x11111111u;}else{w[z+1]=0x3f3f3f3f;w[z+2]=0x15151515;w[z+3]=0x2a2a2a2a;for(int i=4;i<36;i++)w[z+i]=(i&1)?0xfedcba98:0x76543210;}}for(int i=0;i<K;i++){float v=f==0?float((int(step(s)>>16)%511)-255)/512.0f:f==1?float((i%257)-128)/256.0f:float((i%17)-8)/64.0f;x[i]=__float2half(v);}for(int i=0;i<ROWS;i++)r[i]=f==0?float((int(step(s)>>16)%255)-127)/1024.0f:f==1?float((i%31)-15)/128.0f:float((i%7)-3)/32.0f;}
static void launch(int a,float*out,uint32_t*w,half*x,float*r){if(a==0)q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32>>>(out,w,x,r);else if(a==1)q4k_g3_lanemap_gemv_vec_epi_resadd_async1_4096_4096<<<ROWS,32>>>(out,w,x,r);else q4k_g3_lanemap_gemv_vec_epi_resadd_async2_4096_4096<<<ROWS,32>>>(out,w,x,r);}
static double hot(int a,float*out,uint32_t*w,half*x,float*r,int n,cudaEvent_t s,cudaEvent_t e){cudaEventRecord(s);for(int i=0;i<n;i++)launch(a,out,w,x,r);cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms*1000/n;}
static double cold(int a,float*out,uint32_t*ws,half*x,float*r,int n,cudaEvent_t s,cudaEvent_t e){double us=0;for(int i=0;i<n;i++){cudaEventRecord(s);launch(a,out,ws+(size_t)(i%ROTATIONS)*WORDS,x,r);cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);us+=ms*1000;}return us/n;}
int main(int ac,char**av){int hp=ac>1?atoi(av[1]):300,cp=ac>2?atoi(av[2]):32,reps=ac>3?atoi(av[3]):9;bool profile=ac>1&&!strcmp(av[1],"profile");uint32_t*ws;half*x;float*r,*out[3];cudaMalloc(&ws,(size_t)ROTATIONS*WORDS*4);cudaMalloc(&x,K*2);cudaMalloc(&r,ROWS*4);for(int a=0;a<3;a++)cudaMalloc(&out[a],ROWS*4);uint32_t*hw=(uint32_t*)malloc((size_t)WORDS*4);half*hx=(half*)malloc(K*2);float*hr=(float*)malloc(ROWS*4);int exact=1;
for(int f=0;f<3;f++){fill(hw,hx,hr,f);cudaMemcpy(ws,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice);cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice);cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice);for(int a=0;a<3;a++)launch(a,out[a],ws,x,r);cudaDeviceSynchronize();float*h0=(float*)malloc(ROWS*4),*ha=(float*)malloc(ROWS*4);cudaMemcpy(h0,out[0],ROWS*4,cudaMemcpyDeviceToHost);for(int a=1;a<3;a++){cudaMemcpy(ha,out[a],ROWS*4,cudaMemcpyDeviceToHost);int mm=0,finite=1;double ma=0;for(int i=0;i<ROWS;i++){uint32_t u,v;memcpy(&u,h0+i,4);memcpy(&v,ha+i,4);mm+=u!=v;finite&=isfinite(h0[i])&&isfinite(ha[i]);ma=fmax(ma,fabs(double(h0[i])-double(ha[i])));}printf("fixture=%d arm=%d finite=%d mismatched_words=%d max_abs=%.9g\n",f,a,finite,mm,ma);exact&=finite&&mm==0;}free(h0);free(ha);}fill(hw,hx,hr,0);for(int i=0;i<ROTATIONS;i++)cudaMemcpy(ws+(size_t)i*WORDS,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice);cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice);cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice);free(hw);free(hx);free(hr);for(int i=0;i<20;i++)for(int a=0;a<3;a++)launch(a,out[a],ws,x,r);cudaDeviceSynchronize();if(profile){for(int a=0;a<3;a++)launch(a,out[a],ws,x,r);cudaDeviceSynchronize();return exact?0:5;}cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);for(int j=0;j<reps;j++)for(int q=0;q<3;q++){int a=(j&1)?2-q:q;printf("rep=%d arm=%d hot=%.6f cold=%.6f\n",j,a,hot(a,out[a],ws,x,r,hp,s,e),cold(a,out[a],ws,x,r,cp,s,e));}return exact?0:5;}
'''


def _async_count(sass:dict)->int:
  return sum(v for k,v in sass.get("opcodes",{}).items() if k.startswith("LDGSTS") or "CPASYNC" in k)


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--hot-passes",type=int,default=500);ap.add_argument("--cold-passes",type=int,default=32);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--threshold-us",type=float,default=.15);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--artifact-dir",type=Path);ap.add_argument("--ncu",action="store_true");a=ap.parse_args()
  control=_control_source();one=_candidate(control,False);two=_candidate(control,True);source=HARNESS.replace("__BODIES__","\n".join((control,one,two)))
  with tempfile.TemporaryDirectory(prefix="nv_q4k_o_async_") as td:
    cu,bi=Path(td)/"gate.cu",Path(td)/"gate";cu.write_text(source);env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")};build=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(cu),"-o",str(bi)],capture_output=True,text=True,env=env)
    if build.returncode:print(build.stderr[-16000:],file=sys.stderr);return 3
    art=a.artifact_dir
    if art:art.mkdir(parents=True,exist_ok=True);shutil.copy2(cu,art/"gate.cu");shutil.copy2(bi,art/"gate");(art/"ptxas.txt").write_text(build.stderr)
    run=subprocess.run([str(bi),str(a.hot_passes),str(a.cold_passes),str(a.reps)],capture_output=True,text=True);print(run.stdout.strip())
    if run.returncode not in (0,5):print(run.stderr[-8000:],file=sys.stderr);return 4
    fixtures=[{"fixture":int(m[1]),"arm":int(m[2]),"finite":bool(int(m[3])),"mismatched_words":int(m[4]),"max_abs":float(m[5])} for m in re.finditer(r"fixture=(\d+) arm=(\d+) finite=(\d+) mismatched_words=(\d+) max_abs=([0-9.eE+-]+)",run.stdout)]
    rows=[{"rep":int(m[1]),"arm":int(m[2]),"hot_us":float(m[3]),"cold_us":float(m[4])} for m in re.finditer(r"rep=(\d+) arm=(\d+) hot=([0-9.]+) cold=([0-9.]+)",run.stdout)];names=("control","async_one_stage","async_two_stage");med={}
    for arm,name in enumerate(names):
      rr=[x for x in rows if x["arm"]==arm];med[name]={"hot_us":statistics.median(x["hot_us"] for x in rr),"cold_us":statistics.median(x["cold_us"] for x in rr)}
    exact=len(fixtures)==6 and all(x["finite"] and x["mismatched_words"]==0 for x in fixtures);sass={"control":_sass_census(bi,CONTROL),"async_one_stage":_sass_census(bi,ONE),"async_two_stage":_sass_census(bi,TWO)}
    for x in sass.values():x["async_global_to_shared"]= _async_count(x)
    result={"schema":"tinygrad.nv_q4k_o_async_pipeline_microgate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"method":f"installed exact O versus cooperative one/two-stage cp.async global-to-shared Q4_K staging; hot and 16-copy rotated-cold r{a.reps}","shape":{"rows":ROWS,"k":K,"weight_bytes":WEIGHT_BYTES},"correctness":{"fixtures":fixtures,"bitwise_all":exact},"samples":rows,"medians":med,"sass":sass,"ptxas":build.stderr.strip().splitlines(),"promotion_grade_timing":a.reps>=7,"threshold":{"cold_recovery_us_per_call":a.threshold_us,"scoreboard_must_reduce":True,"actual_async_sass_required":True},"arms":{}}
    if a.ncu:
      if art is None:art=a.out.parent/"artifacts";art.mkdir(parents=True,exist_ok=True)
      result["ncu"]={name:_ncu(bi,sym,art/f"ncu-{name}.csv.txt") for name,sym in zip(names,(CONTROL,ONE,TWO))}
    def metric(arm,name):
      row=next((x for x in result.get("ncu",{}).get(arm,{}).get("rows",[]) if x["metric"]==name),None);return None if row is None else float(row["value"].replace(",",""))
    for name in names[1:]:
      cold=med["control"]["cold_us"]-med[name]["cold_us"];hot=med["control"]["hot_us"]-med[name]["hot_us"];cs=metric("control","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct");ns=metric(name,"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct");score_ok=True if cs is None else ns is not None and ns<cs;async_ok=sass[name]["async_global_to_shared"]>0
      passed=exact and async_ok and cold>=a.threshold_us and score_ok
      result["arms"][name]={"hot_recovery_us":hot,"cold_recovery_us":cold,"actual_async_sass":async_ok,"scoreboard_control_pct":cs,"scoreboard_candidate_pct":ns,"scoreboard_reduced":score_ok,"verdict":("ADVANCE_TO_PRODUCTION_WALL" if passed else "STOP") if a.reps>=7 else "COUNTER_ONLY_NOT_PROMOTION_GRADE"}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0 if exact and all(result["arms"][x]["actual_async_sass"] for x in names[1:]) else 5


if __name__=="__main__":raise SystemExit(main())
