#!/usr/bin/env python3
"""Conservative end-to-end gate/up gate for U4Z8.

Control is the installed fused Q4_K gate/up kernel. Candidate deliberately uses
two already-qualified U4Z8 projection kernels plus a separate SiLU*up kernel.
Passing this conservative construction is sufficient to justify a fused U4Z8
emitter; failing it does not reject fusion. No production path is changed.
"""
from __future__ import annotations

import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGateUpLaneMap, _half4_lane,
  _silu_uop, _warp_reduce_sum_staged, q4k_g3_lanemap_gemv_w1w3_kernel)
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
import nv_u4z8_g64_p256_ffn_projection_gate as proj

ROWS, K, CW, UW, ROT = proj.ROWS, proj.K, proj.CW, proj.UW, 16
CUDA_BIN = "/usr/local/cuda-13.2/bin"


FUSED = "u4z8_g64_p256_w1w3fused16_12288_4096"


def _fused_u4():
  lm=Q4KGateUpLaneMap(k=K,n=ROWS);lm.validate()
  def kernel(out:UOp,gate:UOp,up:UOp,x:UOp)->UOp:
    row,lane=UOp.special(ROWS,"gidx0"),UOp.special(32,"lidx0")
    part=LanePartition(lane,lane_extent=lm.lane_extent,words_per_group=lm.words_per_group)
    lblk=UOp.range(lm.blocks_per_group,0,axis_type=AxisType.REDUCE);blk=part.block_group*lm.blocks_per_group+lblk
    bg=(row*lm.k_blocks+blk)*34;hg=gate.index(bg).load(dtype=dtypes.uint32.vec(2));hu=up.index(bg).load(dtype=dtypes.uint32.vec(2))
    cg,cu=UOp.const(dtypes.float32,0.0),UOp.const(dtypes.float32,0.0)
    for gp in range(4):
      qg,qw=gate[bg+2+gp*8+part.word_col],up[bg+2+gp*8+part.word_col]
      for pm in range(2):
        group=2*gp+pm
        sg=hg.gep(gp//2).rshift((gp%2)*16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
        su=hu.gep(gp//2).rshift((gp%2)*16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
        pg=qg.rshift(pm*4).bitwise_and(0x0F0F0F0F);pu=qw.rshift(pm*4).bitwise_and(0x0F0F0F0F)
        xv=x.index(blk*256+group*32+part.word_col*4).load(dtype=dtypes.float16.vec(4))
        for nib in range(4):
          vg=pg.rshift(nib*8).bitwise_and(15).cast(dtypes.float32)-8.0
          vu=pu.rshift(nib*8).bitwise_and(15).cast(dtypes.float32)-8.0
          xx=_half4_lane(xv,nib);cg=cg+sg*vg*xx;cu=cu+su*vu*xx
    ag=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG);au=UOp.placeholder((1,),dtypes.float32,21,addrspace=AddrSpace.REG)
    init=ag[0].store(0.0);init=au.after(init)[0].store(0.0);ag,au=ag.after(init),au.after(init)
    ug=ag[0].store(ag.after(lblk)[0]+cg);uu=au.after(ug)[0].store(au.after(lblk)[0]+cu).end(lblk)
    tg=_warp_reduce_sum_staged(ag.after(uu)[0],part.lane,part.lane_extent,90)
    tu=_warp_reduce_sum_staged(au.after(uu)[0],part.lane,part.lane_extent,95)
    return out[row].store((_silu_uop(tg)*tu).cast(dtypes.float16)).sink(arg=KernelInfo(name=FUSED,opts_to_apply=()))
  return kernel


def sources() -> tuple[str, str]:
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  p = UOp.placeholder
  ctrl = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=True)(
    p((ROWS,), dtypes.float16, 0), p((CW,), dtypes.uint32, 1), p((CW,), dtypes.uint32, 2), p((K,), dtypes.float16, 3))
  cs = next(x.arg for x in to_program(ctrl, ren).src if x.op is Ops.SOURCE)
  cs = cs[cs.index('extern "C" __global__'):]
  _, us = proj.render()
  fu=_fused_u4()(p((ROWS,),dtypes.float16,0),p((UW,),dtypes.uint32,1),p((UW,),dtypes.uint32,2),p((K,),dtypes.float16,3))
  fs=next(x.arg for x in to_program(fu,ren).src if x.op is Ops.SOURCE);fs=fs[fs.index('extern "C" __global__'):]
  return cs, us+"\n"+fs


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }
__CTRL__
__U4__
extern "C" __global__ void u4_gate_up_finish(half *o,const float *g,const float *u) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<12288){float x=g[i];o[i]=__float2half_rn((x/(1.0f+exp2f(x*(-1.4426950408889634f))))*u[i]);} }
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static void control(half*o,uint32_t*g,uint32_t*u,half*x,cudaStream_t s=0){q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288,32,0,s>>>(o,g,u,x);}
static void candidate(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,cudaStream_t s=0){u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_12288_4096<<<12288,32,0,s>>>(gf,g,x,z);u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_12288_4096<<<12288,32,0,s>>>(uf,u,x,z);u4_gate_up_finish<<<48,256,0,s>>>(o,gf,uf);}
static double tc(half*o,uint32_t*g,uint32_t*u,half*x,int n,cudaEvent_t a,cudaEvent_t b){cudaEventRecord(a);for(int i=0;i<n;i++)control(o,g,u,x);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
static double tu(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,int n,cudaEvent_t a,cudaEvent_t b){cudaEventRecord(a);for(int i=0;i<n;i++)candidate(o,gf,uf,g,u,x,z);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):32,reps=ac>2?atoi(av[2]):9;half *oc,*ou,*x;float *gf,*uf,*z;uint32_t *gc,*uc,*gu,*uu;ck(cudaMalloc(&oc,24576),"oc");ck(cudaMalloc(&ou,24576),"ou");ck(cudaMalloc(&gf,49152),"gf");ck(cudaMalloc(&uf,49152),"uf");ck(cudaMalloc(&z,49152),"z");ck(cudaMalloc(&x,8192),"x");ck(cudaMalloc(&gc,(size_t)ROT*CW*4),"gc");ck(cudaMalloc(&uc,(size_t)ROT*CW*4),"uc");ck(cudaMalloc(&gu,(size_t)ROT*UW*4),"gu");ck(cudaMalloc(&uu,(size_t)ROT*UW*4),"uu");cudaMemset(gc,0,(size_t)ROT*CW*4);cudaMemset(uc,0,(size_t)ROT*CW*4);cudaMemset(gu,0,(size_t)ROT*UW*4);cudaMemset(uu,0,(size_t)ROT*UW*4);cudaMemset(x,0,8192);cudaMemset(z,0,49152);control(oc,gc,uc,x);candidate(ou,gf,uf,gu,uu,x,z);ck(cudaDeviceSynchronize(),"warm");unsigned char hc[24576],hu[24576];cudaMemcpy(hc,oc,24576,cudaMemcpyDeviceToHost);cudaMemcpy(hu,ou,24576,cudaMemcpyDeviceToHost);int exact=!memcmp(hc,hu,24576);printf("zero_fixture_bitwise=%d\n",exact);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int r=0;r<reps;r++){double c=0,u=0;for(int i=0;i<n;i++){int j=i%ROT;c+=tc(oc,gc+(size_t)j*CW,uc+(size_t)j*CW,x,1,a,b);u+=tu(ou,gf,uf,gu+(size_t)j*UW,uu+(size_t)j*UW,x,z,1,a,b);}printf("rep=%d control_us=%.6f candidate_us=%.6f\n",r,c/n,u/n);}return exact?0:5;}
'''


def main() -> int:
  ap=argparse.ArgumentParser();ap.add_argument("--passes",type=int,default=32);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
  ctrl,u4=sources(); text=HARNESS.replace("__CTRL__",ctrl).replace("__U4__",u4)
  text=text.replace('static void candidate(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,cudaStream_t s=0){',
    'static void fused(half*o,uint32_t*g,uint32_t*u,half*x,cudaStream_t s=0){u4z8_g64_p256_w1w3fused16_12288_4096<<<12288,32,0,s>>>(o,g,u,x);}\nstatic void candidate(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,cudaStream_t s=0){')
  text=text.replace('static double tu(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,int n,cudaEvent_t a,cudaEvent_t b){',
    'static double tf(half*o,uint32_t*g,uint32_t*u,half*x,int n,cudaEvent_t a,cudaEvent_t b){cudaEventRecord(a);for(int i=0;i<n;i++)fused(o,g,u,x);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}\nstatic double tu(half*o,float*gf,float*uf,uint32_t*g,uint32_t*u,half*x,float*z,int n,cudaEvent_t a,cudaEvent_t b){')
  text=text.replace('int main(int ac,char**av){', 'static uint32_t step(uint32_t&s){s=1664525u*s+1013904223u;return s;}\nstatic void fillu(uint32_t*w,uint32_t seed){uint32_t s=seed;const uint16_t sc[4]={0x1000,0x1400,0x1800,0x1c00};for(size_t b=0;b<(size_t)12288*16;b++){size_t z=b*34;w[z]=uint32_t(sc[b&3])|(uint32_t(sc[(b+1)&3])<<16);w[z+1]=uint32_t(sc[(b+2)&3])|(uint32_t(sc[(b+3)&3])<<16);for(int i=2;i<34;i++)w[z+i]=step(s);}}\nint main(int ac,char**av){')
  text=text.replace('half *oc,*ou,*x;', 'half *oc,*ou,*of,*x;').replace('ck(cudaMalloc(&ou,24576),"ou");', 'ck(cudaMalloc(&ou,24576),"ou");ck(cudaMalloc(&of,24576),"of");')
  text=text.replace('uint32_t *gc,*uc,*gu,*uu;', 'uint32_t *gc,*uc,*gu,*uu,*guf,*uuf;')
  text=text.replace('ck(cudaMalloc(&uu,(size_t)ROT*UW*4),"uu");', 'ck(cudaMalloc(&uu,(size_t)ROT*UW*4),"uu");ck(cudaMalloc(&guf,(size_t)ROT*UW*4),"guf");ck(cudaMalloc(&uuf,(size_t)ROT*UW*4),"uuf");')
  text=text.replace('cudaMemset(uu,0,(size_t)ROT*UW*4);', 'cudaMemset(uu,0,(size_t)ROT*UW*4);cudaMemset(guf,0,(size_t)ROT*UW*4);cudaMemset(uuf,0,(size_t)ROT*UW*4);')
  text=text.replace('cudaMemset(x,0,8192);cudaMemset(z,0,49152);', 'uint32_t*hg=(uint32_t*)malloc((size_t)UW*4);uint32_t*huu=(uint32_t*)malloc((size_t)UW*4);half*hx=(half*)malloc(8192);fillu(hg,0x12345678);fillu(huu,0x9abcdef0);for(int i=0;i<4096;i++)hx[i]=__float2half(float((i%127)-63)/1024.0f);for(int j=0;j<ROT;j++){cudaMemcpy(gu+(size_t)j*UW,hg,(size_t)UW*4,cudaMemcpyHostToDevice);cudaMemcpy(uu+(size_t)j*UW,huu,(size_t)UW*4,cudaMemcpyHostToDevice);cudaMemcpy(guf+(size_t)j*UW,hg,(size_t)UW*4,cudaMemcpyHostToDevice);cudaMemcpy(uuf+(size_t)j*UW,huu,(size_t)UW*4,cudaMemcpyHostToDevice);}cudaMemcpy(x,hx,8192,cudaMemcpyHostToDevice);cudaMemset(z,0,49152);free(hg);free(huu);free(hx);')
  text=text.replace('candidate(ou,gf,uf,gu,uu,x,z);ck(cudaDeviceSynchronize(),"warm");', 'candidate(ou,gf,uf,gu,uu,x,z);fused(of,guf,uuf,x);ck(cudaDeviceSynchronize(),"warm");')
  text=text.replace('unsigned char hc[24576],hu[24576];', 'unsigned char hc[24576],hu[24576],hf[24576];').replace('cudaMemcpy(hu,ou,24576,cudaMemcpyDeviceToHost);', 'cudaMemcpy(hu,ou,24576,cudaMemcpyDeviceToHost);cudaMemcpy(hf,of,24576,cudaMemcpyDeviceToHost);')
  text=text.replace('int exact=!memcmp(hc,hu,24576);printf("zero_fixture_bitwise=%d\\n",exact);', 'int raw=!memcmp(hc,hu,24576),bad=0;float ma=0;half*ahu=(half*)hu;half*ahf=(half*)hf;for(int i=0;i<12288;i++){float va=__half2float(ahu[i]),vb=__half2float(ahf[i]),d=fabsf(va-vb);if(!isfinite(va)||!isfinite(vb)){bad++;continue;}ma=fmaxf(ma,d);bad+=d>0.002f;}int exact=raw||(bad==0);printf("composition_pass=%d raw_bitwise=%d bad=%d max_abs=%.9g\\n",exact,raw,bad,ma);')
  text=text.replace('double c=0,u=0;', 'double c=0,u=0,f=0;').replace('u+=tu(ou,gf,uf,gu+(size_t)j*UW,uu+(size_t)j*UW,x,z,1,a,b);', 'u+=tu(ou,gf,uf,gu+(size_t)j*UW,uu+(size_t)j*UW,x,z,1,a,b);f+=tf(of,guf+(size_t)j*UW,uuf+(size_t)j*UW,x,1,a,b);')
  text=text.replace('candidate_us=%.6f\\n",r,c/n,u/n', 'candidate_us=%.6f fused_us=%.6f\\n",r,c/n,u/n,f/n')
  text=re.sub(r"\bROT\b",str(ROT),text);text=re.sub(r"\bCW\b",str(CW),text);text=re.sub(r"\bUW\b",str(UW),text)
  with tempfile.TemporaryDirectory(prefix="nv_u4_gate_up_chain_") as td:
    src,binp=Path(td)/"gate.cu",Path(td)/"gate";src.write_text(text);env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    b=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17",str(src),"-o",str(binp)],capture_output=True,text=True,env=env)
    if b.returncode:print(b.stderr[-10000:],file=sys.stderr);return 3
    run=subprocess.run([str(binp),str(a.passes),str(a.reps)],capture_output=True,text=True);print(run.stdout)
    if run.returncode not in (0,5):print(run.stderr[-5000:],file=sys.stderr);return 4
    exact="composition_pass=1" in run.stdout;c=[];u=[];f=[];sm=re.search(r"raw_bitwise=(\d+) bad=(\d+) max_abs=([0-9.eE+-]+)",run.stdout)
    for m in re.finditer(r"control_us=([0-9.]+) candidate_us=([0-9.]+) fused_us=([0-9.]+)",run.stdout):c.append(float(m[1]));u.append(float(m[2]));f.append(float(m[3]))
    cm,um,fm=statistics.median(c),statistics.median(u),statistics.median(f);result={"schema":"tinygrad.nv_u4z8_gate_up_chain.v3","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"construction":"installed fused Q4_K versus conservative two U4Z8 projections plus separate finish and fused U4Z8","semantic_scope":"nonzero legal U4Z8 fixture; fused result compared with two independently oracle-qualified projections plus finish; FP16 abs <= 0.002","composition":{"pass":exact,"raw_bitwise":bool(int(sm[1])) if sm else None,"bad":int(sm[2]) if sm else None,"max_abs":float(sm[3]) if sm else None},"timing":{"unit":"us_per_pair_per_call_sync","control":c,"conservative":u,"fused":f,"control_median":cm,"conservative_median":um,"fused_median":fm,"conservative_recovery_us":cm-um,"fused_recovery_us":cm-fm},"verdict":"PASS_FUSED" if exact and fm<cm else "STOP_PERFORMANCE" if exact else "STOP_CORRECTNESS"};a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0 if exact else 5


if __name__=="__main__":raise SystemExit(main())
