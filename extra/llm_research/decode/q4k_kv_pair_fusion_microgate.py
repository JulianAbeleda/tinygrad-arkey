#!/usr/bin/env python3
"""Bit-exact CUDA-event gate for one Q4_K K/V dual-output producer.

The control runs the installed vector single-projection body twice, once per
matrix. The candidate keeps that body's exact per-lane arithmetic but owns two
accumulators and stores both results from one launch. This is a producer-side
rate/launch test only; it does not modify the model route or cache boundary.
"""
from __future__ import annotations

import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.q4k_kv_pair import emit_q4k_kv_pair_vector
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
def _render(ren:CUDARenderer, rows:int, k:int) -> tuple[str, str]:
  words = (k // Q4_K_BLOCK_ELEMS) * 36 * rows
  p = UOp.placeholder
  single = q4k_g3_lanemap_gemv_kernel(rows, k, load_style="vector")(
    p((rows,), dtypes.float32, 0), p((words,), dtypes.uint32, 1), p((k,), dtypes.float16, 2))
  pair = emit_q4k_kv_pair_vector(rows, k)(p((rows,), dtypes.float32, 0), p((rows,), dtypes.float32, 1),
    p((words,), dtypes.uint32, 2), p((words,), dtypes.uint32, 3), p((k,), dtypes.float16, 4))
  def src(u:UOp) -> str:
    text = next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]
  return src(single), src(pair)


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x,y,z,w}; return r; }

__SINGLE__
__PAIR__

static void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); } }

static double time_control(float* out, unsigned int* wk, unsigned int* wv, half* x, int passes) {
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
  for (int i=0;i<passes;i++) {
    SINGLE_NAME<<<ROWS_ARG,32>>>(out, wk, x);
    SINGLE_NAME<<<ROWS_ARG,32>>>(out+ROWS_ARG, wv, x);
  }
  cudaEventRecord(e); ck(cudaDeviceSynchronize(), "control sync"); float ms=0; cudaEventElapsedTime(&ms,s,e);
  cudaEventDestroy(s); cudaEventDestroy(e); return ms*1000.0/passes;
}

static double time_candidate(float* out, unsigned int* wk, unsigned int* wv, half* x, int passes) {
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
  for (int i=0;i<passes;i++) PAIR_NAME<<<ROWS_ARG,32>>>(out,out+ROWS_ARG,wk,wv,x);
  cudaEventRecord(e); ck(cudaDeviceSynchronize(), "candidate sync"); float ms=0; cudaEventElapsedTime(&ms,s,e);
  cudaEventDestroy(s); cudaEventDestroy(e); return ms*1000.0/passes;
}

int main(int argc, char** argv) {
  int passes=argc>1?atoi(argv[1]):500, reps=argc>2?atoi(argv[2]):9;
  float *ctrl=nullptr,*cand=nullptr; unsigned int *wk=nullptr,*wv=nullptr; half *x=nullptr;
  ck(cudaMalloc(&ctrl,2*ROWS_ARG*sizeof(float)),"ctrl"); ck(cudaMalloc(&cand,2*ROWS_ARG*sizeof(float)),"cand");
  ck(cudaMalloc(&wk,WORDS_ARG*sizeof(unsigned int)),"wk"); ck(cudaMalloc(&wv,WORDS_ARG*sizeof(unsigned int)),"wv");
  ck(cudaMalloc(&x,K_ARG*sizeof(half)),"x");
  unsigned int *hwk=(unsigned int*)malloc(WORDS_ARG*sizeof(unsigned int));
  unsigned int *hwv=(unsigned int*)malloc(WORDS_ARG*sizeof(unsigned int)); half *hx=(half*)malloc(K_ARG*sizeof(half));
  for (int i=0;i<WORDS_ARG;i++) { hwk[i]=(unsigned int)((i*2654435761u)^0x9e3779b9u); hwv[i]=(unsigned int)((i*2246822519u)^0x85ebca6bu); }
  for (int i=0;i<K_ARG;i++) hx[i]=__float2half(((i%257)-128)*0.03125f);
  ck(cudaMemcpy(wk,hwk,WORDS_ARG*sizeof(unsigned int),cudaMemcpyHostToDevice),"wk copy");
  ck(cudaMemcpy(wv,hwv,WORDS_ARG*sizeof(unsigned int),cudaMemcpyHostToDevice),"wv copy");
  ck(cudaMemcpy(x,hx,K_ARG*sizeof(half),cudaMemcpyHostToDevice),"x copy"); free(hwk); free(hwv); free(hx);
  SINGLE_NAME<<<ROWS_ARG,32>>>(ctrl,wk,x); SINGLE_NAME<<<ROWS_ARG,32>>>(ctrl+ROWS_ARG,wv,x);
  PAIR_NAME<<<ROWS_ARG,32>>>(cand,cand+ROWS_ARG,wk,wv,x); ck(cudaDeviceSynchronize(),"warmup");
  float *hc=(float*)malloc(2*ROWS_ARG*sizeof(float)), *hp=(float*)malloc(2*ROWS_ARG*sizeof(float));
  ck(cudaMemcpy(hc,ctrl,2*ROWS_ARG*sizeof(float),cudaMemcpyDeviceToHost),"ctrl copy");
  ck(cudaMemcpy(hp,cand,2*ROWS_ARG*sizeof(float),cudaMemcpyDeviceToHost),"cand copy");
  int mismatch=0; for (int i=0;i<2*ROWS_ARG;i++) mismatch += memcmp(&hc[i],&hp[i],sizeof(float))!=0;
  printf("mismatched_words=%d bitwise_identical=%d\n",mismatch,mismatch==0); free(hc); free(hp);
  for (int r=0;r<reps;r++) {
    double c=time_control(ctrl,wk,wv,x,passes), p=time_candidate(cand,wk,wv,x,passes);
    printf("rep=%d control_pair=%.6f candidate_pair=%.6f\n",r,c,p);
  }
  cudaFree(ctrl); cudaFree(cand); cudaFree(wk); cudaFree(wv); cudaFree(x); return mismatch?5:0;
}
'''


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--rows",type=int,default=1024); ap.add_argument("--k",type=int,default=4096)
  ap.add_argument("--passes",type=int,default=500); ap.add_argument("--reps",type=int,default=9); ap.add_argument("--out",type=Path,required=True)
  args=ap.parse_args(); rows,k=args.rows,args.k
  ren=CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False); single,pair=_render(ren,rows,k)
  single_name=f"q4k_g3_lanemap_gemv_vec_{rows}_{k}"; pair_name=f"q4k_g3_lanemap_gemv_pair_vec_{rows}_{k}"
  words=(k//Q4_K_BLOCK_ELEMS)*36*rows
  cu=(HARNESS.replace("__SINGLE__",single).replace("__PAIR__",pair).replace("SINGLE_NAME",single_name)
      .replace("PAIR_NAME",pair_name).replace("ROWS_ARG",str(rows)).replace("K_ARG",str(k)).replace("WORDS_ARG",str(words)))
  with tempfile.TemporaryDirectory(prefix="q4k_kv_pair_") as td:
    src=Path(td)/"gate.cu"; binary=Path(td)/"gate"; src.write_text(cu)
    env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    build=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(src),"-o",str(binary)],
                         capture_output=True,text=True,env=env)
    if build.returncode: print(build.stderr[-8000:],file=sys.stderr); return 3
    run=subprocess.run([str(binary),str(args.passes),str(args.reps)],capture_output=True,text=True)
    print(run.stdout.strip())
    if run.returncode not in (0,5): print(run.stderr[-4000:],file=sys.stderr); return 4
    mm=re.search(r"mismatched_words=(\d+) bitwise_identical=(\d+)",run.stdout); cv,pv=[],[]
    for line in run.stdout.splitlines():
      if (m:=re.search(r"control_pair=([0-9.]+) candidate_pair=([0-9.]+)",line)): cv.append(float(m.group(1))); pv.append(float(m.group(2)))
    cm,pm=statistics.median(cv),statistics.median(pv)
    result={"schema":"tinygrad.q4k_kv_pair_fusion_microgate.v1",
      "commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
      "shape":{"rows":rows,"k":k,"q4_pairs_per_token":18},"passes":args.passes,"reps":args.reps,
      "bitwise_identical":bool(mm and int(mm.group(2))),"mismatched_words":int(mm.group(1)) if mm else None,
      "timing":{"unit":"us_per_pair_cuda_event","control_samples":cv,"candidate_samples":pv,
        "control_median":cm,"candidate_median":pm,"recovery_us_per_pair":cm-pm,
        "projected_18_pair_recovery_us":(cm-pm)*18,"candidate_over_control":pm/cm},
      "ptxas":build.stderr.strip().splitlines(),"verdict":"ADVANCE" if mm and int(mm.group(2)) and pm<cm and (cm-pm)*18>=15 else "NO_GO"}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["bitwise_identical"] else 5


if __name__=="__main__": raise SystemExit(main())
